#include "object_detection.h"
#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

ObjectDetection::ObjectDetection(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "目标检测"), config_(*config) {

  AlgorConfig algor_config;
  
  // 使用配置参数，如果没有提供则使用默认值
  if (config) {
    algor_config.algorName_ = config->det_algor_name;
    algor_config.model_path = config->car_det_model_path;
    algor_config.img_size = config->det_img_size;
    algor_config.conf_thresh = config->det_conf_thresh;
    algor_config.iou_thresh = config->det_iou_thresh;
    algor_config.max_batch_size = config->det_max_batch_size;
    algor_config.min_opt = config->det_min_opt;
    algor_config.mid_opt = config->det_mid_opt;
    algor_config.max_opt = config->det_max_opt;
    algor_config.is_ultralytics = config->det_is_ultralytics;
    algor_config.gpu_id = config->det_gpu_id;
  } else {
    // 默认配置
    algor_config.algorName_ = "object_detect";
    algor_config.model_path = "car_detect.onnx";
    algor_config.img_size = 640;
    algor_config.conf_thresh = 0.25f;
    algor_config.iou_thresh = 0.2f;
    algor_config.max_batch_size = BATCH_SIZE;
    algor_config.min_opt = 1;
    algor_config.mid_opt = 16;
    algor_config.max_opt = 32;
    algor_config.is_ultralytics = 1;
    algor_config.gpu_id = 0;
  }

  // 为每个线程创建独立的车辆检测实例
  car_detect_instances_.resize(num_threads_);
  for (int i = 0; i < num_threads_; ++i) {
    car_detect_instances_[i] = std::unique_ptr<xtkj::IDetect>(xtkj::createDetect());
    car_detect_instances_[i]->init(algor_config);
  }
  
  // 初始化行人检测器（如果启用）
  if (config && config->enable_pedestrian_detect) {
    personal_detect_instances_.resize(num_threads_);
    AlgorConfig person_config = algor_config; // 复制车辆检测配置
    person_config.model_path = config->pedestrian_det_model_path;
    
    for (int i = 0; i < num_threads_; ++i) {
      personal_detect_instances_[i] = std::unique_ptr<xtkj::IDetect>(xtkj::createDetect());
      personal_detect_instances_[i]->init(person_config);
    }
  }
  
  // 初始化CUDA状态
  try {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
      // 预分配GPU内存以提高性能
      gpu_src_cache_.create(1080, 1920, CV_8UC3); // 假设最大输入尺寸
      gpu_dst_cache_.create(1080, 1920, CV_8UC3); // 输出尺寸（resize后可能变化）
      cuda_available_ = true;
      std::cout << "✅ CUDA已启用，目标检测resize将使用GPU加速" << std::endl;
    } else {
      cuda_available_ = false;
      std::cout << "⚠️ 未检测到CUDA设备，目标检测resize将使用CPU" << std::endl;
    }
  } catch (const cv::Exception& e) {
    cuda_available_ = false;
    std::cerr << "⚠️ CUDA初始化失败: " << e.what() << "，目标检测resize将使用CPU" << std::endl;
  }
  
  // 清空输出监控记录
  recent_output_frames_.clear();
  
  std::cout << "🔍 目标检测模块初始化完成（" << num_threads_ << "个线程，批量大小: " << BATCH_SIZE << "）" << std::endl;
}

// 重写 start 方法
void ObjectDetection::start() {
  // 调用基类的启动方法
  ImageProcessor::start();
  
  // 重置状态
  next_expected_frame_ = 0;
  order_thread_running_.store(false);  // 延迟启动顺序输出线程
  
  std::cout << "✅ 目标检测模块已启动，将在首次获取结果时启动顺序输出线程" << std::endl;
}

// 重写 stop 方法
void ObjectDetection::stop() {
  // 先停止基类的工作线程
  ImageProcessor::stop();
  
  // 停止顺序输出线程
  if (order_thread_running_.load()) {
    order_thread_running_.store(false);
    order_cv_.notify_all();
    if (ordered_output_thread_.joinable()) {
      ordered_output_thread_.join();
    }
  }
  
  // 清空顺序缓冲区
  {
    std::lock_guard<std::mutex> lock(order_mutex_);
    ordered_buffer_.clear();
  }
  
  // 清空输出监控记录
  {
    std::lock_guard<std::mutex> monitor_lock(output_monitor_mutex_);
    recent_output_frames_.clear();
  }
  
  std::cout << "✅ 目标检测模块已停止，顺序输出线程已关闭" << std::endl;
}

// 重写 get_processed_image 方法，启动顺序输出线程（延迟启动）
bool ObjectDetection::get_processed_image(ImageDataPtr &image) {
  // 延迟启动顺序输出线程
  if (!order_thread_running_.load()) {
    order_thread_running_.store(true);
    ordered_output_thread_ = std::thread(&ObjectDetection::ordered_output_thread_func, this);
    std::cout << "✅ 目标检测顺序输出线程已启动" << std::endl;
  }
  
  // 调用基类的方法从输出队列获取图像
  return ImageProcessor::get_processed_image(image);
}

// 重写工作线程函数以支持批量处理
void ObjectDetection::worker_thread_func(int thread_id) {
  std::cout << "🔄 " << processor_name_ << "批量工作线程 " << thread_id << " 启动"
            << std::endl;

  const size_t BATCH_SIZE = 32; // 批量处理大小
  std::vector<ImageDataPtr> batch_images;
  batch_images.reserve(BATCH_SIZE);

  while (running_.load()) {
    batch_images.clear();
    
    // 第一步：阻塞等待第一个图像
    ImageDataPtr first_image;
    input_queue_.wait_and_pop(first_image);
    
    // 检查停止信号
    if (!first_image) {
      if (!running_.load()) {
        break;
      }
      continue;
    }
    
    batch_images.push_back(first_image);
    
    // 第二步：非阻塞方式收集剩余图像，带超时机制
    ImageDataPtr image;
    auto collection_start = std::chrono::high_resolution_clock::now();
    const auto timeout_ms = std::chrono::milliseconds(10); // 10ms超时
    
    while (batch_images.size() < BATCH_SIZE && running_.load()) {
      if (input_queue_.try_pop(image)) {
        if (image) {
          batch_images.push_back(image);
        }
      } else {
        // 检查是否超时
        auto now = std::chrono::high_resolution_clock::now();
        if (now - collection_start > timeout_ms) {
          std::cout << "⏱️ [线程 " << thread_id << "] 批量收集超时，当前批次: " 
                    << batch_images.size() << std::endl;
          break;
        }
        // 短暂休眠，避免占用过多CPU
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    
    std::cout << "🔄 [线程 " << thread_id << "] 开始批量处理 " 
              << batch_images.size() << " 张图像" << std::endl;
    
    // 第三步：批量处理
    process_images_batch(batch_images, thread_id);
    
    // 第四步：将所有处理结果添加到顺序缓冲区
    for (auto& processed_image : batch_images) {
      ordered_output_push(processed_image);
    }
    
    std::cout << "✅ [线程 " << thread_id << "] 批量处理完成，输出 " 
              << batch_images.size() << " 张图像" << std::endl;
  }
  
  std::cout << "🔄 " << processor_name_ << "批量工作线程 " << thread_id << " 退出"
            << std::endl;
}

// 批量处理图像
void ObjectDetection::process_images_batch(std::vector<ImageDataPtr>& images, int thread_id) {
  if (images.empty()) {
    return;
  }
  
  // 检查线程ID是否有效
  if (thread_id < 0 || thread_id >= car_detect_instances_.size()) {
    std::cerr << "❌ 批量处理：无效的线程ID: " << thread_id << std::endl;
    return;
  }

  // 使用该线程专属的模型实例
  auto& car_detect_instance = car_detect_instances_[thread_id];
  if (!car_detect_instance) {
    std::cerr << "❌ 批量处理：线程 " << thread_id << " 的车辆检测实例无效" << std::endl;
    return;
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // 批量预处理：为每个图像调用 on_processing_start
  for (auto& image : images) {
    on_processing_start(image, thread_id);
  }
  
  auto preprocess_end = std::chrono::high_resolution_clock::now();
  auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - start_time);
  
  try {
    // 准备批量检测的数据
    std::vector<cv::Mat> mats;
    mats.reserve(images.size());
    
    // 准备批量输入数据
    for (auto& image : images) {
      if (!image) continue;
      
      // 等待mask后处理完成（如果需要）
      bool seg_enabled = config_.enable_segmentation;
      bool mask_enabled = config_.enable_mask_postprocess;
      
      if (seg_enabled && mask_enabled) {
        while (!image->mask_postprocess_completed) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      } else {
        image->mask_postprocess_completed = true;
      }
      
      // 准备ROI裁剪后的图像用于批量检测
      cv::Mat cropped_image = (image->imageMat)(image->roi);
      mats.push_back(cropped_image);
    }
    
    if (mats.empty()) {
      // 标记所有图像完成避免死锁
      for (auto& image : images) {
        image->detection_completed = true;
      }
      return;
    }
    
    // 批量目标检测处理
    auto det_start = std::chrono::high_resolution_clock::now();
    
    // 车辆检测 - 批量处理
    std::vector<detect_result_group_t> car_outs(mats.size());
    std::vector<detect_result_group_t*> car_out_ptrs;
    car_out_ptrs.reserve(mats.size());
    for (auto& out : car_outs) {
      car_out_ptrs.push_back(&out);
    }
    
    car_detect_instance->forward(mats, car_out_ptrs.data());
    
    // 处理车辆检测结果
    for (size_t i = 0; i < images.size() && i < car_outs.size(); ++i) {
      auto& image = images[i];
      auto& car_out = car_outs[i];
      
      if (car_out.count > 0) {
        for (int j = 0; j < car_out.count; ++j) {
          detect_result_t &result = car_out.results[j];
          image->detection_results.push_back({
            result.box.left + image->roi.x, result.box.top + image->roi.y,
            result.box.right + image->roi.x, result.box.bottom + image->roi.y,
            result.prop, result.cls_id, result.track_id});
        }
      }
    }
    
    // 行人检测 - 批量处理（如果启用）
    if (!personal_detect_instances_.empty() && thread_id < personal_detect_instances_.size()) {
      auto& personal_detect_instance = personal_detect_instances_[thread_id];
      if (personal_detect_instance) {
        std::vector<detect_result_group_t> person_outs(mats.size());
        std::vector<detect_result_group_t*> person_out_ptrs;
        person_out_ptrs.reserve(mats.size());
        for (auto& out : person_outs) {
          person_out_ptrs.push_back(&out);
        }
        
        personal_detect_instance->forward(mats, person_out_ptrs.data());
        
        // 处理行人检测结果
        for (size_t i = 0; i < images.size() && i < person_outs.size(); ++i) {
          auto& image = images[i];
          auto& person_out = person_outs[i];
          
          if (person_out.count > 0) {
            for (int j = 0; j < person_out.count; ++j) {
              detect_result_t &result = person_out.results[j];
              image->detection_results.push_back({
                result.box.left + image->roi.x, result.box.top + image->roi.y,
                result.box.right + image->roi.x, result.box.bottom + image->roi.y,
                result.prop, 1, result.track_id}); // 行人检测类别ID设置为1
            }
          }
        }
      }
    }
    
    auto det_end = std::chrono::high_resolution_clock::now();
    auto det_duration = std::chrono::duration_cast<std::chrono::milliseconds>(det_end - det_start);
    
    // 更新性能统计
    total_processed_images_.fetch_add(images.size());
    total_processing_time_ms_.fetch_add((preprocess_duration + det_duration).count());
    
    std::cout << "⚡ [线程 " << thread_id << "] 批量目标检测性能统计 - 预处理: " 
              << preprocess_duration.count() << "ms, 推理: " << det_duration.count() 
              << "ms, 总计: " << (preprocess_duration + det_duration).count() 
              << "ms, 处理 " << images.size() << " 张图像" << std::endl;
    
    // 批量后处理：快速结果分配和标记完成
    for (size_t i = 0; i < images.size(); ++i) {
      if (images[i]) {
        images[i]->detection_completed = true;
        // 调用后处理
        on_processing_complete(images[i], thread_id);
      }
    }
    
  } catch (const std::exception& e) {
    std::cerr << "❌ 目标检测批量处理失败: " << e.what() << std::endl;
    // 标记所有图像检测完成避免阻塞
    for (auto& image : images) {
      if (image) {
        image->detection_completed = true;
      }
    }
  }
}

// 将处理完成的图像添加到顺序缓冲区
void ObjectDetection::ordered_output_push(ImageDataPtr image) {
    std::lock_guard<std::mutex> lock(order_mutex_);
    ordered_buffer_[image->frame_idx] = image;
    order_cv_.notify_one();
}

// 顺序输出线程函数
void ObjectDetection::ordered_output_thread_func() {
  std::cout << "🔄 目标检测顺序输出线程启动" << std::endl;
  
  while (order_thread_running_.load() || !ordered_buffer_.empty()) {
    std::unique_lock<std::mutex> lock(order_mutex_);
    
    // 等待有数据可处理
    order_cv_.wait(lock, [this]() {
      return !ordered_buffer_.empty() || !order_thread_running_.load();
    });
    
    // 按顺序输出连续的帧
    while (!ordered_buffer_.empty()) {
      auto it = ordered_buffer_.find(next_expected_frame_);
      if (it != ordered_buffer_.end()) {
        // 找到了下一个期望的帧，输出它
        ImageDataPtr image = it->second;
        int64_t frame_idx = image->frame_idx;
        ordered_buffer_.erase(it);
        lock.unlock();
        
        // 推送到实际的输出队列
        output_queue_.push(image);
        
        // 更新输出监控记录
        {
          std::lock_guard<std::mutex> monitor_lock(output_monitor_mutex_);
          recent_output_frames_.push_back(frame_idx);
          if (recent_output_frames_.size() > OUTPUT_WINDOW_SIZE) {
            recent_output_frames_.pop_front();
          }
          
          // 定期验证输出顺序（每10帧一次）
          if (frame_idx % 10 == 0 && recent_output_frames_.size() > 1) {
            bool is_ordered = true;
            for (size_t i = 1; i < recent_output_frames_.size(); ++i) {
              if (recent_output_frames_[i] <= recent_output_frames_[i-1]) {
                is_ordered = false;
                break;
              }
            }
            std::cout << "🔍 目标检测输出顺序验证: " << (is_ordered ? "✅ 有序" : "❌ 乱序") 
                      << ", 当前帧: " << frame_idx << std::endl;
          }
        }
        
        // 更新下一个期望的帧序号
        next_expected_frame_++;
        
        // 重新加锁继续处理
        lock.lock();
      } else {
        // 下一个期望的帧还没到，等待
        break;
      }
    }
  }
  
  std::cout << "🔄 目标检测顺序输出线程结束" << std::endl;
}

void ObjectDetection::process_image(ImageDataPtr image, int thread_id) {
  // 这个方法现在主要用于向后兼容，实际批量处理在process_images_batch中
  if (!image) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }
  
  // 将单个图像包装成批量处理
  std::vector<ImageDataPtr> single_batch = {image};
  process_images_batch(single_batch, thread_id);
}

void ObjectDetection::on_processing_start(ImageDataPtr image, int thread_id) {
  // std::cout << "🎯 目标检测准备开始 (线程 " << thread_id << ")" << std::endl;
  int max_dim = std::max(image->width, image->height);
  if (max_dim > 1920) {
    // 如果图像尺寸超过1080p，使用CUDA缩小到1080p
    double scale = 1920.0 / max_dim;
    cv::Size new_size(static_cast<int>(image->width * scale), 
                      static_cast<int>(image->height * scale));
    
    if (cuda_available_) {
      try {
        std::lock_guard<std::mutex> lock(gpu_mutex_); // 保护GPU操作
        
        // 检查是否需要调整缓存大小
        if (gpu_src_cache_.rows < image->imageMat.rows || 
            gpu_src_cache_.cols < image->imageMat.cols) {
          gpu_src_cache_.create(image->imageMat.rows, image->imageMat.cols, CV_8UC3);
        }
        
        // 检查输出缓存大小
        if (gpu_dst_cache_.rows < new_size.height || 
            gpu_dst_cache_.cols < new_size.width) {
          gpu_dst_cache_.create(new_size.height, new_size.width, CV_8UC3);
        }
        
        // 上传到GPU
        cv::cuda::GpuMat gpu_src_roi = gpu_src_cache_(cv::Rect(0, 0, image->imageMat.cols, image->imageMat.rows));
        gpu_src_roi.upload(image->imageMat);
        
        // 在GPU上进行resize操作
        cv::cuda::GpuMat gpu_dst_roi = gpu_dst_cache_(cv::Rect(0, 0, new_size.width, new_size.height));
        cv::cuda::resize(gpu_src_roi, gpu_dst_roi, new_size, 0, 0, cv::INTER_LINEAR);
        
        // 下载回CPU
        gpu_dst_roi.download(image->parkingResizeMat);
        
      } catch (const cv::Exception& e) {
        // 如果CUDA操作失败，标记CUDA不可用并回退到CPU实现
        std::cerr << "⚠️ CUDA resize失败，禁用CUDA并回退到CPU: " << e.what() << std::endl;
        cuda_available_ = false;
        cv::resize(image->imageMat, image->parkingResizeMat, new_size, 0, 0, cv::INTER_LINEAR);
      }
    } else {
      // 使用CPU实现
      cv::resize(image->imageMat, image->parkingResizeMat, new_size, 0, 0, cv::INTER_LINEAR);
    }
  } else {
    // 否则保持原尺寸
    image->parkingResizeMat = image->imageMat.clone();
  }
}

void ObjectDetection::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  // std::cout << "🎯 目标检测处理完成 (线程 " << thread_id << ")" << std::endl;
}

void ObjectDetection::perform_object_detection(ImageDataPtr image,
                                               int thread_id) {
  // 保留单图接口，实际不再直接调用
}

ObjectDetection::~ObjectDetection() {
  // std::cout << "🔄 正在停止目标检测模块..." << std::endl;
  
  // 车辆检测实例数组已经是智能指针，无需手动删除
  car_detect_instances_.clear();
  
  // 行人检测实例数组也是智能指针，无需手动删除
  personal_detect_instances_.clear();
  
  // std::cout << "✅ 目标检测模块已停止" << std::endl;
}
