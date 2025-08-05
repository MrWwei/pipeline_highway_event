#include "semantic_segmentation.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "thread_safe_queue.h"

//析构函数
SemanticSegmentation::~SemanticSegmentation() {
  // 确保正确停止
  stop();
  
  // 释放所有模型实例
  for (auto& instance : road_seg_instances_) {
    if (instance) {
      releasePureTRTPPSeg(instance);
    }
  }
  road_seg_instances_.clear();
}

SemanticSegmentation::SemanticSegmentation(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "语义分割", 100, 100), // 输入队列改为100，输出队列保持100
      next_expected_frame_(0),
      order_thread_running_(false) {

  // 初始化输出监控
  recent_output_frames_.clear();

  // 初始化模型参数
  PPSegInitParameters init_params;
  
  // 使用配置参数，如果没有提供则使用默认值
  if (config) {
    init_params.model_path = config->seg_model_path;
    enable_seg_show_ = config->enable_seg_show;
    seg_show_image_path_ = config->seg_show_image_path;
  } else {
    // 默认配置
    init_params.model_path = "seg_model";
    enable_seg_show_ = false;
    seg_show_image_path_ = "./segmentation_results/";
  }

  // 为每个线程创建独立的模型实例
  road_seg_instances_.resize(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    road_seg_instances_[i] = CreatePureTRTPPSeg();
    int init_result = road_seg_instances_[i]->Init(init_params);
    if (init_result != 0) {
      std::cerr << "❌ 语义分割模型初始化失败，线程 " << i << std::endl;
    } else {
      std::cout << "✅ 语义分割模型初始化成功，线程 " << i << std::endl;
    }
  }
  
  // 初始化CUDA状态
  try {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
      // 预分配GPU内存以提高性能
      gpu_src_cache_.create(1080, 1920, CV_8UC3); // 假设最大输入尺寸
      gpu_dst_cache_.create(1024, 1024, CV_8UC3); // 目标尺寸
      cuda_available_ = true;
      std::cout << "✅ CUDA已启用，语义分割将使用GPU加速" << std::endl;
    } else {
      cuda_available_ = false;
      std::cout << "⚠️ 未检测到CUDA设备，语义分割将使用CPU" << std::endl;
    }
  } catch (const cv::Exception& e) {
    cuda_available_ = false;
    std::cerr << "⚠️ CUDA初始化失败: " << e.what() << "，将使用CPU" << std::endl;
  }
  
  std::cout << "✅ 语义分割模块初始化完成，支持 " << num_threads << " 个线程，每线程独立模型实例" << std::endl;
}

// 重写 start 方法
void SemanticSegmentation::start() {
  // 调用基类的启动方法
  ImageProcessor::start();
  
  // 重置状态
  next_expected_frame_.store(0);
  order_thread_running_.store(false);  // 延迟启动顺序输出线程
  
  std::cout << "✅ 语义分割模块已启动，将在首次获取结果时启动顺序输出线程" << std::endl;
}

// 重写 stop 方法
void SemanticSegmentation::stop() {
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
  
  std::cout << "✅ 语义分割模块已停止，顺序输出线程已关闭" << std::endl;
}

// 重写工作线程函数，支持批量处理
void SemanticSegmentation::worker_thread_func(int thread_id) {
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

// 重写 get_processed_image 方法，启动顺序输出线程（延迟启动）
bool SemanticSegmentation::get_processed_image(ImageDataPtr &image) {
  // 延迟启动顺序输出线程
  if (!order_thread_running_.load()) {
    order_thread_running_.store(true);
    ordered_output_thread_ = std::thread(&SemanticSegmentation::ordered_output_thread_func, this);
    std::cout << "✅ 语义分割顺序输出线程已启动" << std::endl;
  }
  
  // 调用基类的方法从输出队列获取图像
  return ImageProcessor::get_processed_image(image);
}

// 将处理完成的图像添加到顺序缓冲区
void SemanticSegmentation::ordered_output_push(ImageDataPtr image) {
  std::unique_lock<std::mutex> lock(order_mutex_);
  
  // 将图像添加到顺序缓冲区
  ordered_buffer_[image->frame_idx] = image;
  
  // 通知顺序输出线程
  order_cv_.notify_one();
}

// 顺序输出线程函数
void SemanticSegmentation::ordered_output_thread_func() {
  std::cout << "🔄 语义分割顺序输出线程启动" << std::endl;
  
  while (order_thread_running_.load() || !ordered_buffer_.empty()) {
    std::unique_lock<std::mutex> lock(order_mutex_);
    
    // 等待有数据可处理
    order_cv_.wait(lock, [this]() {
      return !ordered_buffer_.empty() || !order_thread_running_.load();
    });
    
    // 按顺序输出连续的帧
    while (!ordered_buffer_.empty()) {
      auto it = ordered_buffer_.find(next_expected_frame_.load());
      if (it != ordered_buffer_.end()) {
        // 找到了下一个期望的帧，输出它
        ImageDataPtr image = it->second;
        int64_t frame_idx = image->frame_idx;
        ordered_buffer_.erase(it);
        lock.unlock();
        
        // 推送到实际的输出队列
        output_queue_.push(image);
        
        // 更新输出监控记录
        // {
        //   std::lock_guard<std::mutex> monitor_lock(output_monitor_mutex_);
        //   recent_output_frames_.push_back(frame_idx);
        //   if (recent_output_frames_.size() > OUTPUT_WINDOW_SIZE) {
        //     recent_output_frames_.pop_front();
        //   }
          
        //   // 打印最近10个输出帧序号用于人工核验
        //   std::cout << "📤 [语义分割输出] 当前帧: " << frame_idx << ", 最近10个输出帧序号: [";
        //   for (size_t i = 0; i < recent_output_frames_.size(); ++i) {
        //     std::cout << recent_output_frames_[i];
        //     if (i < recent_output_frames_.size() - 1) {
        //       std::cout << ", ";
        //     }
        //   }
        //   std::cout << "]";
          
        //   // 检查是否有乱序
        //   bool is_ordered = true;
        //   if (recent_output_frames_.size() > 1) {
        //     for (size_t i = 1; i < recent_output_frames_.size(); ++i) {
        //       if (recent_output_frames_[i] <= recent_output_frames_[i-1]) {
        //         is_ordered = false;
        //         break;
        //       }
        //     }
        //   }
          
        //   if (!is_ordered) {
        //     std::cout << " ⚠️ 检测到输出乱序！";
        //   } else {
        //     std::cout << " ✅ 输出有序";
        //   }
        //   std::cout << std::endl;
        // }
        
        // 更新下一个期望的帧序号
        next_expected_frame_.fetch_add(1);
        
        // 重新加锁继续处理
        lock.lock();
      } else {
        // 下一个期望的帧还没到，等待
        break;
      }
    }
  }
  
  std::cout << "🔄 语义分割顺序输出线程结束" << std::endl;
}

void SemanticSegmentation::set_seg_show_interval(int interval) {
  if (interval > 0) {
    seg_show_interval_ = interval;
    std::cout << "🎯 分割结果保存间隔已设置为: " << interval << " 帧" << std::endl;
  }
}

void SemanticSegmentation::process_image(ImageDataPtr image, int thread_id) {
  if (!image || image->imageMat.empty()) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }

  // 检查线程ID是否有效
  if (thread_id < 0 || thread_id >= road_seg_instances_.size()) {
    std::cerr << "❌ 无效的线程ID: " << thread_id << std::endl;
    return;
  }

  // 使用该线程专属的模型实例
  auto& seg_instance = road_seg_instances_[thread_id];
  if (!seg_instance) {
    std::cerr << "❌ 线程 " << thread_id << " 的模型实例无效" << std::endl;
    return;
  }

  // 执行单个分割
  std::vector<SegmentationResult> seg_results;
  std::vector<cv::Mat> inputs;
  inputs.push_back(image->segInResizeMat);
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (!seg_instance->Predict(inputs, seg_results)) {
    std::cerr << "❌ 语义分割执行失败，帧序号: " << image->frame_idx 
              << "，线程: " << thread_id << std::endl;
    // 设置失败状态但仍标记完成，避免死锁
    image->segmentation_completed = true;
    return;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  // 更新性能统计
  total_processed_images_.fetch_add(1);
  total_processing_time_ms_.fetch_add(duration.count());

  // 处理分割结果
  if (!seg_results.empty() && !seg_results[0].label_map.empty()) {
    image->label_map = std::move(seg_results[0].label_map);
    image->mask_height = 1024; // 固定值，避免重复访问
    image->mask_width = 1024;  // 固定值，避免重复访问
    
    // 保存分割结果（如果启用）
    if (enable_seg_show_ && (image->frame_idx % seg_show_interval_ == 0)) {
      // 在这里可以添加保存逻辑
      std::cout << "💾 保存分割结果，帧序号: " << image->frame_idx << std::endl;
    }
    
    std::cout << "✅ [线程 " << thread_id << "] 语义分割完成，帧序号: " 
              << image->frame_idx << "，耗时: " << duration.count() << "ms" << std::endl;
  } else {
    // 即使语义分割失败也要设置基本信息，避免后续模块死等
    std::cerr << "⚠️ 语义分割结果为空，帧序号: " << image->frame_idx 
              << "，线程: " << thread_id << std::endl;
    image->mask_height = 1024; 
    image->mask_width = 1024;
  }

  // 标记语义分割完成
  image->segmentation_completed = true;
}

// 批量处理方法
void SemanticSegmentation::process_images_batch(std::vector<ImageDataPtr>& images, int thread_id) {
  if (images.empty()) {
    return;
  }
  
  // 检查线程ID是否有效
  if (thread_id < 0 || thread_id >= road_seg_instances_.size()) {
    std::cerr << "❌ 批量处理：无效的线程ID: " << thread_id << std::endl;
    return;
  }

  // 使用该线程专属的模型实例
  auto& seg_instance = road_seg_instances_[thread_id];
  if (!seg_instance) {
    std::cerr << "❌ 批量处理：线程 " << thread_id << " 的模型实例无效" << std::endl;
    return;
  }
  
  // 准备批量输入数据
  std::vector<cv::Mat> image_mats;
  image_mats.reserve(images.size());
  
  auto preprocess_start = std::chrono::high_resolution_clock::now();
  
  // 批量预处理：为每个图像调用 on_processing_start
  for (auto& image : images) {
    on_processing_start(image, thread_id);
    image_mats.push_back(image->segInResizeMat);
  }
  
  auto preprocess_end = std::chrono::high_resolution_clock::now();
  auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
  
  // 批量语义分割处理
  std::vector<SegmentationResult> seg_results;
  auto seg_start = std::chrono::high_resolution_clock::now();
  
  if (!seg_instance->Predict(image_mats, seg_results)) {
    std::cerr << "❌ 批量语义分割执行失败，线程: " << thread_id << std::endl;
    // 即使推理失败，也要标记所有图像的分割已完成，避免死锁
    for (auto& image : images) {
      image->mask_height = 1024;
      image->mask_width = 1024;
      image->segmentation_completed = true;
    }
    return;
  }
  
  auto seg_end = std::chrono::high_resolution_clock::now();
  auto seg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seg_end - seg_start);
  
  // 更新性能统计
  total_processed_images_.fetch_add(images.size());
  total_processing_time_ms_.fetch_add((preprocess_duration + seg_duration).count());
  
  std::cout << "⚡ [线程 " << thread_id << "] 批量语义分割性能统计 - 预处理: " 
            << preprocess_duration.count() << "ms, 推理: " << seg_duration.count() 
            << "ms, 总计: " << (preprocess_duration + seg_duration).count() 
            << "ms, 处理 " << images.size() << " 张图像" << std::endl;
  
  // 处理批量结果
  if (seg_results.size() != images.size()) {
    std::cerr << "❌ 批量分割结果数量不匹配，期望: " << images.size() 
              << "，实际: " << seg_results.size() << "，线程: " << thread_id << std::endl;
    // 标记所有图像完成避免死锁
    for (auto& image : images) {
      image->mask_height = 1024;
      image->mask_width = 1024;
      image->segmentation_completed = true;
    }
    return;
  }
  
  // 批量后处理：快速结果分配
  for (size_t i = 0; i < images.size(); ++i) {
    if (!seg_results[i].label_map.empty()) {
      images[i]->label_map = std::move(seg_results[i].label_map);
      images[i]->mask_height = 1024; // 固定值，避免重复访问
      images[i]->mask_width = 1024;  // 固定值，避免重复访问
      
      // 保存分割结果（如果启用）
      if (enable_seg_show_ && (images[i]->frame_idx % seg_show_interval_ == 0)) {
        std::cout << "💾 保存分割结果，帧序号: " << images[i]->frame_idx << std::endl;
      }
    } else {
      // 即使语义分割失败也要设置基本信息，避免后续模块死等
      std::cerr << "⚠️ 语义分割结果为空，帧序号: " << images[i]->frame_idx 
                << "，线程: " << thread_id << std::endl;
      images[i]->mask_height = 1024; 
      images[i]->mask_width = 1024;
    }
    
    // 无论成功还是失败，都标记语义分割已完成，避免死锁
    images[i]->segmentation_completed = true;
    
    // 调用后处理
    on_processing_complete(images[i], thread_id);
  }
}

void SemanticSegmentation::on_processing_start(ImageDataPtr image, int thread_id) {
  // 使用CUDA进行图像resize处理（带缓存优化）
  if (cuda_available_) {
    try {
      std::lock_guard<std::mutex> lock(gpu_mutex_); // 保护GPU操作
      
      // 检查是否需要调整缓存大小
      if (gpu_src_cache_.rows < image->imageMat.rows || 
          gpu_src_cache_.cols < image->imageMat.cols) {
        gpu_src_cache_.create(image->imageMat.rows, image->imageMat.cols, CV_8UC3);
      }
      
      // 上传到GPU (只上传实际需要的区域)
      cv::cuda::GpuMat gpu_src_roi = gpu_src_cache_(cv::Rect(0, 0, image->imageMat.cols, image->imageMat.rows));
      gpu_src_roi.upload(image->imageMat);
      
      // 在GPU上进行resize操作
      cv::cuda::resize(gpu_src_roi, gpu_dst_cache_, cv::Size(1024, 1024));
      
      // 下载回CPU
      gpu_dst_cache_.download(image->segInResizeMat);
      
    } catch (const cv::Exception& e) {
      // 如果CUDA操作失败，标记CUDA不可用并回退到CPU实现
      std::cerr << "⚠️ CUDA resize失败，禁用CUDA并回退到CPU: " << e.what() << std::endl;
      cuda_available_ = false;
      cv::resize(image->imageMat, image->segInResizeMat, cv::Size(1024, 1024));
    }
  } else {
    // 使用CPU实现
    cv::resize(image->imageMat, image->segInResizeMat, cv::Size(1024, 1024));
  }
}

void SemanticSegmentation::on_processing_complete(ImageDataPtr image, int thread_id) {
  // 可以在这里添加语义分割特有的后处理逻辑
  // 例如：结果验证、统计信息更新等
}

void SemanticSegmentation::change_params(const PipelineConfig &config) {
  if (config.enable_seg_show) {
    enable_seg_show_ = config.enable_seg_show;
    seg_show_image_path_ = config.seg_show_image_path;
  }
}

void SemanticSegmentation::segmentation_worker() {
  // 这个方法保留为空，因为我们现在使用基类的工作线程机制
}
