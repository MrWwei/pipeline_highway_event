#include "object_detection.h"
#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
const int det_batch_size = 8; // 目标检测批量大小

ObjectDetection::ObjectDetection(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "目标检测"), config_(*config) { // 使用传入的线程数

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
    algor_config.max_batch_size = det_batch_size; // 使用批量大小
    algor_config.min_opt = 1;
    algor_config.mid_opt = 16;
    algor_config.max_opt = 32;
    algor_config.is_ultralytics = 1;
    algor_config.gpu_id = 0;
  }

  // 初始化检测器
  car_detect_instance_ = xtkj::createDetect();
  car_detect_instance_->init(algor_config);
  
  // 初始化行人检测器（如果启用）
  if (config && config->enable_pedestrian_detect) {
    personal_detect_instance_ = xtkj::createDetect();
    // 为行人检测使用单独的配置，只修改模型路径
    AlgorConfig person_config = algor_config; // 复制车辆检测配置
    person_config.model_path = config->pedestrian_det_model_path;
    personal_detect_instance_->init(person_config);
  } else {
    personal_detect_instance_ = nullptr;
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
  
  std::cout << "🔍 目标检测模块初始化完成（批量处理模式，批量大小: " << BATCH_SIZE << "）" << std::endl;
}

// 重写工作线程函数以支持批量处理
void ObjectDetection::worker_thread_func(int thread_id) {
    std::vector<ImageDataPtr> batch_images;
    batch_images.reserve(BATCH_SIZE);
    
    while (running_) {
        batch_images.clear();
        
        // 收集批量数据
        ImageDataPtr image;
        if (input_queue_.wait_and_pop(image)) {
            batch_images.push_back(image);
            
            // 尝试收集更多图像直到达到批量大小
            while (batch_images.size() < BATCH_SIZE && input_queue_.try_pop(image)) {
                batch_images.push_back(image);
            }
            
            // 处理批量数据
            if (!batch_images.empty()) {
                process_images_batch(batch_images, thread_id);
                
                // 更新统计信息
                total_processed_images_ += batch_images.size();
                total_batch_count_++;
            }
        }
    }
}

// 批量处理图像
void ObjectDetection::process_images_batch(std::vector<ImageDataPtr>& images, int thread_id) {
    if (images.empty()) return;
    
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // 准备批量检测的数据
        std::vector<cv::Mat> mats;
        mats.reserve(images.size());
        
        // 预处理每个图像并准备批量数据
        for (auto& image : images) {
            if (!image) continue;
            
            // 调用预处理
            on_processing_start(image, thread_id);
            
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
        
        if (mats.empty()) return;
        
        // 车辆检测 - 批量处理
        std::vector<detect_result_group_t> car_outs(mats.size());
        std::vector<detect_result_group_t*> car_out_ptrs;
        car_out_ptrs.reserve(mats.size());
        for (auto& out : car_outs) {
            car_out_ptrs.push_back(&out);
        }
        
        car_detect_instance_->forward(mats, car_out_ptrs.data());
        
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
        if (personal_detect_instance_) {
            std::vector<detect_result_group_t> person_outs(mats.size());
            std::vector<detect_result_group_t*> person_out_ptrs;
            person_out_ptrs.reserve(mats.size());
            for (auto& out : person_outs) {
                person_out_ptrs.push_back(&out);
            }
            
            personal_detect_instance_->forward(mats, person_out_ptrs.data());
            
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
        
        // 标记所有图像检测完成并调用后处理
        for (auto& image : images) {
            if (image) {
                image->detection_completed = true;
                on_processing_complete(image, thread_id);
                // 将处理完成的图像添加到输出队列
                output_queue_.push(image);
            }
        }
        
        // 计算并记录处理时间
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        total_processing_time_ms_ += duration.count();
        
        // 定期输出统计信息（每处理100个批次）
        if (total_batch_count_ % 100 == 0) {
            double avg_batch_time = static_cast<double>(total_processing_time_ms_.load()) / total_batch_count_.load();
            double avg_images_per_batch = static_cast<double>(total_processed_images_.load()) / total_batch_count_.load();
            std::cout << "🔍 目标检测统计 - 批次: " << total_batch_count_.load() 
                      << ", 总图像: " << total_processed_images_.load()
                      << ", 平均批次时间: " << avg_batch_time << "ms"
                      << ", 平均批量大小: " << avg_images_per_batch << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 目标检测批量处理失败: " << e.what() << std::endl;
        // 标记所有图像检测完成避免阻塞，并添加到输出队列
        for (auto& image : images) {
            if (image) {
                image->detection_completed = true;
                output_queue_.push(image);
            }
        }
    }
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
  
  if (car_detect_instance_) {
    delete car_detect_instance_;
    car_detect_instance_ = nullptr;
  }
  
  if (personal_detect_instance_) {
    delete personal_detect_instance_;
    personal_detect_instance_ = nullptr;
  }
  
  // std::cout << "✅ 目标检测模块已停止" << std::endl;
}
