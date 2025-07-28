#include "object_detection.h"
#include "pipeline_manager.h"
#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <thread>
const int det_batch_size = 8;
ObjectDetection::ObjectDetection(int num_threads, const PipelineConfig* config)
    : ImageProcessor(0, "目标检测"), stop_worker_(false), config_(config) { // 设置基类线程数为0

  // 初始化处理队列
  detection_queue_ =
      std::make_unique<ThreadSafeQueue<ImageDataPtr>>(100); // 设置队列容量为100

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
    algor_config.max_batch_size = det_batch_size;
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
  // std::cout << "🔍 目标检测模块初始化完成（正常模式）" << std::endl;

  // 启动工作线程
  worker_thread_ = std::thread(&ObjectDetection::detection_worker, this);

  // std::cout << "🔍 目标检测模块启动完成" << std::endl;

}

void ObjectDetection::add_image(ImageDataPtr image) {
  if (!image) {
    std::cerr << "Error: Invalid image data in add_image" << std::endl;
    return;
  }
  // 去除检测接收图像打印
  detection_queue_->push(image);
  // 直接添加到检测队列，不使用基类的input_queue_
}

void ObjectDetection::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }
  // std::cout << "📥 目标检测接收图像: 帧 " << image->frame_idx << " (线程 " << thread_id << ")" << std::endl;
  detection_queue_->push(image);
  // 注意：不在这里设置promise，而是在detection_worker中设置
}

void ObjectDetection::on_processing_start(ImageDataPtr image, int thread_id) {
  // std::cout << "🎯 目标检测准备开始 (线程 " << thread_id << ")" << std::endl;
}

void ObjectDetection::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  // std::cout << "🎯 目标检测处理完成 (线程 " << thread_id << ")" << std::endl;
}

void ObjectDetection::perform_object_detection(ImageDataPtr image,
                                               int thread_id) {
  // 保留单图接口，实际不再直接调用
}

void ObjectDetection::detection_worker() {
  while (!stop_worker_) {
    // 收集一批图像进行批处理，保持接收顺序
    std::vector<ImageDataPtr> batch_images;
    batch_images.reserve(det_batch_size); // 内存优化：预分配批次大小
    
    // 阻塞等待第一个图像
    ImageDataPtr first_img;
    try {
      detection_queue_->wait_and_pop(first_img);
    } catch (...) {
      // 队列可能被销毁，退出循环
      break;
    }
    
    if (!first_img) {
      continue;
    }
    
    // 检查是否需要停止
    if (stop_worker_) {
      // 设置promise避免阻塞
      try {
        if (first_img->detection_promise && 
            first_img->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
          first_img->detection_promise->set_value();
        }
      } catch (const std::future_error&) {
        // Promise已经被设置，忽略
      }
      break;
    }
    
    // std::cout << "🔄 目标检测开始处理批次，首帧: " << first_img->frame_idx << std::endl;
    batch_images.push_back(first_img);
    
    // 尝试收集更多图像组成批次，但不阻塞等待，保持顺序
    while (batch_images.size() < det_batch_size && !stop_worker_) {
      ImageDataPtr img;
      if (detection_queue_->try_pop(img)) {
        if (img) {
          batch_images.push_back(img);
        }
      } else {
        break; // 队列中没有更多图像，立即处理当前批次
      }
    }

    // 打印批次中的帧序号顺序（简化输出）
    // 仅在调试时显示批次信息
    // std::cout << "📊 检测批次: " << batch_images.size() << " 帧" << std::endl;

    // 等待所有图像的 Mask 后处理完成（仅在语义分割和mask后处理都启用时）
    for (auto& img : batch_images) {
      try {
        std::cout << "步骤2: 等待批量处理完成并获取结果, 帧: " << img->frame_idx << std::endl;
        
        // 检查当前配置状态
        bool seg_enabled = config_ && config_->enable_segmentation;
        bool mask_enabled = config_ && config_->enable_mask_postprocess;
        std::cout << "配置状态 - 语义分割: " << (seg_enabled ? "启用" : "禁用") 
                  << ", Mask后处理: " << (mask_enabled ? "启用" : "禁用") 
                  << ", 帧: " << img->frame_idx << std::endl;
        
        // 只有在语义分割和mask后处理都启用时才等待mask后处理完成
        if (seg_enabled && mask_enabled) {
          std::cout << "等待mask后处理完成，帧: " << img->frame_idx << std::endl;
          // 使用超时等待避免死锁
          auto status = img->mask_postprocess_future.wait_for(std::chrono::seconds(5));
          if (status == std::future_status::ready) {
            img->mask_postprocess_future.get(); // 获取结果
            std::cout << "mask后处理已完成，帧: " << img->frame_idx << std::endl;
          } else if (status == std::future_status::timeout) {
            std::cerr << "❌ Mask后处理超时（5秒），跳过，帧: " << img->frame_idx << std::endl;
            continue; // 跳过这个图像
          } else {
            std::cerr << "❌ Mask后处理状态异常，跳过，帧: " << img->frame_idx << std::endl;
            continue; // 跳过这个图像
          }
        } else {
          // 如果语义分割或mask后处理被禁用，确保promise已被设置
          auto status = img->mask_postprocess_future.wait_for(std::chrono::milliseconds(1));
          if (status != std::future_status::ready) {
            std::cout << "🔧 手动设置mask后处理promise（模块已禁用），帧: " << img->frame_idx << std::endl;
            if (img->mask_postprocess_promise) {
              img->mask_postprocess_promise->set_value();
            }
          } else {
            std::cout << "mask后处理promise已设置（模块已禁用），帧: " << img->frame_idx << std::endl;
          }
        }
        
        std::cout << "步骤2: 完成, 帧: " << img->frame_idx << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "❌ Mask后处理失败，帧 " << img->frame_idx << ": " << e.what() << std::endl;
        // 如果 Mask 后处理失败，跳过这个图像的目标检测
        continue;
      }
    }

    // 处理批次（无论大小）
    try {
      // 内存优化：预分配批处理缓冲区
      std::vector<cv::Mat> mats;
      mats.reserve(batch_images.size()); // 预分配避免重复扩容
      
      for (auto &img : batch_images) {
        // 使用ROI引用，避免数据拷贝（这已经是最优的）
        cv::Mat cropped_image = (img->imageMat)(img->roi);
        mats.push_back(cropped_image);
      }
      
      // 车辆检测
      std::vector<detect_result_group_t*> car_outs;
      car_outs.reserve(batch_images.size());
      for (size_t i = 0; i < batch_images.size(); ++i) {
        car_outs.push_back(new detect_result_group_t());
      }
      
      car_detect_instance_->forward(mats, car_outs.data());
      
      // 行人检测（如果启用）
      std::vector<detect_result_group_t*> person_outs;
      if (personal_detect_instance_) {
        person_outs.reserve(batch_images.size());
        for (size_t i = 0; i < batch_images.size(); ++i) {
          person_outs.push_back(new detect_result_group_t());
        }
        personal_detect_instance_->forward(mats, person_outs.data());
      }
      
      // 处理每个图像的检测结果
      for (size_t idx = 0; idx < batch_images.size(); ++idx) {
        auto &image = batch_images[idx];
        std::cout << "🔄 处理检测结果，帧: " << image->frame_idx << std::endl;
        int total_detections = 0;
        
        // 处理车辆检测结果 (class_id保持原值，通常是0)
        if (car_outs[idx]->count > 0) {
          for (int i = 0; i < car_outs[idx]->count; ++i) {
            detect_result_t &result = car_outs[idx]->results[i];
            image->detection_results.push_back({
                result.box.left+image->roi.x, result.box.top+image->roi.y, 
                result.box.right+image->roi.x, result.box.bottom+image->roi.y,
                result.prop, result.cls_id, result.track_id});
          }
          total_detections += car_outs[idx]->count;
        }
        
        // 处理行人检测结果 (class_id设置为1)
        if (personal_detect_instance_ && person_outs[idx]->count > 0) {
          for (int i = 0; i < person_outs[idx]->count; ++i) {
            detect_result_t &result = person_outs[idx]->results[i];
            image->detection_results.push_back({
                result.box.left+image->roi.x, result.box.top+image->roi.y, 
                result.box.right+image->roi.x, result.box.bottom+image->roi.y,
                result.prop, 1, result.track_id}); // 行人检测类别ID设置为1
          }
          total_detections += person_outs[idx]->count;
        }
        
        // if (total_detections > 0) {
        //   // 去除检测完成输出
        //   // std::cout << "✅ 目标检测完成 (帧 " << image->frame_idx << ")，检测到 " << total_detections << " 个目标" << std::endl;
        // } else {
        //   // 去除未检测到目标的输出
        //   // std::cout << "⚠️ 目标检测完成 (帧 " << image->frame_idx << ")，但未检测到目标" << std::endl;
        // }
        
        // 设置promise完成 - 先检查是否已经设置
        try {
          if (image->detection_promise && 
              image->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            std::cout << "✅ 设置检测promise完成，帧: " << image->frame_idx << std::endl;
            image->detection_promise->set_value();
          }
        } catch (const std::future_error& e) {
          std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
        }
      }
      
      // 内存优化：释放车辆检测结果
      for (auto* result : car_outs) {
        delete result;
      }
      
      // 内存优化：释放行人检测结果
      if (personal_detect_instance_) {
        for (auto* result : person_outs) {
          delete result;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "目标检测处理失败: " << e.what() << std::endl;
      // 设置异常状态
      for (auto &img : batch_images) {
        if (img) {
          try {
            if (img->detection_promise && 
                img->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
              img->detection_promise->set_exception(std::current_exception());
            }
          } catch (const std::future_error& e) {
            // std::cout << "⚠️ Promise异常已被设置，帧 " << img->frame_idx << ": " << e.what() << std::endl;
          }
        }
      }
    }
  }
}

ObjectDetection::~ObjectDetection() {
  // std::cout << "🔄 正在停止目标检测模块..." << std::endl;
  stop_worker_ = true;
  
  // 清空队列中剩余的图像，避免阻塞
  ImageDataPtr remaining_img;
  while (detection_queue_->try_pop(remaining_img)) {
    if (remaining_img) {
      try {
        if (remaining_img->detection_promise && 
            remaining_img->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
          remaining_img->detection_promise->set_value();
        }
      } catch (const std::future_error&) {
        // Promise已经被设置，忽略
      }
    }
  }
  
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  
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
