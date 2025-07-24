#include "object_detection.h"
#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <thread>
const int det_batch_size = 8;
ObjectDetection::ObjectDetection(int num_threads)
    : ImageProcessor(0, "目标检测"), stop_worker_(false) { // 设置基类线程数为0

  // 初始化处理队列
  detection_queue_ =
      std::make_unique<ThreadSafeQueue<ImageDataPtr>>(100); // 设置队列容量为100

  AlgorConfig config;
  config.algorName_ = "object_detect";
  config.model_path = "car_detect.onnx";
  config.img_size = 640;
  config.conf_thresh = 0.25f;
  config.iou_thresh = 0.2f;
  config.max_batch_size = det_batch_size;
  config.min_opt = 1;
  config.mid_opt = 16;
  config.max_opt = 32;
  config.is_ultralytics = 1;
  config.gpu_id = 0;

  // 初始化检测器
  car_detect_instance_ = xtkj::createDetect();
  car_detect_instance_->init(config);
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

    // 等待所有图像的 Mask 后处理完成
    for (auto& img : batch_images) {
      try {
        // 去除等待打印信息
        img->mask_postprocess_future.get(); // 阻塞等待
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
        cv::Mat cropped_image = (*img->imageMat)(img->roi);
        mats.push_back(cropped_image);
      }
      
      // 内存优化：预分配检测结果数组
      std::vector<detect_result_group_t*> outs;
      outs.reserve(batch_images.size());
      for (size_t i = 0; i < batch_images.size(); ++i) {
        outs.push_back(new detect_result_group_t());
      }
      
      car_detect_instance_->forward(mats, outs.data());
      
      // 处理每个图像的检测结果
      for (size_t idx = 0; idx < batch_images.size(); ++idx) {
        auto &image = batch_images[idx];
        if (outs[idx]->count > 0) {
          for (int i = 0; i < outs[idx]->count; ++i) {
            detect_result_t &result = outs[idx]->results[i];
            image->detection_results.push_back({
                result.box.left+image->roi.x, result.box.top+image->roi.y, result.box.right+image->roi.x, result.box.bottom+image->roi.y,
                result.prop, result.cls_id, result.track_id});
          }
          // 去除检测完成输出
          // std::cout << "✅ 目标检测完成 (帧 " << image->frame_idx << ")，检测到 " << outs[idx]->count << " 个目标" << std::endl;
        } else {
          // 去除未检测到目标的输出
          // std::cout << "⚠️ 目标检测完成 (帧 " << image->frame_idx << ")，但未检测到目标" << std::endl;
        }
        
        // 设置promise完成 - 先检查是否已经设置
        try {
          if (image->detection_promise && 
              image->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            image->detection_promise->set_value();
          }
        } catch (const std::future_error& e) {
          // std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
        }
      }
      
      // 内存优化：使用vector自动管理内存
      for (auto* result : outs) {
        delete result; // 释放每个结果组
      }
      // vector会自动释放
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
  
  // std::cout << "✅ 目标检测模块已停止" << std::endl;
}
