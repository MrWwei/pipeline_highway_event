#include "object_tracking.h"
#include <chrono>
#include <future>
#include <iostream>
#include <algorithm>
#include "image_data.h"

ObjectTracking::ObjectTracking(int num_threads)
    : ImageProcessor(num_threads, "目标跟踪"), stop_worker_(false), next_expected_frame_(0) {
  
  // 调试模式：跳过跟踪器初始化
  car_track_instance_ = xtkj::createTracker(30, 30, 0.5, 0.6, 0.8);
  std::cout << "🚫 目标跟踪模块已禁用（调试模式），线程数: " << num_threads << std::endl;
  
  // 启动顺序处理工作线程
  worker_thread_ = std::thread(&ObjectTracking::sequential_tracking_worker, this);
}

ObjectTracking::~ObjectTracking() {
  stop_worker_ = true;
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  
  if (car_track_instance_) {
    delete car_track_instance_;
    car_track_instance_ = nullptr;
  }
}

void ObjectTracking::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "⚠️ [目标跟踪] 收到空图像指针" << std::endl;
    return;
  }

  // 将图像添加到待处理队列，等待顺序处理
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    
    // 更新最近输入帧序号的滑动窗口
    recent_input_frames_.push_back(image->frame_idx);
    if (recent_input_frames_.size() > WINDOW_SIZE) {
      recent_input_frames_.pop_front();
    }
    
    // 打印最近输入的帧序号窗口
    std::cout << "🎯 跟踪输入帧序号 [" << image->frame_idx << "] 最近窗口: [";
    for (size_t i = 0; i < recent_input_frames_.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << recent_input_frames_[i];
    }
    std::cout << "] 期望帧: " << next_expected_frame_ << std::endl;
    
    pending_images_.push_back(image);
  }
  
  // 等待检测promise完成
  try {
    image->detection_future.get();
    // 去除检测完成输出
    // std::cout << "✅ 检测已完成，准备跟踪，帧 " << image->frame_idx << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "❌ 检测阶段失败，跳过跟踪，帧 " << image->frame_idx << ": " << e.what() << std::endl;
    try {
      if (image->tracking_promise && 
          image->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        image->tracking_promise->set_exception(std::current_exception());
      }
    } catch (const std::future_error& e) {
      std::cout << "⚠️ Promise异常已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
    }
    return;
  }

  // 将图像添加到待处理队列（重新加锁进行排序操作）
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    
    // 按帧序号排序
    std::sort(pending_images_.begin(), pending_images_.end(), 
              [](const ImageDataPtr& a, const ImageDataPtr& b) {
                return a->frame_idx < b->frame_idx;
              });
    
    // 打印当前等待队列状态（简化输出）
    // std::cout << "📋 跟踪等待队列 [大小: " << pending_images_.size() << "] 帧序号: ";
    // for (const auto& img : pending_images_) {
    //   std::cout << img->frame_idx << " ";
    // }
    std::cout << std::endl;
  }
}

void ObjectTracking::on_processing_start(ImageDataPtr image, int thread_id) {
  // 跟踪特有的预处理
}

void ObjectTracking::on_processing_complete(ImageDataPtr image, int thread_id) {
  // 跟踪特有的后处理
  std::cout << "✅ 目标跟踪完成，帧 " << image->frame_idx << std::endl;
}

void ObjectTracking::sequential_tracking_worker() {
  std::cout << "🔄 目标跟踪顺序处理线程启动" << std::endl;
  
  while (!stop_worker_.load()) {
    ImageDataPtr next_image = nullptr;
    
    // 检查是否有下一个期望的帧
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      
      // 查找下一个期望的帧
      auto it = std::find_if(pending_images_.begin(), pending_images_.end(),
                            [this](const ImageDataPtr& img) {
                              return img->frame_idx == next_expected_frame_;
                            });
      
      if (it != pending_images_.end()) {
        next_image = *it;
        pending_images_.erase(it);
        std::cout << "✅ 找到期望帧 " << next_expected_frame_ 
                  << "，剩余等待帧数: " << pending_images_.size() << std::endl;
        next_expected_frame_++;
      } else if (!pending_images_.empty()) {
        // 如果没有找到期望的帧，但有其他帧在等待，显示等待状态
        auto min_frame = std::min_element(pending_images_.begin(), pending_images_.end(),
                                         [](const ImageDataPtr& a, const ImageDataPtr& b) {
                                           return a->frame_idx < b->frame_idx;
                                         });
        if (min_frame != pending_images_.end()) {
          std::cout << "⏳ 等待帧 " << next_expected_frame_ 
                    << "，当前最小帧: " << (*min_frame)->frame_idx 
                    << "，等待队列: " << pending_images_.size() << " 帧" << std::endl;
        }
      }
    }
    
    if (next_image) {
      std::cout << "🎯 按序处理跟踪，帧 " << next_image->frame_idx 
                << " (期望序列正确)" << std::endl;
      perform_tracking(next_image);
      
      // 将处理完成的图像添加到输出队列
      output_queue_.push(next_image);
    } else {
      // 没有可处理的帧，短暂休眠
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  
  std::cout << "⏹️ 目标跟踪顺序处理线程结束" << std::endl;
}

void ObjectTracking::perform_tracking(ImageDataPtr image) {
  if (!image) {
    std::cerr << "❌ 跟踪参数无效" << std::endl;
    return;
  }
  
  detect_result_group_t *out = new detect_result_group_t();
  for(auto detect_box:image->detection_results) {
    detect_result_t result;
    result.cls_id = detect_box.class_id;
    result.box.left = detect_box.left - image->roi.x;
    result.box.top = detect_box.top - image->roi.y;
    result.box.right = detect_box.right - image->roi.x;
    result.box.bottom = detect_box.bottom - image->roi.y;
    result.prop = detect_box.confidence;
    result.track_id = detect_box.track_id; // 保留跟踪ID
    out->results[out->count++] = result;
  }
  car_track_instance_->track(out, image->roi.width,
                                       image->roi.height);
  image->track_results.clear();
  for (int i = 0; i < out->count; ++i) {
    detect_result_t &result = out->results[i];
    ImageData::BoundingBox box;
    box.left = result.box.left + image->roi.x;
    box.top = result.box.top + image->roi.y;
    box.right = result.box.right + image->roi.x;
    box.bottom = result.box.bottom + image->roi.y;
    box.confidence = result.prop;
    box.class_id = result.cls_id;
    box.track_id = result.track_id;
    image->track_results.push_back(box);
  }
  
  try {
    // 直接设置跟踪完成，不执行实际跟踪 - 先检查是否已经设置
    if (image->tracking_promise && 
        image->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      image->tracking_promise->set_value();
    }
  } catch (const std::future_error& e) {
    std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
  }
}
