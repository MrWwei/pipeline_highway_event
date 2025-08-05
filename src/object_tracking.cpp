#include "object_tracking.h"
#include <chrono>
#include <future>
#include <iostream>
#include <algorithm>
#include <deque>
#include "image_data.h"

ObjectTracking::ObjectTracking(int num_threads)
    : ImageProcessor(num_threads, "目标跟踪"), stop_worker_(false){
  
  // 调试模式：跳过跟踪器初始化
  car_track_instance_ = xtkj::createTracker(30, 30, 0.5, 0.6, 0.8);
  vehicle_parking_instance_ = createVehicleParkingDetect();
  
  // 初始化帧序号监控窗口
  recent_input_frames_.clear();
  std::cout << "✅ 目标跟踪模块初始化完成，已启用帧序号监控（窗口大小: " << WINDOW_SIZE << "）" << std::endl;
}

ObjectTracking::~ObjectTracking() {
  stop_worker_ = true;
  if (worker_thread_.joinable()) {
    // 使用 future 来实现超时等待
    auto future = std::async(std::launch::async, [this]() {
      if (worker_thread_.joinable()) {
        worker_thread_.join();
      }
    });
    
    if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
      std::cout << "⚠️ 目标跟踪工作线程超时，强制分离" << std::endl;
      worker_thread_.detach();
    }
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
  
  // 记录当前帧序号并维护最近10个帧序号的窗口
  // recent_input_frames_.push_back(image->frame_idx);
  // if (recent_input_frames_.size() > WINDOW_SIZE) {
  //   recent_input_frames_.pop_front();
  // }
  
  // // 打印最近10个帧序号用于人工校验
  // std::cout << "📋 [目标跟踪] 当前帧: " << image->frame_idx << ", 最近10个帧序号: [";
  // for (size_t i = 0; i < recent_input_frames_.size(); ++i) {
  //   std::cout << recent_input_frames_[i];
  //   if (i < recent_input_frames_.size() - 1) {
  //     std::cout << ", ";
  //   }
  // }
  // std::cout << "]" << std::endl;
  
  // // 检查是否有乱序
  // bool is_ordered = true;
  // if (recent_input_frames_.size() > 1) {
  //   for (size_t i = 1; i < recent_input_frames_.size(); ++i) {
  //     if (recent_input_frames_[i] <= recent_input_frames_[i-1]) {
  //       is_ordered = false;
  //       break;
  //     }
  //   }
  // }
  
  // if (!is_ordered) {
  //   std::cout << "⚠️ [目标跟踪] 检测到帧序号乱序！" << std::endl;
  // }
  
  perform_tracking(image);
}

void ObjectTracking::on_processing_start(ImageDataPtr image, int thread_id) {
  // 跟踪特有的预处理
}

void ObjectTracking::on_processing_complete(ImageDataPtr image, int thread_id) {
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
  std::vector<TrackBox> track_boxes;
  for (int i = 0; i < out->count; ++i) {
    detect_result_t &result = out->results[i];
    // 这里的box是resize后的坐标，需要转换回原图像坐标系
    TrackBox box = TrackBox(result.track_id, 
                                cv::Rect((result.box.left + image->roi.x) * image->parkingResizeMat.cols / image->width, 
                                (result.box.top + image->roi.y) * image->parkingResizeMat.rows / image->height,
                                (result.box.right - result.box.left) * image->parkingResizeMat.cols / image->width,
                                (result.box.bottom - result.box.top) * image->parkingResizeMat.rows / image->height),
                                result.cls_id, 
                                result.prop, 
                                false, 0.0);

    // track_boxes.push_back(box);
    ImageData::BoundingBox track_box;
    track_box.track_id = box.track_id;
    track_box.left = box.box.x;
    track_box.top = box.box.y;
    track_box.right = box.box.x + box.box.width;
    track_box.bottom = box.box.y + box.box.height;
    track_box.confidence = box.confidence;
    track_box.class_id = box.cls_id;
    track_box.is_still = box.is_still;
    image->track_results.push_back(track_box);
  }
  // auto start_time = std::chrono::high_resolution_clock::now();
  // vehicle_parking_instance_->detect(image->parkingResizeMat, track_boxes);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "🚗 车辆违停检测耗时: " << duration.count() << " ms" << std::endl;
  // for(const auto &track_box : track_boxes) {
  //   ImageData::BoundingBox box;
  //   box.track_id = track_box.track_id;
  //   box.left = track_box.box.x * image->width / image->parkingResizeMat.cols;
  //   box.top = track_box.box.y * image->height / image->parkingResizeMat.rows;
  //   box.right = (track_box.box.x + track_box.box.width) * image->width / image->parkingResizeMat.cols;
  //   box.bottom = (track_box.box.y + track_box.box.height) * image->height / image->parkingResizeMat.rows;
  //   box.confidence = track_box.confidence;
  //   box.class_id = track_box.cls_id;
  //   box.is_still = track_box.is_still;
  //   image->track_results.push_back(box);
  // }
  
  
  // 释放分配的内存，防止内存泄漏
  delete out;
  image->track_completed = true;
  out = nullptr;
}
