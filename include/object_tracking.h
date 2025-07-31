#pragma once

#include "image_processor.h"
#include "byte_track.h"
#include <deque>
#include "vehicle_parking_detect.h"

/**
 * 目标跟踪模块
 * 对检测结果进行跟踪处理，确保按帧序号顺序处理
 */
class ObjectTracking : public ImageProcessor {
private:
  xtkj::ITracker *car_track_instance_;
  VehicleParkingDetect *vehicle_parking_instance_; // 停车检测实例
  
  // 跟踪工作线程
  std::thread worker_thread_;
  std::atomic<bool> stop_worker_;
  
  
  // 用于监控输入顺序的滑动窗口
  std::deque<uint64_t> recent_input_frames_; // 最近接收到的帧序号
  static const size_t WINDOW_SIZE = 10; // 窗口大小

public:
  ObjectTracking(int num_threads = 1);
  ~ObjectTracking();

  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;
  

private:
  // 执行跟踪算法
  void perform_tracking(ImageDataPtr image);
};
