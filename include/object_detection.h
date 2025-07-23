#pragma once

#include "detect.h"
#include "image_processor.h"
#include "thread_safe_queue.h"

/**
 * 目标检测模块
 * 对图像进行目标检测，批量处理以提高效率
 */
class ObjectDetection : public ImageProcessor {
public:
  ObjectDetection(int num_threads = 1);
  virtual ~ObjectDetection();

  // 重写add_image以直接处理而不使用基类工作线程
  void add_image(ImageDataPtr image) override;
  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;

private:
  xtkj::IDetect *car_detect_instance_; // 目标检测实例
  int det_batch_size = 16; // 批处理大小
  std::unique_ptr<ThreadSafeQueue<ImageDataPtr>> detection_queue_; // 目标检测队列
  std::atomic<bool> stop_worker_; // 控制工作线程的停止
  std::thread worker_thread_; // 工作线程
  // 执行目标检测算法
  void perform_object_detection(ImageDataPtr image, int thread_id);
  void detection_worker();
  
};
