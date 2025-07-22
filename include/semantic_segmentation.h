#pragma once

#include "image_processor.h"
#include "road_seg.h"
#include "seg_utils.h"

/**
 * 语义分割处理器
 * 继承自ImageProcessor基类，专门负责对输入图像进行语义分割处理
 * 支持多线程并发处理
 */
class SemanticSegmentation : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  SemanticSegmentation(int num_threads = 1);

  // 析构函数（自定义实现，负责线程清理）
  virtual ~SemanticSegmentation();

protected:
  // 重写基类的纯虚函数：执行语义分割算法（模拟）
  virtual void process_image(ImageDataPtr image, int thread_id) override;

  // 重写处理开始前的准备工作
  virtual void on_processing_start(ImageDataPtr image, int thread_id) override;

  // 重写处理完成后的清理工作
  virtual void on_processing_complete(ImageDataPtr image,
                                      int thread_id) override;

public:
  // 启动处理线程
  void start() override {
    ImageProcessor::start(); // 调用基类的start
    if (!worker_thread_.joinable()) {
      stop_worker_ = false;
      worker_thread_ =
          std::thread(&SemanticSegmentation::segmentation_worker, this);
    }
  }

  // 停止处理线程
  void stop() override {
    stop_worker_ = true;
    if (worker_thread_.joinable()) {
      if (!segmentation_queue_->empty()) {
        std::cout << "Waiting for segmentation queue to empty..." << std::endl;
      }
      worker_thread_.join();
    }
    ImageProcessor::stop(); // 调用基类的stop
  }

private:
  // 具体的语义分割算法实现
  void perform_semantic_segmentation(ImageDataPtr image, int thread_id);
  void segmentation_worker();

  std::unique_ptr<ThreadSafeQueue<ImageDataPtr>> segmentation_queue_;
  std::thread worker_thread_;
  std::atomic<bool> stop_worker_;
  IRoadSeg *road_seg_instance_; // 支持多线程的SDK实例列表
};
