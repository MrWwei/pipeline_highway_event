#pragma once

#include "image_data.h"
#include "object_detection.h"
#include "semantic_segmentation.h"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

/**
 * 流水线管理器
 * 管理语义分割和目标检测的流水线处理
 */
class PipelineManager {
private:
  std::unique_ptr<SemanticSegmentation> semantic_seg_;
  std::unique_ptr<ObjectDetection> object_det_;

  std::atomic<bool> running_;
  std::thread coordinator_thread_;

  ThreadSafeQueue<ImageDataPtr> final_results_;

public:
  // 构造函数，可指定语义分割和目标检测的线程数量
  PipelineManager(int semantic_threads = 2, int detection_threads = 2);
  ~PipelineManager();

  // 启动流水线
  void start();

  // 停止流水线
  void stop();

  // 添加图像到流水线
  void add_image(const std::string &image_path);

  // 获取完全处理完成的结果
  bool get_final_result(ImageDataPtr &result);

  // 获取流水线状态信息
  void print_status() const;

  // 获取线程配置信息
  void print_thread_info() const;

private:
  // 协调器线程函数
  void coordinator_thread_func();
};
