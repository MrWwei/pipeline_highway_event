#pragma once

#include "image_data.h"
#include "mask_postprocess.h"
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
  std::unique_ptr<MaskPostProcess> mask_postprocess_;
  std::unique_ptr<ObjectDetection> object_det_;

  std::atomic<bool> running_;
  // 为每个阶段创建独立的协调线程
  std::thread seg_to_mask_thread_;     // 语义分割->Mask后处理
  std::thread mask_to_detect_thread_;  // Mask后处理->目标检测
  std::thread detect_to_final_thread_; // 目标检测->最终结果

  ThreadSafeQueue<ImageDataPtr> final_results_;
  std::map<uint64_t, ImageDataPtr> pending_results_; // 用于暂存未按序的结果
  std::mutex pending_results_mutex_;
  uint64_t next_frame_idx_; // 下一个应该输出的帧序号

private:
  // 各阶段的处理函数
  void seg_to_mask_thread_func(); // 处理语义分割到Mask后处理的数据流转
  void mask_to_detect_thread_func(); // 处理Mask后处理到目标检测的数据流转
  void detect_to_final_thread_func(); // 处理目标检测到最终结果的数据流转

public:
  // 构造函数，可指定各个模块的线程数量
  PipelineManager(int semantic_threads = 2, int mask_postprocess_threads = 1,
                  int detection_threads = 2);
  ~PipelineManager();

  // 启动流水线
  void start();

  // 停止流水线
  void stop();

  // 添加图像到流水线
  void add_image(const ImageDataPtr &img_data);

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
