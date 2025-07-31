#pragma once

#include "image_data.h"
#include "mask_postprocess.h"
#include "object_detection.h"
#include "object_tracking.h"
#include "event_determine.h"
#include "semantic_segmentation.h"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <string>
#include "pipeline_config.h"
/**
 * 流水线管理器配置参数
 */


/**
 * 流水线管理器
 * 管理语义分割、Mask后处理、目标检测、目标跟踪和目标框筛选的流水线处理
 */
class PipelineManager {
private:
  PipelineConfig config_;  // 保存配置，用于判断模块是否启用
  
  std::unique_ptr<SemanticSegmentation> semantic_seg_;
  std::unique_ptr<MaskPostProcess> mask_postprocess_;
  std::unique_ptr<ObjectDetection> object_det_;
  std::unique_ptr<ObjectTracking> object_track_;
  std::unique_ptr<EventDetermine> event_determine_;

  std::atomic<bool> running_;
  
  // 输入缓冲队列和管理线程
  ThreadSafeQueue<ImageDataPtr> input_buffer_queue_; // 流水线输入缓冲队列
  std::thread input_feeder_thread_;                   // 输入馈送线程
  
  // 各阶段的协调线程
  std::thread seg_to_mask_thread_;         // 语义分割->Mask后处理
  std::thread mask_to_detect_thread_;      // Mask后处理->目标检测
  std::thread detect_to_track_thread_;     // 目标检测->目标跟踪
  std::thread track_to_event_thread_;      // 目标跟踪->事件判定
  std::thread event_to_final_thread_;      // 事件判定->最终结果

  ThreadSafeQueue<ImageDataPtr> final_results_; // 最终结果队列

private:
  // 输入馈送线程函数
  void input_feeder_thread_func();        // 从输入缓冲队列向第一个模块馈送数据
  
  // 各阶段的处理函数
  void seg_to_mask_thread_func();         // 处理语义分割到Mask后处理的数据流转
  void mask_to_detect_thread_func();      // 处理Mask后处理到目标检测的数据流转
  void detect_to_track_thread_func();     // 处理目标检测到目标跟踪的数据流转
  void track_to_event_thread_func();      // 处理目标跟踪到事件判定的数据流转
  void event_to_final_thread_func();      // 处理事件判定到最终结果的数据流转

public:
  // 构造函数，使用配置结构体
  PipelineManager(const PipelineConfig& config = PipelineConfig());
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
  
  // 更新流水线配置参数
  void change_params(const PipelineConfig& config);
};
