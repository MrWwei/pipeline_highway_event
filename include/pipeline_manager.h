#pragma once

#include "image_data.h"
#include "mask_postprocess.h"
#include "object_detection.h"
#include "object_tracking.h"
#include "box_filter.h"
#include "semantic_segmentation.h"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <string>

/**
 * 流水线管理器配置参数
 */
struct PipelineConfig {
    // 线程配置
    int semantic_threads = 2;              // 语义分割线程数
    int mask_postprocess_threads = 1;      // Mask后处理线程数
    int detection_threads = 2;             // 目标检测线程数
    int tracking_threads = 1;              // 目标跟踪线程数
    int box_filter_threads = 1;            // 目标框筛选线程数
    
    // 模块开关配置
    bool enable_mask_postprocess = true;  // 启用Mask后处理模块
    bool enable_detection = true;          // 启用目标检测模块
    bool enable_tracking = true;           // 启用目标跟踪模块
    bool enable_box_filter = true;         // 启用目标框筛选模块
    
    // 语义分割模型配置
    std::string seg_model_path = "seg_model";               // 语义分割模型路径
    bool seg_enable_show = false;                           // 是否启用分割结果可视化
    std::string seg_show_image_path = "./segmentation_results/"; // 分割结果图像保存路径
    
    // 目标检测算法配置
    std::string det_algor_name = "object_detect";           // 算法名称
    std::string det_model_path = "car_detect.onnx";         // 目标检测模型路径
    int det_img_size = 640;                                 // 输入图像尺寸
    float det_conf_thresh = 0.25f;                          // 置信度阈值
    float det_iou_thresh = 0.2f;                            // NMS IoU阈值
    int det_max_batch_size = 16;                            // 最大批处理大小
    int det_min_opt = 1;                                    // 最小优化尺寸
    int det_mid_opt = 16;                                   // 中等优化尺寸
    int det_max_opt = 32;                                   // 最大优化尺寸
    int det_is_ultralytics = 1;                             // 是否使用Ultralytics格式
    int det_gpu_id = 0;                                     // GPU设备ID
    
    // 目标框筛选配置
    float box_filter_top_fraction = 4.0f / 7.0f;           // 筛选区域上边界比例
    float box_filter_bottom_fraction = 8.0f / 9.0f;        // 筛选区域下边界比例
    float times_car_width = 3.0f;                          // 车宽倍数，用于计算车道线位置
    
    // 队列配置
    int final_result_queue_capacity = 500; // 最终结果队列容量
};

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
  std::unique_ptr<BoxFilter> box_filter_;

  std::atomic<bool> running_;
  // 为每个阶段创建独立的协调线程
  std::thread seg_to_mask_thread_;         // 语义分割->Mask后处理
  std::thread mask_to_detect_thread_;      // Mask后处理->目标检测（直接到目标跟踪）
  std::thread track_to_filter_thread_;     // 目标跟踪->目标框筛选
  std::thread filter_to_final_thread_;     // 目标框筛选->最终结果

  ThreadSafeQueue<ImageDataPtr> final_results_; // 最终结果队列
  std::map<uint64_t, ImageDataPtr> pending_results_; // 用于暂存未按序的结果
  std::mutex pending_results_mutex_;
  uint64_t next_frame_idx_; // 下一个应该输出的帧序号

private:
  // 各阶段的处理函数
  void seg_to_mask_thread_func();         // 处理语义分割到Mask后处理的数据流转
  void mask_to_detect_thread_func();      // 处理Mask后处理到目标检测并直接流转到目标跟踪
  void track_to_filter_thread_func();     // 处理目标跟踪到目标框筛选的数据流转
  void filter_to_final_thread_func();     // 处理目标框筛选到最终结果的数据流转

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

private:
  // 协调器线程函数
  void coordinator_thread_func();
};
