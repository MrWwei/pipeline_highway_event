#pragma once

#include "detect.h"
#include "image_processor.h"
#include "thread_safe_queue.h"
#include <string>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <map>
#include <deque>
#include <vector>
#include <memory>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

// 前向声明
struct PipelineConfig;

/**
 * 目标检测模块
 * 对图像进行目标检测，支持多线程独立实例和批量处理以提高效率
 */
class ObjectDetection : public ImageProcessor {
public:
  ObjectDetection(int num_threads = 1, const PipelineConfig* config = nullptr);
  virtual ~ObjectDetection();

  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;
  
  // 重写基类方法以支持顺序输出
  bool get_processed_image(ImageDataPtr &image) override;
  
  // 重写 start 和 stop 方法
  void start() override;
  void stop() override;

protected:
  // 重写基类的工作线程函数以支持批量处理
  void worker_thread_func(int thread_id) override;
  
  // 新增：批量处理方法
  virtual void process_images_batch(std::vector<ImageDataPtr>& images, int thread_id);

private:
  // 多线程独立实例相关
  std::vector<std::unique_ptr<xtkj::IDetect>> car_detect_instances_; // 车辆检测实例数组
  std::vector<std::unique_ptr<xtkj::IDetect>> personal_detect_instances_; // 行人检测实例数组（可选）
  PipelineConfig config_; // 保存配置
  
  // 有序输出相关
  std::map<int64_t, ImageDataPtr> ordered_buffer_; // 按帧序号排序的缓冲区
  std::mutex order_mutex_; // 保护有序缓冲区的互斥锁
  std::condition_variable order_cv_; // 条件变量用于唤醒顺序输出线程
  std::atomic<bool> order_thread_running_{false}; // 顺序输出线程运行标志
  std::thread ordered_output_thread_; // 顺序输出线程
  int64_t next_expected_frame_{0}; // 下一个期望的帧序号
  
  // 输出监控相关
  static constexpr int OUTPUT_WINDOW_SIZE = 10; // 监控窗口大小
  std::deque<int64_t> recent_output_frames_; // 最近输出的帧序号
  std::mutex output_monitor_mutex_; // 保护输出监控数据的互斥锁
  
  // 顺序输出相关方法
  void ordered_output_push(ImageDataPtr image);
  void ordered_output_thread_func();
  
  // 执行目标检测算法
  void perform_object_detection(ImageDataPtr image, int thread_id);
  
  // CUDA优化相关
  mutable std::mutex gpu_mutex_; // GPU操作互斥锁
  cv::cuda::GpuMat gpu_src_cache_; // GPU源图像缓存
  cv::cuda::GpuMat gpu_dst_cache_; // GPU目标图像缓存
  bool cuda_available_ = true; // CUDA是否可用
  
  // 批量处理相关
  static constexpr int BATCH_SIZE = 32; // 批量处理大小
  static constexpr int BATCH_TIMEOUT_MS = 50; // 批量收集超时（毫秒）
  
  // 性能统计相关
  std::atomic<uint64_t> total_processed_images_{0}; // 总处理图像数
  std::atomic<uint64_t> total_batch_count_{0};      // 总批次数
  std::atomic<uint64_t> total_processing_time_ms_{0}; // 总处理时间（毫秒）
  
};
