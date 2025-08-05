#pragma once

#include "detect.h"
#include "image_processor.h"
#include "thread_safe_queue.h"
#include <string>
#include <mutex>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

// 前向声明
struct PipelineConfig;

/**
 * 目标检测模块
 * 对图像进行目标检测，支持批量处理以提高效率
 */
class ObjectDetection : public ImageProcessor {
public:
  ObjectDetection(int num_threads = 1, const PipelineConfig* config = nullptr);
  virtual ~ObjectDetection();

  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;

protected:
  // 重写基类的工作线程函数以支持批量处理
  void worker_thread_func(int thread_id) override;
  
  // 新增：批量处理方法
  virtual void process_images_batch(std::vector<ImageDataPtr>& images, int thread_id);

private:
  xtkj::IDetect *car_detect_instance_; // 目标检测实例
  xtkj::IDetect *personal_detect_instance_; // 个人物体检测实例
  PipelineConfig config_; // 保存配置指针
  
  // 执行目标检测算法
  void perform_object_detection(ImageDataPtr image, int thread_id);
  
  // CUDA优化相关
  mutable std::mutex gpu_mutex_; // GPU操作互斥锁
  cv::cuda::GpuMat gpu_src_cache_; // GPU源图像缓存
  cv::cuda::GpuMat gpu_dst_cache_; // GPU目标图像缓存
  bool cuda_available_ = true; // CUDA是否可用
  
  // 批量处理相关
  static constexpr int BATCH_SIZE = 8; // 批量处理大小
  
  // 性能统计相关
  std::atomic<uint64_t> total_processed_images_{0}; // 总处理图像数
  std::atomic<uint64_t> total_batch_count_{0};      // 总批次数
  std::atomic<uint64_t> total_processing_time_ms_{0}; // 总处理时间（毫秒）
  
};
