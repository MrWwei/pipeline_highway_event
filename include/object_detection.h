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
 * 对图像进行目标检测，批量处理以提高效率
 */
class ObjectDetection : public ImageProcessor {
public:
  ObjectDetection(int num_threads = 1, const PipelineConfig* config = nullptr);
  virtual ~ObjectDetection();

  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;

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
  
};
