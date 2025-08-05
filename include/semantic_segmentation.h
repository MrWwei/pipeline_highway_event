#pragma once

#include "image_processor.h"
// #include "road_seg.h"
#include "trt_seg_model.h"
#include "event_utils.h"
#include <future>
#include <string>
#include <mutex>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <atomic>

// 前向声明
struct PipelineConfig;

/**
 * 语义分割处理器
 * 继承自ImageProcessor基类，专门负责对输入图像进行语义分割处理
 * 支持多线程并发处理
 */
class SemanticSegmentation : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  SemanticSegmentation(int num_threads = 1, const PipelineConfig* config = nullptr);

  // 析构函数（自定义实现，负责线程清理）
  virtual ~SemanticSegmentation();
  
  // 设置分割结果保存间隔（帧数）
  void set_seg_show_interval(int interval);
  
  // 重写参数更新方法
  virtual void change_params(const PipelineConfig &config) override;

protected:
  // 重写基类的工作线程函数以支持批量处理
  void worker_thread_func(int thread_id) override;

  // 重写基类的纯虚函数：执行语义分割算法（单个处理，保留兼容性）
  virtual void process_image(ImageDataPtr image, int thread_id) override;

  // 新增：批量处理方法
  virtual void process_images_batch(std::vector<ImageDataPtr>& images, int thread_id);

  // 重写处理开始前的准备工作
  virtual void on_processing_start(ImageDataPtr image, int thread_id) override;

  // 重写处理完成后的清理工作
  virtual void on_processing_complete(ImageDataPtr image,
                                      int thread_id) override;
  
public:



private:
  void segmentation_worker();

  std::unique_ptr<ThreadSafeQueue<ImageDataPtr>> segmentation_queue_;
  std::unique_ptr<PureTRTPPSeg> road_seg_instance_; // 支持多线程的SDK实例列表

  bool enable_seg_show_; // 是否启用分割结果可视化
  std::string seg_show_image_path_; // 分割结果图像保存路径
  int seg_show_interval_ = 200; // 分割结果保存间隔（帧数）
  
  // 批量处理相关
  static constexpr int BATCH_SIZE = 32; // 批量处理大小
  
  // CUDA优化相关
  mutable std::mutex gpu_mutex_; // GPU操作互斥锁
  cv::cuda::GpuMat gpu_src_cache_; // GPU源图像缓存
  cv::cuda::GpuMat gpu_dst_cache_; // GPU目标图像缓存
  bool cuda_available_ = true; // CUDA是否可用
  
  // 性能统计相关
  std::atomic<uint64_t> total_processed_images_{0}; // 总处理图像数
  std::atomic<uint64_t> total_batch_count_{0};      // 总批次数
  std::atomic<uint64_t> total_processing_time_ms_{0}; // 总处理时间（毫秒）
  
};
