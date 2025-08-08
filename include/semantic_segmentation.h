#pragma once

#include "image_processor.h"
// #include "road_seg.h"
#include "trt_seg_model.h"
#include "event_utils.h"
#include <future>
#include <string>
#include <mutex>
#include <map>
#include <deque>
#include <condition_variable>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <atomic>

// 前向声明
struct PipelineConfig;

/**
 * 语义分割处理器
 * 继承自ImageProcessor基类，专门负责对输入图像进行语义分割处理
 * 支持多线程并发处理，每个线程独立的模型实例，保证输出顺序
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
  
  // 重写基类方法以支持顺序输出
  bool get_processed_image(ImageDataPtr &image) override;
  void start() override;
  void stop() override;

protected:
  // 重写基类的工作线程函数
  void worker_thread_func(int thread_id) override;

  // 重写基类的纯虚函数：执行语义分割算法
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
  void ordered_output_push(ImageDataPtr image);
  void ordered_output_thread_func();

  // 多线程模型实例 - 每个线程独立的模型实例
  std::vector<std::unique_ptr<PureTRTPPSeg>> road_seg_instances_;

  bool enable_seg_show_; // 是否启用分割结果可视化
  std::string seg_show_image_path_; // 分割结果图像保存路径
  int seg_show_interval_ = 200; // 分割结果保存间隔（帧数）
  
  // CUDA优化相关
  mutable std::mutex gpu_mutex_; // GPU操作互斥锁
  cv::cuda::GpuMat gpu_src_cache_; // GPU源图像缓存
  cv::cuda::GpuMat gpu_dst_cache_; // GPU目标图像缓存
  bool cuda_available_ = true; // CUDA是否可用
  
  // 顺序输出相关成员
  std::map<int64_t, ImageDataPtr> ordered_buffer_;  // 顺序缓冲区，按帧序号排序
  std::atomic<int64_t> next_expected_frame_;        // 下一个期望输出的帧序号
  std::mutex order_mutex_;                          // 顺序缓冲区的互斥锁
  std::condition_variable order_cv_;                // 条件变量
  std::thread ordered_output_thread_;               // 顺序输出线程
  std::atomic<bool> order_thread_running_;          // 顺序输出线程运行标志
  
  // 批量处理相关成员
  std::atomic<bool> batch_ready_;                   // 批次准备标志
  std::atomic<bool> batch_processing_;              // 批次处理中标志
  std::mutex batch_mutex_;                          // 批次处理互斥锁
  std::condition_variable batch_cv_;                // 批次处理条件变量
  std::vector<ImageDataPtr> current_batch_;         // 当前批次数据
  std::atomic<int> batch_completion_count_;         // 批次完成计数器
  
  // 输出顺序监控
  std::deque<int64_t> recent_output_frames_;        // 最近输出的帧序号（用于人工核验）
  static const size_t OUTPUT_WINDOW_SIZE = 10;     // 输出监控窗口大小
  std::mutex output_monitor_mutex_;                 // 输出监控互斥锁
  
  // 性能统计相关
  std::atomic<uint64_t> total_processed_images_{0}; // 总处理图像数
  std::atomic<uint64_t> total_processing_time_ms_{0}; // 总处理时间（毫秒）
  
};
