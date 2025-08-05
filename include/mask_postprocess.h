#pragma once

#include "image_processor.h"
#include <opencv2/opencv.hpp>
#include <map>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * Mask后处理模块
 * 负责对语义分割结果进行后处理，如去除小的白色区域等
 * 支持多线程处理的同时保证输出顺序
 */
class MaskPostProcess : public ImageProcessor {
public:
  explicit MaskPostProcess(int num_threads);
  ~MaskPostProcess();

  // 重写基类方法以支持顺序输出
  bool get_processed_image(ImageDataPtr &image) override;
  void start() override;
  void stop() override;

protected:
  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;
  void worker_thread_func(int thread_id) override;

private:
  void perform_mask_postprocess(ImageDataPtr image, int thread_id);
  void ordered_output_push(ImageDataPtr image);
  void ordered_output_thread_func();

  // 顺序输出相关成员
  std::map<int64_t, ImageDataPtr> ordered_buffer_;  // 顺序缓冲区，按帧序号排序
  std::atomic<int64_t> next_expected_frame_;        // 下一个期望输出的帧序号
  std::mutex order_mutex_;                          // 顺序缓冲区的互斥锁
  std::condition_variable order_cv_;                // 条件变量
  std::thread ordered_output_thread_;               // 顺序输出线程
  std::atomic<bool> order_thread_running_;          // 顺序输出线程运行标志
};
