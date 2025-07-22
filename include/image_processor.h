#pragma once

#include "image_data.h"
#include "thread_safe_queue.h"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

/**
 * 图像处理器基类
 * 定义了图像处理模块的通用接口和行为
 * 提供多线程处理能力的基础框架
 */
class ImageProcessor {
protected:
  std::atomic<bool> running_;
  std::vector<std::thread> worker_threads_;
  ThreadSafeQueue<ImageDataPtr> input_queue_;
  ThreadSafeQueue<ImageDataPtr> output_queue_;
  int num_threads_;
  std::string processor_name_;

public:
  // 构造函数，可指定线程数量和处理器名称
  ImageProcessor(int num_threads = 1,
                 const std::string &name = "ImageProcessor");

  // 虚析构函数，确保派生类正确销毁
  virtual ~ImageProcessor();

  // 启动处理线程
  virtual void start();

  // 停止处理线程
  virtual void stop();

  // 添加图像到处理队列
  virtual void add_image(ImageDataPtr image);

  // 获取处理完成的图像
  virtual bool get_processed_image(ImageDataPtr &image);

  // 获取输入队列大小
  virtual size_t get_queue_size() const;
  // 获取输出队列大小
  virtual size_t get_output_queue_size() const;

  // 获取当前活跃线程数量
  virtual int get_thread_count() const;

  // 获取处理器名称
  virtual std::string get_processor_name() const;

protected:
  // 工作线程函数 - 调用派生类的具体处理方法
  void worker_thread_func(int thread_id);

  // 纯虚函数：具体的图像处理算法，由派生类实现
  virtual void process_image(ImageDataPtr image, int thread_id) = 0;

  // 虚函数：处理前的准备工作，派生类可以重写
  virtual void on_processing_start(ImageDataPtr image, int thread_id) {}

  // 虚函数：处理后的清理工作，派生类可以重写
  virtual void on_processing_complete(ImageDataPtr image, int thread_id) {}
};
