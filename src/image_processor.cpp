#include "image_processor.h"
#include <chrono>
#include <iostream>
#include <thread>

ImageProcessor::ImageProcessor(int num_threads, const std::string &name)
    : running_(false), num_threads_(num_threads), processor_name_(name),
      input_queue_(100), output_queue_(100) {
  if (num_threads_ <= 0) {
    num_threads_ = 1;
  }
  std::cout << "🔍 " << processor_name_
            << "模块初始化完成 (线程数: " << num_threads_ << ")" << std::endl;
}

ImageProcessor::~ImageProcessor() { stop(); }

void ImageProcessor::start() {
  if (running_.load()) {
    return; // 已经在运行
  }

  running_.store(true);
  worker_threads_.clear();
  worker_threads_.reserve(num_threads_);

  // 启动工作线程
  for (int i = 0; i < num_threads_; ++i) {
    worker_threads_.emplace_back(&ImageProcessor::worker_thread_func, this, i);
  }

  std::cout << "🚀 " << processor_name_ << "处理线程启动 (" << num_threads_
            << "个线程)" << std::endl;
}

void ImageProcessor::stop() {
  if (!running_.load()) {
    return; // 已经停止
  }

  running_.store(false);

  // 等待所有工作线程结束
  for (auto &thread : worker_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  worker_threads_.clear();

  std::cout << "⏹️ " << processor_name_ << "处理线程已停止" << std::endl;
}

void ImageProcessor::add_image(ImageDataPtr image) {
  if (!image) {
    std::cerr << "❌ " << processor_name_ << ": 尝试添加空图像指针" << std::endl;
    return;
  }

  // 检查队列容量
  size_t current_size = input_queue_.size();
  if (current_size >= 90) { // 90% 容量警告
    std::cout << "⚠️ " << processor_name_
              << " 输入队列接近满容量: " << current_size << "/100" << std::endl;
  }

  input_queue_.push(image);
}

bool ImageProcessor::get_processed_image(ImageDataPtr &image) {
  return output_queue_.try_pop(image);
}

size_t ImageProcessor::get_queue_size() const { return input_queue_.size(); }

int ImageProcessor::get_thread_count() const { return num_threads_; }

std::string ImageProcessor::get_processor_name() const {
  return processor_name_;
}

void ImageProcessor::worker_thread_func(int thread_id) {
  std::cout << "🔄 " << processor_name_ << "工作线程 " << thread_id << " 启动"
            << std::endl;

  while (running_.load()) {
    ImageDataPtr image;

    // 从输入队列获取图像
    if (input_queue_.try_pop(image)) {
      if (!image) {
        continue; // 跳过空指针
      }

      try {
        // 处理前准备
        on_processing_start(image, thread_id);

        // 执行具体的图像处理算法
        process_image(image, thread_id);

        // 处理后清理
        on_processing_complete(image, thread_id);

        // 将处理完成的图像放入输出队列
        output_queue_.push(image);

      } catch (const std::exception &e) {
        std::cerr << "❌ " << processor_name_ << " 线程 " << thread_id
                  << " 处理图像时发生异常: " << e.what() << std::endl;
      }
    } else {
      // 没有数据时短暂休眠，避免过度消耗CPU
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  std::cout << "🏁 " << processor_name_ << "工作线程 " << thread_id << " 结束"
            << std::endl;
}
