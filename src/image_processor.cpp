#include "image_processor.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <future>

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

  std::cout << "  停止 " << processor_name_ << " 处理器..." << std::endl;
  running_.store(false);

  // 向输入队列推送空数据来唤醒阻塞的工作线程
  for (int i = 0; i < num_threads_; ++i) {
    input_queue_.push(nullptr);
  }

  std::cout << "  等待 " << processor_name_ << " 工作线程退出..." << std::endl;
  
  // 等待所有工作线程结束，添加超时机制
  for (auto &thread : worker_threads_) {
    if (thread.joinable()) {
      // 使用 future 来实现超时等待
      auto future = std::async(std::launch::async, [&thread]() {
        if (thread.joinable()) {
          thread.join();
        }
      });
      
      if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        std::cout << "  ⚠️ " << processor_name_ << " 工作线程超时，强制分离" << std::endl;
        thread.detach();
      }
    }
  }
  worker_threads_.clear();

  // 清理输入和输出队列
  std::cout << "  清理 " << processor_name_ << " 队列..." << std::endl;
  input_queue_.clear();
  output_queue_.clear();

  std::cout << "⏹️ " << processor_name_ << "处理线程已停止" << std::endl;
}
void ImageProcessor::change_params(const PipelineConfig &config) {
  // 默认实现为空，派生类可以重写以更新参数
  std::cout << "🔧 " << processor_name_ << " 更新参数 (默认实现)" << std::endl;
}
void ImageProcessor::add_image(ImageDataPtr image) {
  if (!image) {
    std::cerr << "❌ " << processor_name_ << ": 尝试添加空图像指针" << std::endl;
    return;
  }

  // 检查队列容量
  // size_t current_size = input_queue_.size();
  // if (current_size >= 90) { // 90% 容量警告
  //   std::cout << "⚠️ " << processor_name_
  //             << " 输入队列接近满容量: " << current_size << "/100" << std::endl;
  // }
  input_queue_.push(image);
}

bool ImageProcessor::get_processed_image(ImageDataPtr &image) {
  if (output_queue_.empty()) {
    return false;
  }
  output_queue_.wait_and_pop(image);
  return true;
}

size_t ImageProcessor::get_queue_size() const { return input_queue_.size(); }
size_t ImageProcessor::get_output_queue_size() const {
  return output_queue_.size();
}

int ImageProcessor::get_thread_count() const { return num_threads_; }

std::string ImageProcessor::get_processor_name() const {
  return processor_name_;
}

void ImageProcessor::worker_thread_func(int thread_id) {
  std::cout << "🔄 " << processor_name_ << "工作线程 " << thread_id << " 启动"
            << std::endl;

  while (running_.load()) {
    ImageDataPtr image;
    
    // 阻塞等待队列中的数据
    input_queue_.wait_and_pop(image);
    
    // 检查是否是停止信号（空数据）
    if (!image) {
      if (!running_.load()) {
        break;  // 收到停止信号，退出循环
      }
      continue;  // 忽略空数据，继续处理
    }
    
    on_processing_start(image, thread_id);

    // 执行具体的图像处理算法
    process_image(image, thread_id);

    // 处理后清理
    on_processing_complete(image, thread_id);
    output_queue_.push(image);
  }
  
  std::cout << "🔄 " << processor_name_ << "工作线程 " << thread_id << " 退出"
            << std::endl;
}
