#include "mask_postprocess.h"
#include "process_mask.h"
#include "event_utils.h"
#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>

MaskPostProcess::MaskPostProcess(int num_threads)
    : ImageProcessor(num_threads, "Mask后处理", 100, 100),
      next_expected_frame_(0),
      order_thread_running_(false) {
  // 基类已经完成了初始化工作
  std::cout << "✅ Mask后处理模块初始化完成，支持顺序输出保证" << std::endl;
}

MaskPostProcess::~MaskPostProcess() {
  // 确保正确停止
  stop();
}

// 重写 start 方法
void MaskPostProcess::start() {
  // 调用基类的启动方法
  ImageProcessor::start();
  
  // 重置状态
  next_expected_frame_.store(0);
  order_thread_running_.store(false);  // 延迟启动顺序输出线程
  
  std::cout << "✅ Mask后处理模块已启动，将在首次获取结果时启动顺序输出线程" << std::endl;
}

// 重写 stop 方法
void MaskPostProcess::stop() {
  // 先停止基类的工作线程
  ImageProcessor::stop();
  
  // 停止顺序输出线程
  if (order_thread_running_.load()) {
    order_thread_running_.store(false);
    order_cv_.notify_all();
    if (ordered_output_thread_.joinable()) {
      ordered_output_thread_.join();
    }
  }
  
  // 清空顺序缓冲区
  {
    std::lock_guard<std::mutex> lock(order_mutex_);
    ordered_buffer_.clear();
  }
  
  std::cout << "✅ Mask后处理模块已停止，顺序输出线程已关闭" << std::endl;
}

// 重写工作线程函数，处理完成后不直接推送到输出队列
void MaskPostProcess::worker_thread_func(int thread_id) {
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
    
    // 将处理完成的图像推送到顺序缓冲区，而不是直接推送到输出队列
    ordered_output_push(image);
  }
}

// 将处理完成的图像添加到顺序缓冲区
void MaskPostProcess::ordered_output_push(ImageDataPtr image) {
  std::unique_lock<std::mutex> lock(order_mutex_);
  
  // 将图像添加到顺序缓冲区
  ordered_buffer_[image->frame_idx] = image;
  
  // 通知顺序输出线程
  order_cv_.notify_one();
}

// 顺序输出线程函数
void MaskPostProcess::ordered_output_thread_func() {
  std::cout << "🔄 Mask后处理顺序输出线程启动" << std::endl;
  
  while (order_thread_running_.load() || !ordered_buffer_.empty()) {
    std::unique_lock<std::mutex> lock(order_mutex_);
    
    // 等待有数据可处理
    order_cv_.wait(lock, [this]() {
      return !ordered_buffer_.empty() || !order_thread_running_.load();
    });
    
    // 按顺序输出连续的帧
    while (!ordered_buffer_.empty()) {
      auto it = ordered_buffer_.find(next_expected_frame_.load());
      if (it != ordered_buffer_.end()) {
        // 找到了下一个期望的帧，输出它
        ImageDataPtr image = it->second;
        ordered_buffer_.erase(it);
        lock.unlock();
        
        // 推送到实际的输出队列
        output_queue_.push(image);
        
        // 更新下一个期望的帧序号
        next_expected_frame_.fetch_add(1);
        
        // 重新加锁继续处理
        lock.lock();
      } else {
        // 下一个期望的帧还没到，等待
        break;
      }
    }
  }
  
  std::cout << "🔄 Mask后处理顺序输出线程结束" << std::endl;
}

// 重写 get_processed_image 方法，启动顺序输出线程（延迟启动）
bool MaskPostProcess::get_processed_image(ImageDataPtr &image) {
  // 延迟启动顺序输出线程
  if (!order_thread_running_.load()) {
    order_thread_running_.store(true);
    ordered_output_thread_ = std::thread(&MaskPostProcess::ordered_output_thread_func, this);
    std::cout << "✅ Mask后处理顺序输出线程已启动" << std::endl;
  }
  
  // 调用基类的方法从输出队列获取图像
  return ImageProcessor::get_processed_image(image);
}

void MaskPostProcess::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "Error: Invalid image pointer in Mask post-process"
              << std::endl;
    return;
  }

  // 等待语义分割完成
  // std::cout << "⏳ [线程 " << thread_id << "] 等待语义分割完成..." << std::endl;
  while (!image->segmentation_completed) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // std::cout << "✅ [线程 " << thread_id << "] 语义分割已完成" << std::endl;

  // 检查结果是否有效
  if (image->label_map.empty()) {
    std::cerr << "⚠️ [线程 " << thread_id << "] 语义分割结果无效，跳过mask后处理" << std::endl;
    // 设置默认ROI为全图
    image->roi = cv::Rect(0, 0, image->width, image->height);
    // 标记Mask后处理完成
    image->mask_postprocess_completed = true;
    return;
  }
  // 语义分割已完成，执行Mask后处理
  perform_mask_postprocess(image, thread_id);
}

void MaskPostProcess::on_processing_start(ImageDataPtr image, int thread_id) {
  // std::cout << "🔍 Mask后处理准备开始 (线程 " << thread_id << ", 帧 " << image->frame_idx << ")" << std::endl;
}

void MaskPostProcess::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  // std::cout << "🔍 Mask后处理完成 (线程 " << thread_id << ", 帧 " << image->frame_idx << ")" << std::endl;
}

void MaskPostProcess::perform_mask_postprocess(ImageDataPtr image,
                                               int thread_id) {
  auto start_time = std::chrono::high_resolution_clock::now();
  cv::Mat mask(image->mask_height, image->mask_width, CV_8UC1);

  // 将label_map数据复制到mask中
  for (int j = 0; j < image->label_map.size(); ++j) {
    mask.data[j] = image->label_map[j];
  }

  // 去除小的白色区域
  // cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

  image->mask = remove_small_white_regions_cuda(mask);
  cv::threshold(image->mask, image->mask, 0, 255, cv::THRESH_BINARY);
  // cv::imwrite("mask_postprocess_result.png", image->mask);
  // std::cout << "✅ [线程 " << thread_id << "] Mask后处理完成，耗时: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  std::chrono::high_resolution_clock::now() - start_time)
  //                  .count()
  //           << "ms" << std::endl;
  // exit(0);
  DetectRegion detect_region = crop_detect_region_optimized(
      image->mask, image->mask.rows, image->mask.cols);
  //将resize的roi映射回原图大小
  detect_region.x1 = static_cast<int>(detect_region.x1 * image->width /
                                      static_cast<double>(image->mask_width));
  detect_region.x2 = static_cast<int>(detect_region.x2 * image->width /
                                      static_cast<double>(image->mask_width));
  detect_region.y1 = static_cast<int>(detect_region.y1 * image->height /
                                      static_cast<double>(image->mask_height));
  detect_region.y2 = static_cast<int>(detect_region.y2 * image->height /
                                      static_cast<double>(image->mask_height));
  image->roi = cv::Rect(detect_region.x1, detect_region.y1,
                        detect_region.x2 - detect_region.x1,
                        detect_region.y2 - detect_region.y1);

  // 标记Mask后处理完成
  image->mask_postprocess_completed = true;
}
