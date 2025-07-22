#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

PipelineManager::PipelineManager(int semantic_threads,
                                 int mask_postprocess_threads,
                                 int detection_threads)
    : running_(false), next_frame_idx_(0) {
  semantic_seg_ = std::make_unique<SemanticSegmentation>(semantic_threads);
  mask_postprocess_ =
      std::make_unique<MaskPostProcess>(mask_postprocess_threads);
  object_det_ = std::make_unique<ObjectDetection>(detection_threads);
}
PipelineManager::~PipelineManager() { stop(); }

void PipelineManager::start() {
  if (running_.load()) {
    return;
  }

  running_.store(true);

  // 启动各个处理模块
  semantic_seg_->start();
  mask_postprocess_->start();
  object_det_->start();

  // 启动各阶段的协调线程
  seg_to_mask_thread_ =
      std::thread(&PipelineManager::seg_to_mask_thread_func, this);
  mask_to_detect_thread_ =
      std::thread(&PipelineManager::mask_to_detect_thread_func, this);
  detect_to_final_thread_ =
      std::thread(&PipelineManager::detect_to_final_thread_func, this);
}

void PipelineManager::stop() {
  if (!running_.load()) {
    return;
  }

  running_.store(false);

  // 停止各个处理模块
  semantic_seg_->stop();
  mask_postprocess_->stop();
  object_det_->stop();

  // 等待所有线程完成
  if (seg_to_mask_thread_.joinable()) {
    seg_to_mask_thread_.join();
  }
  if (mask_to_detect_thread_.joinable()) {
    mask_to_detect_thread_.join();
  }
  if (detect_to_final_thread_.joinable()) {
    detect_to_final_thread_.join();
  }

  std::cout << "⏹️ 停止所有管道处理线程" << std::endl;
}

void PipelineManager::add_image(const ImageDataPtr &img_data) {
  if (!running_.load() || !img_data) {
    return;
  }

  // 直接将图像数据添加到语义分割队列（流水线的第一步）
  semantic_seg_->add_image(img_data);
}

bool PipelineManager::get_final_result(ImageDataPtr &result) {
  final_results_.wait_and_pop(result);
  // std::this_thread::sleep_for(std::chrono::milliseconds(4000));

  return true;
}

void PipelineManager::print_status() const {
  // 清除屏幕
  std::cout << "\033[2J\033[1;1H";

  std::cout << "\n🔄 Pipeline 实时状态:" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            << std::endl;

  // 语义分割阶段
  std::cout << "📊 语义分割阶段" << std::endl;
  std::cout << "   输入队列: ["
            << std::string(semantic_seg_->get_queue_size() > 0 ? "🟢" : "⚪")
            << "] " << semantic_seg_->get_queue_size() << std::endl;
  std::cout << "   输出队列: ["
            << std::string(semantic_seg_->get_output_queue_size() > 0 ? "🟢"
                                                                      : "⚪")
            << "] " << semantic_seg_->get_output_queue_size() << std::endl;

  // Mask后处理阶段
  std::cout << "\n📊 Mask后处理阶段" << std::endl;
  std::cout << "   输入队列: ["
            << std::string(mask_postprocess_->get_queue_size() > 0 ? "🟢"
                                                                   : "⚪")
            << "] " << mask_postprocess_->get_queue_size() << std::endl;
  std::cout << "   输出队列: ["
            << std::string(mask_postprocess_->get_output_queue_size() > 0
                               ? "🟢"
                               : "⚪")
            << "] " << mask_postprocess_->get_output_queue_size() << std::endl;

  // 目标检测阶段
  std::cout << "\n📊 目标检测阶段" << std::endl;
  std::cout << "   输入队列: ["
            << std::string(object_det_->get_queue_size() > 0 ? "🟢" : "⚪")
            << "] " << object_det_->get_queue_size() << std::endl;
  std::cout << "   输出队列: ["
            << std::string(object_det_->get_output_queue_size() > 0 ? "🟢"
                                                                    : "⚪")
            << "] " << object_det_->get_output_queue_size() << std::endl;

  // 最终结果队列
  std::cout << "\n📊 最终结果" << std::endl;
  std::cout << "   结果队列: ["
            << std::string(final_results_.size() > 0 ? "🟢" : "⚪") << "] "
            << final_results_.size() << std::endl;

  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            << std::endl;
}

void PipelineManager::print_thread_info() const {
  std::cout << "\n🧵 线程配置信息:" << std::endl;
  std::cout << "   语义分割线程数: " << semantic_seg_->get_thread_count()
            << std::endl;
  std::cout << "   Mask后处理线程数: " << mask_postprocess_->get_thread_count()
            << std::endl;
  std::cout << "   目标检测线程数: " << object_det_->get_thread_count()
            << std::endl;
  std::cout << "   协调器线程数: 1" << std::endl;
  std::cout << "   总工作线程数: "
            << (semantic_seg_->get_thread_count() +
                mask_postprocess_->get_thread_count() +
                object_det_->get_thread_count() + 1)
            << std::endl;
}

// 语义分割->Mask后处理的数据流转
void PipelineManager::seg_to_mask_thread_func() {
  auto last_status_time = std::chrono::steady_clock::now();

  while (running_.load()) {
    bool has_work = false;
    size_t processed = 0;

    // 每秒更新一次状态
    auto current_time = std::chrono::steady_clock::now();
    if (current_time - last_status_time > std::chrono::milliseconds(1000)) {
      print_status();
      last_status_time = current_time;
    }

    // 检查输出队列
    if (semantic_seg_->get_output_queue_size() > 0) {
      ImageDataPtr seg_result;

      // 批量处理数据
      while (semantic_seg_->get_processed_image(seg_result)) {
        if (seg_result) {
          has_work = true;
          processed++;
          mask_postprocess_->add_image(seg_result);
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

// Mask后处理->目标检测的数据流转
void PipelineManager::mask_to_detect_thread_func() {

  while (running_.load()) {
    bool has_work = false;
    size_t processed = 0;

    // 检查输出队列
    if (mask_postprocess_->get_output_queue_size() > 0) {
      ImageDataPtr mask_result;

      // 批量处理数据
      while (mask_postprocess_->get_processed_image(mask_result)) {
        if (mask_result) {
          has_work = true;
          processed++;
          object_det_->add_image(mask_result);
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

// 目标检测->最终结果的数据流转
void PipelineManager::detect_to_final_thread_func() {
  while (running_.load()) {
    bool has_work = false;
    size_t processed = 0;

    // 检查输出队列
    if (object_det_->get_output_queue_size() > 0) {
      ImageDataPtr detect_result;

      // 批量处理数据
      while (object_det_->get_processed_image(detect_result)) {
        if (detect_result) {
          has_work = true;
          processed++;

          // 使用互斥锁保护对pending_results_的访问
          std::lock_guard<std::mutex> lock(pending_results_mutex_);

          // 将结果添加到pending_results_中
          pending_results_[detect_result->frame_idx] = detect_result;

          // 检查是否有可以按序输出的结果
          while (pending_results_.find(next_frame_idx_) !=
                 pending_results_.end()) {
            auto next_result = pending_results_[next_frame_idx_];
            final_results_.push(next_result);
            pending_results_.erase(next_frame_idx_);
            next_frame_idx_++;
          }
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}
