#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

PipelineManager::PipelineManager(int semantic_threads, int detection_threads)
    : running_(false) {
  semantic_seg_ = std::make_unique<SemanticSegmentation>(semantic_threads);
  object_det_ = std::make_unique<ObjectDetection>(detection_threads);
  std::cout << "🏗️  流水线管理器初始化完成" << std::endl;
  std::cout << "   📊 语义分割线程数: " << semantic_threads << std::endl;
  std::cout << "   📊 目标检测线程数: " << detection_threads << std::endl;
}
PipelineManager::~PipelineManager() { stop(); }

void PipelineManager::start() {
  if (running_.load()) {
    return;
  }

  running_.store(true);

  // 启动各个处理模块
  semantic_seg_->start();
  object_det_->start();

  // 启动协调器线程
  coordinator_thread_ =
      std::thread(&PipelineManager::coordinator_thread_func, this);

  std::cout << "🚀 流水线启动完成" << std::endl;
}

void PipelineManager::stop() {
  if (!running_.load()) {
    return;
  }

  running_.store(false);

  // 停止各个处理模块
  semantic_seg_->stop();
  object_det_->stop();

  // 停止协调器线程
  if (coordinator_thread_.joinable()) {
    coordinator_thread_.join();
  }

  std::cout << "⏹️  流水线停止完成" << std::endl;
}

void PipelineManager::add_image(const std::string &image_path) {
  if (!running_.load()) {
    std::cout << "❌ 流水线未启动，无法添加图像" << std::endl;
    return;
  }

  // 创建新的图像数据
  auto image_data = std::make_shared<ImageData>(image_path);

  // 只将图像添加到语义分割队列（流水线的第一步）
  semantic_seg_->add_image(image_data);

  //   std::cout << "📤 图像添加到流水线: " << image_path << std::endl;
}

bool PipelineManager::get_final_result(ImageDataPtr &result) {
  return final_results_.try_pop(result);
}

void PipelineManager::print_status() const {
  std::cout << "\n📊 流水线状态报告:" << std::endl;
  std::cout << "   语义分割队列: " << semantic_seg_->get_queue_size()
            << "/100 (满: "
            << (semantic_seg_->get_queue_size() >= 100 ? "是" : "否") << ")"
            << std::endl;
  std::cout << "   目标检测队列: " << object_det_->get_queue_size()
            << "/100 (满: "
            << (object_det_->get_queue_size() >= 100 ? "是" : "否") << ")"
            << std::endl;
  std::cout << "   最终结果队列: " << final_results_.size()
            << "/100 (满: " << (final_results_.size() >= 100 ? "是" : "否")
            << ")" << std::endl;
}

void PipelineManager::print_thread_info() const {
  std::cout << "\n🧵 线程配置信息:" << std::endl;
  std::cout << "   语义分割线程数: " << semantic_seg_->get_thread_count()
            << std::endl;
  std::cout << "   目标检测线程数: " << object_det_->get_thread_count()
            << std::endl;
  std::cout << "   协调器线程数: 1" << std::endl;
  std::cout << "   总工作线程数: "
            << (semantic_seg_->get_thread_count() +
                object_det_->get_thread_count() + 1)
            << std::endl;
}

void PipelineManager::coordinator_thread_func() {
  std::cout << "🔄 流水线协调器线程启动" << std::endl;

  while (running_.load()) {
    bool has_work = false;

    // 第一步：处理语义分割的输出，将完成的图像传递给目标检测
    ImageDataPtr seg_result;
    while (semantic_seg_->get_processed_image(seg_result)) {
      has_work = true;
      std::cout << "📋 协调器收到语义分割结果: " << seg_result->image_path
                << std::endl;

      // 将完成语义分割的图像传递给目标检测（流水线的第二步）
      object_det_->add_image(seg_result);
      std::cout << "🔄 图像从语义分割传递到目标检测: " << seg_result->image_path
                << std::endl;
    }

    // 第二步：处理目标检测的输出，这些是完全处理完成的结果
    ImageDataPtr final_result;
    while (object_det_->get_processed_image(final_result)) {
      has_work = true;
      std::cout << "📋 协调器收到目标检测结果: " << final_result->image_path
                << std::endl;

      // 验证图像是否完全处理完成
      if (final_result->is_fully_processed()) {
        final_results_.push(final_result);
        std::cout << "🎉 图像完全处理完成: " << final_result->image_path
                  << std::endl;
      } else {
        std::cout << "⚠️  图像处理不完整: "
                  << final_result->image_path << std::endl;
      }
    }

    if (!has_work) {
      // 短暂休息，避免忙等待
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  std::cout << "⏹️  流水线协调器线程停止" << std::endl;
}
