#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <map>
#include <thread>
#include <future> // for thread timeout handling
/**
 * 创建不同处理模块的实例
 * add_image 方法将图像添加到相应的处理队列中
 */

PipelineManager::PipelineManager(const PipelineConfig& config)
    : running_(false), 
      input_buffer_queue_(100), // 输入缓冲队列大小为100
      final_results_(config.final_result_queue_capacity), 
      config_(config) {
  
  // 根据开关决定是否创建语义分割模块
  if (config.enable_segmentation) {
    semantic_seg_ = std::make_unique<SemanticSegmentation>(config.semantic_threads, &config);
  }
  
  // 根据开关决定是否创建模块
  // 注意：mask后处理和event_determine依赖于语义分割，如果语义分割禁用，它们也必须禁用
  if (config.enable_segmentation && config.enable_mask_postprocess) {
    mask_postprocess_ = std::make_unique<MaskPostProcess>(config.mask_postprocess_threads);
  }
  if (config.enable_detection) {
    object_det_ = std::make_unique<ObjectDetection>(config.detection_threads, &config);
  }
  
  // 目标跟踪依赖于目标检测，如果检测禁用，跟踪也必须禁用
  if (config.enable_detection && config.enable_tracking) {
    object_track_ = std::make_unique<ObjectTracking>(config.tracking_threads);
  }
  
  // event_determine依赖于语义分割的mask结果，如果语义分割禁用，event_determine也必须禁用
  if (config.enable_segmentation && config.enable_event_determine) {
    event_determine_ = std::make_unique<EventDetermine>(config.event_determine_threads, &config);
  }
}
PipelineManager::~PipelineManager() { stop(); }

void PipelineManager::start() {
  if (running_.load()) {
    return;
  }

  running_.store(true);
  
  // 重置结果队列状态
  final_results_.reset();

  // 启动各个处理模块（根据配置）
  if (config_.enable_segmentation && semantic_seg_) {
    semantic_seg_->start();
    std::cout << "🔄 语义分割模块已启动，线程数: " << config_.semantic_threads << std::endl;
  } else {
    std::cout << "⚠️ 语义分割模块已禁用" << std::endl;
  }
  
  if (config_.enable_segmentation && config_.enable_mask_postprocess && mask_postprocess_) {
    mask_postprocess_->start();
    std::cout << "🔍 Mask后处理模块已启用" << std::endl;
  } else {
    if (!config_.enable_segmentation) {
      std::cout << "⚠️ Mask后处理模块已禁用 (语义分割已禁用)" << std::endl;
    } else {
      std::cout << "⚠️ Mask后处理模块已禁用" << std::endl;
    }
  }
  
  if (config_.enable_detection && object_det_) {
    object_det_->start();
    std::cout << "🔍 目标检测模块已启用" << std::endl;
  } else {
    std::cout << "⚠️ 目标检测模块已禁用" << std::endl;
  }
  
  if (config_.enable_detection && config_.enable_tracking && object_track_) {
    object_track_->start();
    std::cout << "🎯 目标跟踪模块已启用" << std::endl;
  } else {
    if (!config_.enable_detection) {
      std::cout << "⚠️ 目标跟踪模块已禁用 (目标检测已禁用)" << std::endl;
    } else {
      std::cout << "⚠️ 目标跟踪模块已禁用" << std::endl;
    }
  }
  
  if (config_.enable_segmentation && config_.enable_event_determine && event_determine_) {
    event_determine_->start();
    std::cout << "📋 事件判定模块已启用" << std::endl;
  } else {
    if (!config_.enable_segmentation) {
      std::cout << "⚠️ 事件判定模块已禁用 (语义分割已禁用)" << std::endl;
    } else {
      std::cout << "⚠️ 事件判定模块已禁用" << std::endl;
    }
  }

  // 启动各阶段的协调线程
  input_feeder_thread_ =
      std::thread(&PipelineManager::input_feeder_thread_func, this);
  seg_to_mask_thread_ =
      std::thread(&PipelineManager::seg_to_mask_thread_func, this);
  mask_to_detect_thread_ =
      std::thread(&PipelineManager::mask_to_detect_thread_func, this);
  detect_to_track_thread_ =
      std::thread(&PipelineManager::detect_to_track_thread_func, this);
  track_to_event_thread_ =
      std::thread(&PipelineManager::track_to_event_thread_func, this);
  event_to_final_thread_ =
      std::thread(&PipelineManager::event_to_final_thread_func, this);
}

void PipelineManager::stop() {
  if (!running_.load()) {
    return;
  }

  std::cout << "开始停止流水线..." << std::endl;
  running_.store(false);

  // 停止各个处理模块
  if (semantic_seg_) {
    std::cout << "停止语义分割模块..." << std::endl;
    semantic_seg_->stop();
  }
  
  if (mask_postprocess_) {
    std::cout << "停止Mask后处理模块..." << std::endl;
    mask_postprocess_->stop();
  }
  
  if (object_det_) {
    std::cout << "停止目标检测模块..." << std::endl;
    object_det_->stop();
  }
  
  if (object_track_) {
    std::cout << "停止目标跟踪模块..." << std::endl;
    object_track_->stop();
  }
  
  if (event_determine_) {
    std::cout << "停止事件判定模块..." << std::endl;
    event_determine_->stop();
  }

  std::cout << "等待协调线程结束..." << std::endl;
  
  // 等待所有线程完成，添加超时机制
  auto join_with_timeout = [](std::thread& t, const std::string& name) {
    if (t.joinable()) {
      std::cout << "等待 " << name << " 线程..." << std::endl;
      
      // 使用 future 来实现超时等待
      auto future = std::async(std::launch::async, [&t]() {
        if (t.joinable()) {
          t.join();
        }
      });
      
      if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        std::cout << "⚠️ " << name << " 线程超时，强制分离" << std::endl;
        t.detach();
      } else {
        std::cout << "✅ " << name << " 线程已正常退出" << std::endl;
      }
    }
  };
  
  join_with_timeout(input_feeder_thread_, "input_feeder");
  join_with_timeout(seg_to_mask_thread_, "seg_to_mask");
  join_with_timeout(mask_to_detect_thread_, "mask_to_detect");
  join_with_timeout(detect_to_track_thread_, "detect_to_track");
  join_with_timeout(track_to_event_thread_, "track_to_event");
  join_with_timeout(event_to_final_thread_, "event_to_final");

  // 清理流水线管理器自己的队列和资源
  std::cout << "清理流水线队列和缓存..." << std::endl;
  input_buffer_queue_.shutdown();
  input_buffer_queue_.clear();
  final_results_.shutdown(); // 关闭结果队列，唤醒所有等待的线程
  final_results_.clear();

  std::cout << "⏹️ 停止所有管道处理线程" << std::endl;
  
}

void PipelineManager::add_image(const ImageDataPtr &img_data) {
  if (!running_.load() || !img_data) {
    return;
  }

  // 将图像数据添加到输入缓冲队列，由输入馈送线程负责分发到具体模块
  input_buffer_queue_.push(img_data);
}

// 输入馈送线程：从输入缓冲队列向第一个启用的模块馈送数据
void PipelineManager::input_feeder_thread_func() {
  std::cout << "input_feeder_thread 已启动" << std::endl;
  
  while (running_.load()) {
    ImageDataPtr img_data;
    
    // 从输入缓冲队列获取数据
    if (input_buffer_queue_.wait_and_pop(img_data)) {
      if (!img_data) {
        if (!running_.load()) {
          break; // 收到停止信号
        }
        continue;
      }
      
      // 根据配置决定流转路径，简化为线性流水线
      if (config_.enable_segmentation && semantic_seg_) {
        // 启用语义分割：将图像数据添加到语义分割队列（流水线的第一步）
        semantic_seg_->add_image(img_data);
      } else if (config_.enable_mask_postprocess && mask_postprocess_) {
        // 跳过语义分割，直接到Mask后处理（设置默认分割结果）
        img_data->roi = cv::Rect(0, 0, img_data->width, img_data->height);
        mask_postprocess_->add_image(img_data);
      } else if (config_.enable_detection && object_det_) {
        // 跳过语义分割和Mask后处理：直接进入检测阶段
        img_data->roi = cv::Rect(0, 0, img_data->width, img_data->height);
        object_det_->add_image(img_data);
      } else if (config_.enable_tracking && object_track_) {
        // 跳过检测，直接到跟踪（设置空的检测结果）
        img_data->roi = cv::Rect(0, 0, img_data->width, img_data->height);
        img_data->detection_results.clear();
        object_track_->add_image(img_data);
      } else if (config_.enable_event_determine && event_determine_) {
        // 跳过前面所有模块，直接到事件判定
        img_data->roi = cv::Rect(0, 0, img_data->width, img_data->height);
        img_data->detection_results.clear();
        img_data->track_results.clear();
        event_determine_->add_image(img_data);
      } else {
        // 跳过所有处理，直接到最终结果
        img_data->roi = cv::Rect(0, 0, img_data->width, img_data->height);
        img_data->detection_results.clear();
        img_data->track_results.clear();
        final_results_.push(img_data);
      }
    } else {
      // 队列可能已关闭或出错
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  
  std::cout << "input_feeder_thread 已退出" << std::endl;
}

bool PipelineManager::get_final_result(ImageDataPtr &result) {
  return final_results_.wait_and_pop(result);
}

void PipelineManager::print_status() const {
  // 清除屏幕
  std::cout << "\033[2J\033[1;1H";

  std::cout << "\n🔄 Pipeline 实时状态:" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            << std::endl;

  // 输入缓冲队列状态
  std::cout << "📥 输入缓冲队列 [启用]" << std::endl;
  std::cout << "   缓冲队列: ["
            << std::string(input_buffer_queue_.size() > 0 ? "🟢" : "⚪")
            << "] " << input_buffer_queue_.size() << "/100" << std::endl;

  // 语义分割阶段
  if (config_.enable_segmentation && semantic_seg_) {
    std::cout << "📊 语义分割阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(semantic_seg_->get_queue_size() > 0 ? "🟢" : "⚪")
              << "] " << semantic_seg_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(semantic_seg_->get_output_queue_size() > 0 ? "🟢"
                                                                        : "⚪")
              << "] " << semantic_seg_->get_output_queue_size() << std::endl;
  } else {
    std::cout << "📊 语义分割阶段 [已禁用]" << std::endl;
  }

  // Mask后处理阶段
  if (config_.enable_segmentation && config_.enable_mask_postprocess && mask_postprocess_) {
    std::cout << "\n📊 Mask后处理阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(mask_postprocess_->get_queue_size() > 0 ? "🟢"
                                                                     : "⚪")
              << "] " << mask_postprocess_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(mask_postprocess_->get_output_queue_size() > 0
                                 ? "🟢"
                                 : "⚪")
              << "] " << mask_postprocess_->get_output_queue_size() << std::endl;
  } else {
    if (!config_.enable_segmentation) {
      std::cout << "\n📊 Mask后处理阶段 [已禁用 - 语义分割已禁用]" << std::endl;
    } else {
      std::cout << "\n📊 Mask后处理阶段 [已禁用]" << std::endl;
    }
  }

  // 目标检测阶段
  if (config_.enable_detection && object_det_) {
    std::cout << "\n📊 目标检测阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(object_det_->get_queue_size() > 0 ? "🟢" : "⚪")
              << "] " << object_det_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(object_det_->get_output_queue_size() > 0 ? "🟢"
                                                                      : "⚪")
              << "] " << object_det_->get_output_queue_size() << std::endl;
  } else {
    std::cout << "\n📊 目标检测阶段 [已禁用]" << std::endl;
  }

  // 目标跟踪阶段
  if (config_.enable_detection && config_.enable_tracking && object_track_) {
    std::cout << "\n🎯 目标跟踪阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(object_track_->get_queue_size() > 0 ? "🟢" : "⚪")
              << "] " << object_track_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(object_track_->get_output_queue_size() > 0 ? "🟢"
                                                                        : "⚪")
              << "] " << object_track_->get_output_queue_size() << std::endl;
  } else {
    if (!config_.enable_detection) {
      std::cout << "\n🎯 目标跟踪阶段 [已禁用 - 目标检测已禁用]" << std::endl;
    } else {
      std::cout << "\n🎯 目标跟踪阶段 [已禁用]" << std::endl;
    }
  }

  // 事件判定阶段
  if (config_.enable_segmentation && config_.enable_event_determine && event_determine_) {
    std::cout << "\n📦 事件判定阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(event_determine_->get_queue_size() > 0 ? "🟢" : "⚪")
              << "] " << event_determine_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(event_determine_->get_output_queue_size() > 0 ? "🟢"
                                                                      : "⚪")
              << "] " << event_determine_->get_output_queue_size() << std::endl;
  } else {
    if (!config_.enable_segmentation) {
      std::cout << "\n📦 事件判定阶段 [已禁用 - 语义分割已禁用]" << std::endl;
    } else {
      std::cout << "\n📦 事件判定阶段 [已禁用]" << std::endl;
    }
  }

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
  
  if (semantic_seg_) {
    std::cout << "   语义分割线程数: " << semantic_seg_->get_thread_count()
              << std::endl;
  } else {
    std::cout << "   语义分割线程数: 0 (已禁用)" << std::endl;
  }
  
  if (mask_postprocess_) {
    std::cout << "   Mask后处理线程数: " << mask_postprocess_->get_thread_count()
              << std::endl;
  } else {
    std::cout << "   Mask后处理线程数: 0 (已禁用)" << std::endl;
  }
  
  if (object_det_) {
    std::cout << "   目标检测线程数: " << object_det_->get_thread_count()
              << std::endl;
  } else {
    std::cout << "   目标检测线程数: 0 (已禁用)" << std::endl;
  }
  
  if (object_track_) {
    std::cout << "   目标跟踪线程数: " << object_track_->get_thread_count()
              << std::endl;
  } else {
    std::cout << "   目标跟踪线程数: 0 (已禁用)" << std::endl;
  }
  
  if (event_determine_) {
    std::cout << "   事件判定线程数: " << event_determine_->get_thread_count()
              << std::endl;
  } else {
    std::cout << "   目标框筛选线程数: 0 (已禁用)" << std::endl;
  }
  
  std::cout << "   协调器线程数: 5" << std::endl;
  
  int total_threads = 5; // 协调器线程数
  if (semantic_seg_) total_threads += semantic_seg_->get_thread_count();
  if (mask_postprocess_) total_threads += mask_postprocess_->get_thread_count();
  if (object_det_) total_threads += object_det_->get_thread_count();
  if (object_track_) total_threads += object_track_->get_thread_count();
  if (event_determine_) total_threads += event_determine_->get_thread_count();
  
  std::cout << "   总工作线程数: " << total_threads << std::endl;
}

void PipelineManager::change_params(const PipelineConfig& config) {
  config_ = config;
  
  // 更新各个模块的配置
  if (semantic_seg_) {
    semantic_seg_->change_params(config);
  }
  if (mask_postprocess_) {
    mask_postprocess_->change_params(config);
  }
  if (object_det_) {
    object_det_->change_params(config);
  }
  if (object_track_) {
    object_track_->change_params(config);
  }
  if (event_determine_) {
    event_determine_->change_params(config);
  }
}

// 语义分割->Mask后处理的数据流转
void PipelineManager::seg_to_mask_thread_func() {
  // 如果语义分割被禁用，此线程直接退出
  if (!config_.enable_segmentation || !semantic_seg_) {
    std::cout << "seg_to_mask_thread 已跳过（语义分割未启用）" << std::endl;
    return;
  }

  while (running_.load()) {
    ImageDataPtr seg_result;
    
    // 等待语义分割结果
    if (semantic_seg_->get_processed_image(seg_result)) {
      if (seg_result) {
        if (config_.enable_mask_postprocess && mask_postprocess_) {
          // 传递给Mask后处理模块
          mask_postprocess_->add_image(seg_result);
        } else if (config_.enable_detection && object_det_) {
          // 跳过Mask后处理，直接传递给目标检测
          seg_result->roi = cv::Rect(0, 0, seg_result->width, seg_result->height);
          object_det_->add_image(seg_result);
        } else if (config_.enable_tracking && object_track_) {
          // 跳过检测，直接到跟踪
          seg_result->roi = cv::Rect(0, 0, seg_result->width, seg_result->height);
          seg_result->detection_results.clear();
          object_track_->add_image(seg_result);
        } else if (config_.enable_event_determine && event_determine_) {
          // 跳过检测和跟踪，直接到事件判定
          seg_result->roi = cv::Rect(0, 0, seg_result->width, seg_result->height);
          seg_result->detection_results.clear();
          seg_result->track_results.clear();
          event_determine_->add_image(seg_result);
        } else {
          // 所有后续模块都禁用，直接到最终结果
          seg_result->roi = cv::Rect(0, 0, seg_result->width, seg_result->height);
          seg_result->detection_results.clear();
          seg_result->track_results.clear();
          final_results_.push(seg_result);
        }
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::cout << "seg_to_mask_thread 已退出" << std::endl;
}

// Mask后处理->目标检测的数据流转
void PipelineManager::mask_to_detect_thread_func() {
  // 如果Mask后处理被禁用，此线程直接退出
  if (!config_.enable_mask_postprocess || !mask_postprocess_) {
    std::cout << "mask_to_detect_thread 已跳过（Mask后处理未启用）" << std::endl;
    return;
  }

  while (running_.load()) {
    ImageDataPtr mask_result;
    
    // 等待Mask后处理结果
    if (mask_postprocess_->get_processed_image(mask_result)) {
      if (mask_result) {
        if (config_.enable_detection && object_det_) {
          // 传递给目标检测模块
          object_det_->add_image(mask_result);
        } else if (config_.enable_tracking && object_track_) {
          // 跳过检测，直接到跟踪
          mask_result->detection_results.clear();
          object_track_->add_image(mask_result);
        } else if (config_.enable_event_determine && event_determine_) {
          // 跳过检测和跟踪，直接到事件判定
          mask_result->detection_results.clear();
          mask_result->track_results.clear();
          event_determine_->add_image(mask_result);
        } else {
          // 所有后续模块都禁用，直接到最终结果
          mask_result->detection_results.clear();
          mask_result->track_results.clear();
          final_results_.push(mask_result);
        }
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::cout << "mask_to_detect_thread 已退出" << std::endl;
}

// 目标检测->目标跟踪的数据流转
void PipelineManager::detect_to_track_thread_func() {
  // 如果目标检测被禁用，此线程直接退出
  if (!config_.enable_detection || !object_det_) {
    std::cout << "detect_to_track_thread 已跳过（目标检测未启用）" << std::endl;
    return;
  }

  while (running_.load()) {
    ImageDataPtr detect_result;
    
    // 等待目标检测结果
    if (object_det_->get_processed_image(detect_result)) {
      if (detect_result) {
        if (config_.enable_tracking && object_track_) {
          // 传递给目标跟踪模块
          object_track_->add_image(detect_result);
        } else if (config_.enable_event_determine && event_determine_) {
          // 跳过跟踪，直接到事件判定
          detect_result->track_results = detect_result->detection_results;
          event_determine_->add_image(detect_result);
        } else {
          // 所有后续模块都禁用，直接到最终结果
          detect_result->track_results = detect_result->detection_results;
          final_results_.push(detect_result);
        }
      }
    } else {
      // std::cout << "detect_to_track_thread 等待目标检测结果... 睡眠1秒" << std::endl;
      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::cout << "detect_to_track_thread 已退出" << std::endl;
}

// 目标跟踪->事件判定的数据流转
void PipelineManager::track_to_event_thread_func() {
  // 如果目标跟踪被禁用，此线程直接退出
  if (!config_.enable_tracking || !object_track_) {
    std::cout << "track_to_event_thread 已跳过（目标跟踪未启用）" << std::endl;
    return;
  }

  while (running_.load()) {
    ImageDataPtr track_result;
    
    // 等待目标跟踪结果
    if (object_track_->get_processed_image(track_result)) {
      if (track_result) {
        if (config_.enable_event_determine && event_determine_) {
          // 传递给事件判定模块
          event_determine_->add_image(track_result);
        } else {
          // 事件判定禁用，直接到最终结果
          final_results_.push(track_result);
        }
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::cout << "track_to_event_thread 已退出" << std::endl;
}

// 事件判定->最终结果的数据流转
void PipelineManager::event_to_final_thread_func() {
  // 如果事件判定被禁用，此线程直接退出
  if (!config_.enable_event_determine || !event_determine_) {
    std::cout << "event_to_final_thread 已跳过（事件判定未启用）" << std::endl;
    return;
  }

  while (running_.load()) {
    ImageDataPtr event_result;
    
    // 等待事件判定结果
    if (event_determine_->get_processed_image(event_result)) {
      if (event_result) {
        // 直接添加到最终结果队列
        final_results_.push(event_result);
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::cout << "event_to_final_thread 已退出" << std::endl;
}
