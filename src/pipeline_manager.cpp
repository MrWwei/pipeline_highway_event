#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <map>
#include <thread>
#include <future>

PipelineManager::PipelineManager(const PipelineConfig& config)
    : running_(false), next_frame_idx_(0), final_results_(config.final_result_queue_capacity), config_(config) {
  semantic_seg_ = std::make_unique<SemanticSegmentation>(config.semantic_threads, &config);
  
  // 根据开关决定是否创建模块
  if (config.enable_mask_postprocess) {
    mask_postprocess_ =
        std::make_unique<MaskPostProcess>(config.mask_postprocess_threads);
  }
  
  if (config.enable_detection) {
    object_det_ = std::make_unique<ObjectDetection>(config.detection_threads, &config);
  }
  
  if (config.enable_tracking) {
    object_track_ = std::make_unique<ObjectTracking>(config.tracking_threads);
  }
  
  if (config.enable_box_filter) {
    box_filter_ = std::make_unique<BoxFilter>(config.box_filter_threads, &config);
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
  next_frame_idx_ = 0;
  {
    std::lock_guard<std::mutex> lock(pending_results_mutex_);
    pending_results_.clear();
  }

  // 启动各个处理模块（根据配置）
  semantic_seg_->start();

  std::cout << "🔄 语义分割模块已启动，线程数: " << config_.semantic_threads << std::endl;
  
  if (config_.enable_mask_postprocess && mask_postprocess_) {
    mask_postprocess_->start();
    std::cout << "🔍 Mask后处理模块已启用" << std::endl;
  } else {
    std::cout << "⚠️ Mask后处理模块已禁用" << std::endl;
  }
  
  if (config_.enable_detection && object_det_) {
    object_det_->start();
    std::cout << "🔍 目标检测模块已启用" << std::endl;
  } else {
    std::cout << "⚠️ 目标检测模块已禁用" << std::endl;
  }
  
  if (config_.enable_tracking && object_track_) {
    object_track_->start();
    std::cout << "🎯 目标跟踪模块已启用" << std::endl;
  } else {
    std::cout << "⚠️ 目标跟踪模块已禁用" << std::endl;
  }
  
  if (config_.enable_box_filter && box_filter_) {
    box_filter_->start();
    std::cout << "📋 目标框筛选模块已启用" << std::endl;
  } else {
    std::cout << "⚠️ 目标框筛选模块已禁用" << std::endl;
  }

  // 启动各阶段的协调线程
  seg_to_mask_thread_ =
      std::thread(&PipelineManager::seg_to_mask_thread_func, this);
  mask_to_detect_thread_ =
      std::thread(&PipelineManager::mask_to_detect_thread_func, this);
  track_to_filter_thread_ =
      std::thread(&PipelineManager::track_to_filter_thread_func, this);
  filter_to_final_thread_ =
      std::thread(&PipelineManager::filter_to_final_thread_func, this);
}

void PipelineManager::stop() {
  if (!running_.load()) {
    return;
  }

  std::cout << "开始停止流水线..." << std::endl;
  running_.store(false);

  // 停止各个处理模块
  std::cout << "停止语义分割模块..." << std::endl;
  semantic_seg_->stop();
  
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
  
  if (box_filter_) {
    std::cout << "停止目标框筛选模块..." << std::endl;
    box_filter_->stop();
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
  
  join_with_timeout(seg_to_mask_thread_, "seg_to_mask");
  join_with_timeout(mask_to_detect_thread_, "mask_to_detect");
  join_with_timeout(track_to_filter_thread_, "track_to_filter");
  join_with_timeout(filter_to_final_thread_, "filter_to_final");

  // 清理流水线管理器自己的队列和资源
  std::cout << "清理流水线队列和缓存..." << std::endl;
  final_results_.shutdown(); // 关闭结果队列，唤醒所有等待的线程
  final_results_.clear();
  {
    std::lock_guard<std::mutex> lock(pending_results_mutex_);
    pending_results_.clear();
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
  return final_results_.wait_and_pop(result);
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
  if (config_.enable_mask_postprocess && mask_postprocess_) {
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
    std::cout << "\n📊 Mask后处理阶段 [已禁用]" << std::endl;
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
  if (config_.enable_tracking && object_track_) {
    std::cout << "\n🎯 目标跟踪阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(object_track_->get_queue_size() > 0 ? "🟢" : "⚪")
              << "] " << object_track_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(object_track_->get_output_queue_size() > 0 ? "🟢"
                                                                        : "⚪")
              << "] " << object_track_->get_output_queue_size() << std::endl;
  } else {
    std::cout << "\n🎯 目标跟踪阶段 [已禁用]" << std::endl;
  }

  // 目标框筛选阶段
  if (config_.enable_box_filter && box_filter_) {
    std::cout << "\n📦 目标框筛选阶段 [启用]" << std::endl;
    std::cout << "   输入队列: ["
              << std::string(box_filter_->get_queue_size() > 0 ? "🟢" : "⚪")
              << "] " << box_filter_->get_queue_size() << std::endl;
    std::cout << "   输出队列: ["
              << std::string(box_filter_->get_output_queue_size() > 0 ? "🟢"
                                                                      : "⚪")
              << "] " << box_filter_->get_output_queue_size() << std::endl;
  } else {
    std::cout << "\n📦 目标框筛选阶段 [已禁用]" << std::endl;
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
  std::cout << "   语义分割线程数: " << semantic_seg_->get_thread_count()
            << std::endl;
  
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
  
  if (box_filter_) {
    std::cout << "   目标框筛选线程数: " << box_filter_->get_thread_count()
              << std::endl;
  } else {
    std::cout << "   目标框筛选线程数: 0 (已禁用)" << std::endl;
  }
  
  std::cout << "   协调器线程数: 4" << std::endl;
  
  int total_threads = semantic_seg_->get_thread_count() + 4;
  if (mask_postprocess_) total_threads += mask_postprocess_->get_thread_count();
  if (object_det_) total_threads += object_det_->get_thread_count();
  if (object_track_) total_threads += object_track_->get_thread_count();
  if (box_filter_) total_threads += box_filter_->get_thread_count();
  
  std::cout << "   总工作线程数: " << total_threads << std::endl;
}

// 语义分割->Mask后处理的数据流转
void PipelineManager::seg_to_mask_thread_func() {
  auto last_status_time = std::chrono::steady_clock::now();

  while (running_.load()) {
    bool has_work = false;
    size_t processed = 0;

    // 检查输出队列
    if (semantic_seg_->get_output_queue_size() > 0) {
      ImageDataPtr seg_result;

      // 批量处理数据
      while (semantic_seg_->get_processed_image(seg_result) && running_.load()) {
        if (seg_result) {
          has_work = true;
          processed++;
          
          // 根据配置决定流转路径
          if (config_.enable_mask_postprocess && mask_postprocess_) {
            // 启用Mask后处理：传递给Mask后处理模块
            mask_postprocess_->add_image(seg_result);
          } else {
            // 跳过Mask后处理：直接传递给下一阶段
            // 模拟Mask后处理完成（创建默认ROI）
            seg_result->roi = cv::Rect(0, 0, seg_result->width, seg_result->height);
            if (seg_result->mask_postprocess_promise && 
                seg_result->mask_postprocess_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
              seg_result->mask_postprocess_promise->set_value();
            }
            
            // 将结果直接添加到mask_to_detect线程要处理的队列中
            // 这里我们需要手动调用下一阶段的逻辑
            if (config_.enable_detection && object_det_) {
              object_det_->add_image(seg_result);
            } else if (config_.enable_tracking && object_track_) {
              // 跳过检测，直接到跟踪
              seg_result->detection_results.clear();
              if (seg_result->detection_promise && 
                  seg_result->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                seg_result->detection_promise->set_value();
              }
              object_track_->add_image(seg_result);
            } else if (config_.enable_box_filter && box_filter_) {
              // 跳过检测和跟踪，直接到筛选
              seg_result->detection_results.clear();
              seg_result->track_results = seg_result->detection_results;
              if (seg_result->detection_promise && 
                  seg_result->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                seg_result->detection_promise->set_value();
              }
              if (seg_result->tracking_promise &&
                  seg_result->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                seg_result->tracking_promise->set_value();
              }
              box_filter_->add_image(seg_result);
            } else {
              // 所有后续模块都禁用，直接到最终结果
              seg_result->detection_results.clear();
              seg_result->track_results = seg_result->detection_results;
              if (seg_result->detection_promise && 
                  seg_result->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                seg_result->detection_promise->set_value();
              }
              if (seg_result->tracking_promise &&
                  seg_result->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                seg_result->tracking_promise->set_value();
              }
              if (seg_result->box_filter_promise &&
                  seg_result->box_filter_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                seg_result->box_filter_promise->set_value();
              }
              final_results_.push(seg_result);
            }
          }
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  std::cout << "seg_to_mask_thread 已退出" << std::endl;
}

// Mask后处理->目标检测->目标跟踪的数据流转
void PipelineManager::mask_to_detect_thread_func() {
  std::vector<ImageDataPtr> pending_images; // 存储等待检测完成的图像
  uint64_t next_expected_detection_frame = 0; // 下一个期望传递给跟踪的帧序号

  while (running_.load()) {
    bool has_work = false;

    // 根据配置决定数据来源
    if (config_.enable_mask_postprocess && mask_postprocess_) {
      // 从mask后处理获取新的图像
      if (mask_postprocess_->get_output_queue_size() > 0) {
        ImageDataPtr mask_result;
        while (mask_postprocess_->get_processed_image(mask_result) && running_.load()) {
          if (mask_result) {
            has_work = true;
            
            // 根据配置决定流转路径
            if (config_.enable_detection && object_det_) {
              // 启用检测模块：传递给目标检测
              object_det_->add_image(mask_result);
              pending_images.push_back(mask_result); // 添加到待处理列表
            } else {
              // 跳过检测模块：直接传递给跟踪或筛选
              // 模拟检测完成（空结果）
              mask_result->detection_results.clear();
              if (mask_result->detection_promise && 
                  mask_result->detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                mask_result->detection_promise->set_value();
              }
              
              if (config_.enable_tracking && object_track_) {
                object_track_->add_image(mask_result);
              } else if (config_.enable_box_filter && box_filter_) {
                // 跳过跟踪，直接到筛选
                mask_result->track_results = mask_result->detection_results;
                if (mask_result->tracking_promise &&
                    mask_result->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  mask_result->tracking_promise->set_value();
                }
                box_filter_->add_image(mask_result);
              } else {
                // 所有后续模块都禁用，直接到最终结果
                mask_result->track_results = mask_result->detection_results;
                if (mask_result->tracking_promise &&
                    mask_result->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  mask_result->tracking_promise->set_value();
                }
                if (mask_result->box_filter_promise &&
                    mask_result->box_filter_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  mask_result->box_filter_promise->set_value();
                }
                final_results_.push(mask_result);
              }
            }
          }
        }
      }
    }
    // 注意：如果Mask后处理被禁用，数据会在seg_to_mask_thread中直接处理

    // 如果启用了检测模块，按顺序检查已完成的检测任务
    if (config_.enable_detection && object_det_) {
      auto it = pending_images.begin();
      while (it != pending_images.end()) {
        auto& image = *it;
        
        // 只处理下一个期望的帧序号
        if (image->frame_idx == next_expected_detection_frame) {
          // 检查检测是否完成（非阻塞检查）
          if (image->detection_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
              image->detection_future.get(); // 确保没有异常
              
              if (config_.enable_tracking && object_track_) {
                object_track_->add_image(image);
              } else if (config_.enable_box_filter && box_filter_) {
                // 跳过跟踪，直接到筛选
                image->track_results = image->detection_results;
                if (image->tracking_promise &&
                    image->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  image->tracking_promise->set_value();
                }
                box_filter_->add_image(image);
              } else {
                // 跟踪和筛选都禁用，直接到最终结果
                image->track_results = image->detection_results;
                if (image->tracking_promise &&
                    image->tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  image->tracking_promise->set_value();
                }
                if (image->box_filter_promise &&
                    image->box_filter_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  image->box_filter_promise->set_value();
                }
                final_results_.push(image);
              }
              
              it = pending_images.erase(it); // 从待处理列表中移除
              next_expected_detection_frame++; // 更新期望的下一帧
              has_work = true;
            } catch (const std::exception& e) {
              std::cerr << "❌ 目标检测失败，帧 " << image->frame_idx << ": " << e.what() << std::endl;
              it = pending_images.erase(it); // 即使失败也要移除
              next_expected_detection_frame++; // 跳过失败的帧
            }
          } else {
            break; // 当前期望的帧还未完成，等待
          }
        } else {
          // 不是期望的帧序号，继续检查下一个
          ++it;
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

// 目标跟踪->目标框筛选的数据流转
void PipelineManager::track_to_filter_thread_func() {
  while (running_.load()) {
    bool has_work = false;

    // 根据配置决定数据来源
    if (config_.enable_tracking && object_track_) {
      // 从目标跟踪获取新的图像
      if (object_track_->get_output_queue_size() > 0) {
        ImageDataPtr track_result;
        while (object_track_->get_processed_image(track_result) && running_.load()) {
          if (track_result) {
            has_work = true;
            
            if (config_.enable_box_filter && box_filter_) {
              // 传递给目标框筛选
              box_filter_->add_image(track_result);
            } else {
              // 跳过筛选，直接到最终结果
              if (track_result->box_filter_promise &&
                  track_result->box_filter_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                track_result->box_filter_promise->set_value();
              }
              final_results_.push(track_result);
            }
          }
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  std::cout << "track_to_filter_thread 已退出" << std::endl;
}

// 目标框筛选->最终结果的数据流转
void PipelineManager::filter_to_final_thread_func() {
  uint64_t cleanup_counter = 0; // 清理计数器
  
  while (running_.load()) {
    bool has_work = false;
    size_t processed = 0;

    // 检查box_filter_是否启用且存在
    if (config_.enable_box_filter && box_filter_ && box_filter_->get_output_queue_size() > 0) {
      ImageDataPtr filter_result;

      // 批量处理数据
      while (box_filter_->get_processed_image(filter_result) && running_.load()) {
        if (filter_result) {
          has_work = true;
          processed++;

          // 使用互斥锁保护对pending_results_的访问
          std::lock_guard<std::mutex> lock(pending_results_mutex_);

          // 将结果添加到pending_results_中
          pending_results_[filter_result->frame_idx] = filter_result;

          // 检查是否有可以按序输出的结果
          while (pending_results_.find(next_frame_idx_) !=
                 pending_results_.end()) {
            auto next_result = pending_results_[next_frame_idx_];
            final_results_.push(next_result);
            pending_results_.erase(next_frame_idx_);
            next_frame_idx_++;
          }
          
          // 定期清理pending_results_中的旧结果，防止内存积累
          cleanup_counter++;
          if (cleanup_counter % 100 == 0) {
            // 清理超出预期范围太远的结果
            auto it = pending_results_.begin();
            while (it != pending_results_.end()) {
              if (it->first < next_frame_idx_ - 50) { // 保留最近50个乱序结果
                it = pending_results_.erase(it);
              } else {
                ++it;
              }
            }
          }
        }
      }
    }

    if (!has_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  std::cout << "filter_to_final_thread 已退出" << std::endl;
}
