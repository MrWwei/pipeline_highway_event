#include "batch_pipeline_manager.h"
#include "logger_manager.h"
#include <iostream>
#include <iomanip>
#include <future>
#include <queue>

BatchPipelineManager::BatchPipelineManager(const PipelineConfig& config)
    : config_(config), running_(false), stop_requested_(false),
      status_print_interval_(std::chrono::seconds(5)) {
    
    LOG_INFO("初始化批次流水线管理器...");
    
    // 创建批次收集器，设置就绪队列限制为50个批次
    // 这样可以防止语义分割模块处理慢时内存无限增长
    input_buffer_ = std::make_unique<BatchBuffer>(
        std::chrono::milliseconds(10000),  // 100ms超时刷新
        1                               // 最多50个就绪批次，实现背压
    );
    
    // 创建结果连接器
    final_result_connector_ = std::make_unique<BatchConnector>(20); // 允许更多批次排队
    
    // 初始化处理阶段
    if (!initialize_stages()) {
        LOG_ERROR("批次流水线阶段初始化失败");
    }
    
    LOG_INFO("批次流水线管理器初始化完成");
}

BatchPipelineManager::~BatchPipelineManager() {
    stop();
    cleanup_stages();
}

void BatchPipelineManager::start() {
    if (running_.load()) {
        LOG_WARN("批次流水线已经在运行中");
        return;
    }
    
    running_.store(true);
    stop_requested_.store(false);
    start_time_ = std::chrono::high_resolution_clock::now();
    
    LOG_INFO("启动批次流水线...");
    
    // 启动批次收集器
    input_buffer_->start();
    
    // 启动处理阶段
    if (config_.enable_segmentation && semantic_seg_) {
        LOG_INFO("启动语义分割阶段...");
        semantic_seg_->start();
    }
    if (config_.enable_mask_postprocess && mask_postprocess_) {
        mask_postprocess_->start();
    }
    if (config_.enable_detection && object_detection_) {
        object_detection_->start();
    }
    if (config_.enable_tracking && object_tracking_) {
        object_tracking_->start();
    }
    if (config_.enable_event_determine && event_determine_) {
        event_determine_->start();
    }
    
    // 启动连接器
    if (seg_to_mask_connector_) seg_to_mask_connector_->start();
    if (mask_to_detection_connector_) mask_to_detection_connector_->start();
    if (detection_to_tracking_connector_) detection_to_tracking_connector_->start();
    if (tracking_to_event_connector_) tracking_to_event_connector_->start();
    final_result_connector_->start();
    
    // 启动协调线程
    seg_coordinator_thread_ = std::thread(&BatchPipelineManager::seg_coordinator_func, this);
    mask_coordinator_thread_ = std::thread(&BatchPipelineManager::mask_coordinator_func, this);
    detection_coordinator_thread_ = std::thread(&BatchPipelineManager::detection_coordinator_func, this);
    tracking_coordinator_thread_ = std::thread(&BatchPipelineManager::tracking_coordinator_func, this);
    event_coordinator_thread_ = std::thread(&BatchPipelineManager::event_coordinator_func, this);
    result_collector_thread_ = std::thread(&BatchPipelineManager::result_collector_func, this);
    
    // 启动状态监控线程
    status_monitor_thread_ = std::thread(&BatchPipelineManager::status_monitor_func, this);
    
    LOG_INFO("批次流水线启动完成");
}

void BatchPipelineManager::stop() {
    if (!running_.load()) {
        return;
    }
    
    LOG_INFO("正在停止批次流水线...");
    
    stop_requested_.store(true);
    running_.store(false);
    
    // 停止批次收集器
    input_buffer_->stop();
    
    // 停止处理阶段
    if (semantic_seg_) semantic_seg_->stop();
    if (mask_postprocess_) mask_postprocess_->stop();
    if (object_detection_) object_detection_->stop();
    if (object_tracking_) object_tracking_->stop();
    if (event_determine_) event_determine_->stop();
    
    // 停止连接器
    if (seg_to_mask_connector_) seg_to_mask_connector_->stop();
    if (mask_to_detection_connector_) mask_to_detection_connector_->stop();
    if (detection_to_tracking_connector_) detection_to_tracking_connector_->stop();
    if (tracking_to_event_connector_) tracking_to_event_connector_->stop();
    final_result_connector_->stop();
    
    // 通知结果等待线程
    result_queue_cv_.notify_all();
    
    // 等待协调线程结束
    if (seg_coordinator_thread_.joinable()) seg_coordinator_thread_.join();
    if (mask_coordinator_thread_.joinable()) mask_coordinator_thread_.join();
    if (detection_coordinator_thread_.joinable()) detection_coordinator_thread_.join();
    if (tracking_coordinator_thread_.joinable()) tracking_coordinator_thread_.join();
    if (event_coordinator_thread_.joinable()) event_coordinator_thread_.join();
    if (result_collector_thread_.joinable()) result_collector_thread_.join();
    if (status_monitor_thread_.joinable()) status_monitor_thread_.join();
    
    LOG_INFO("批次流水线已停止");
}

bool BatchPipelineManager::add_image(ImageDataPtr image) {
    if (!running_.load() || !image) {
        return false;
    }
    
    total_images_input_.fetch_add(1);
    return input_buffer_->add_image(image);
}

bool BatchPipelineManager::get_result_batch(BatchPtr& batch) {
    return final_result_connector_->receive_batch(batch);
}

bool BatchPipelineManager::get_result_image(ImageDataPtr& image) {
    std::unique_lock<std::mutex> lock(result_queue_mutex_);
    
    result_queue_cv_.wait(lock, [this]() {
        return !result_image_queue_.empty() || !running_.load();
    });
    
    if (!running_.load() && result_image_queue_.empty()) {
        return false;
    }
    
    if (!result_image_queue_.empty()) {
        image = result_image_queue_.front();
        result_image_queue_.pop();
        return true;
    }
    
    return false;
}

void BatchPipelineManager::seg_coordinator_func() {
    LOG_DEBUG("语义分割协调线程已启动");
    
    while (running_.load()) {
        BatchPtr batch;
        
        // 从输入缓冲区获取批次
        if (input_buffer_->get_ready_batch(batch)) {
            if (batch && config_.enable_segmentation && semantic_seg_) {
                // std::cout << "🎨 向语义分割阶段发送批次 " << batch->batch_id << std::endl;
                
                // 发送到语义分割阶段
                if (!semantic_seg_->add_batch(batch)) {
                    LOG_ERROR("无法发送批次到语义分割阶段");
                }
                
                // 获取处理完成的批次
                BatchPtr processed_batch;
                if (semantic_seg_->get_processed_batch(processed_batch)) {
                    if (config_.enable_mask_postprocess && seg_to_mask_connector_) {
                        // 发送到Mask后处理阶段
                        seg_to_mask_connector_->send_batch(processed_batch);
                    } else if (config_.enable_detection && mask_to_detection_connector_) {
                        // 跳过Mask后处理，直接发送到检测阶段
                        mask_to_detection_connector_->send_batch(processed_batch);
                    } else {
                        // 直接发送到结果收集器
                        final_result_connector_->send_batch(processed_batch);
                    }
                }
            } else if (!config_.enable_segmentation) {
                // 语义分割被禁用，直接发送到下一阶段
                if (config_.enable_mask_postprocess && seg_to_mask_connector_) {
                    seg_to_mask_connector_->send_batch(batch);
                } else if (config_.enable_detection && mask_to_detection_connector_) {
                    mask_to_detection_connector_->send_batch(batch);
                } else {
                    std::cout << "实际batch大小是 " << batch->actual_size << std::endl;
                    final_result_connector_->send_batch(batch);
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
    
    LOG_DEBUG("语义分割协调线程已结束");
}

void BatchPipelineManager::mask_coordinator_func() {
    if (!config_.enable_mask_postprocess || !seg_to_mask_connector_ || !mask_postprocess_) {
        return;
    }
    
    LOG_INFO("🔧 Mask后处理协调线程已启动");
    
    while (running_.load()) {
        BatchPtr batch;
        
        // 从语义分割获取批次
        if (seg_to_mask_connector_->receive_batch(batch)) {
            if (batch) {
                // std::cout << "🔧 向Mask后处理阶段发送批次 " << batch->batch_id << std::endl;
                
                // 发送到Mask后处理阶段
                if (!mask_postprocess_->add_batch(batch)) {
                    LOG_ERROR("无法发送批次到Mask后处理阶段");
                }
                
                // 获取处理完成的批次
                BatchPtr processed_batch;
                if (mask_postprocess_->get_processed_batch(processed_batch)) {
                    if (config_.enable_detection && mask_to_detection_connector_) {
                        // 发送到检测阶段
                        mask_to_detection_connector_->send_batch(processed_batch);
                    } else {
                        // 直接发送到结果收集器
                        final_result_connector_->send_batch(processed_batch);
                    }
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
    
    LOG_INFO("🔧 Mask后处理协调线程已结束");
}

void BatchPipelineManager::detection_coordinator_func() {
    if (!config_.enable_detection || !mask_to_detection_connector_ || !object_detection_) {
        return;
    }
    
    LOG_INFO("🎯 目标检测协调线程已启动");
    
    while (running_.load()) {
        BatchPtr batch;
        
        // 从Mask后处理获取批次
        if (mask_to_detection_connector_->receive_batch(batch)) {
            if (batch) {
                // std::cout << "🎯 向目标检测阶段发送批次 " << batch->batch_id << std::endl;
                
                // 发送到目标检测阶段
                if (!object_detection_->add_batch(batch)) {
                    LOG_ERROR("无法发送批次到目标检测阶段");
                }
                
                // 获取处理完成的批次
                BatchPtr processed_batch;
                if (object_detection_->get_processed_batch(processed_batch)) {
                    if (config_.enable_tracking && detection_to_tracking_connector_) {
                        // 发送到跟踪阶段
                        detection_to_tracking_connector_->send_batch(processed_batch);
                    } else {
                        // 直接发送到结果收集器
                        final_result_connector_->send_batch(processed_batch);
                    }
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
    
    LOG_INFO("🎯 目标检测协调线程已结束");
}

void BatchPipelineManager::tracking_coordinator_func() {
    if (!config_.enable_tracking || !detection_to_tracking_connector_ || !object_tracking_) {
        return;
    }
    
    LOG_INFO("🎯 目标跟踪协调线程已启动");
    
    while (running_.load()) {
        BatchPtr batch;
        
        // 从目标检测获取批次
        if (detection_to_tracking_connector_->receive_batch(batch)) {
            if (batch) {
                // std::cout << "🎯 向目标跟踪阶段发送批次 " << batch->batch_id << std::endl;
                
                // 发送到目标跟踪阶段
                if (!object_tracking_->add_batch(batch)) {
                    LOG_ERROR("无法发送批次到目标跟踪阶段");
                }
                
                // 获取处理完成的批次
                BatchPtr processed_batch;
                if (object_tracking_->get_processed_batch(processed_batch)) {
                    if (config_.enable_event_determine && tracking_to_event_connector_) {
                        // 发送到事件判定阶段
                        tracking_to_event_connector_->send_batch(processed_batch);
                    } else {
                        // 直接发送到结果收集器
                        final_result_connector_->send_batch(processed_batch);
                    }
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
    
    LOG_INFO("🎯 目标跟踪协调线程已结束");
}

void BatchPipelineManager::event_coordinator_func() {
    if (!config_.enable_event_determine || !tracking_to_event_connector_ || !event_determine_) {
        return;
    }
    
    LOG_INFO("🎯 事件判定协调线程已启动");
    
    while (running_.load()) {
        BatchPtr batch;
        
        // 从目标跟踪获取批次
        if (tracking_to_event_connector_->receive_batch(batch)) {
            if (batch) {
                // std::cout << "🎯 向事件判定阶段发送批次 " << batch->batch_id << std::endl;
                
                // 发送到事件判定阶段
                if (!event_determine_->add_batch(batch)) {
                    LOG_ERROR("无法发送批次到事件判定阶段");
                }
                
                // 获取处理完成的批次
                BatchPtr processed_batch;
                if (event_determine_->get_processed_batch(processed_batch)) {
                    // 发送到结果收集器
                    final_result_connector_->send_batch(processed_batch);
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
    
    LOG_INFO("🎯 事件判定协调线程已结束");
}

void BatchPipelineManager::result_collector_func() {
    LOG_INFO("📦 结果收集线程已启动");
    
    while (running_.load()) {
        BatchPtr batch;
        
        // 从最终结果连接器获取批次
        if (final_result_connector_->receive_batch(batch)) {
            if (batch) {
                // std::cout << "📦 收集批次 " << batch->batch_id << " 的处理结果" << std::endl;
                
                // 将批次分解为单个图像并加入结果队列
                decompose_batch_to_images(batch);
                
                total_batches_processed_.fetch_add(1);
                total_images_output_.fetch_add(batch->actual_size);
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
    
    LOG_INFO("📦 结果收集线程已结束");
}

void BatchPipelineManager::decompose_batch_to_images(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(result_queue_mutex_);
    
    for (size_t i = 0; i < batch->actual_size; ++i) {
        result_image_queue_.push(batch->images[i]);
    }
    
    // 通知等待结果的线程
    result_queue_cv_.notify_all();
}

void BatchPipelineManager::status_monitor_func() {
    while (running_.load()) {
        std::this_thread::sleep_for(status_print_interval_);
        
        if (!running_.load()) {
            break;
        }
        
        print_status();
    }
}

bool BatchPipelineManager::initialize_stages() {
    LOG_INFO("🏗️ 初始化批次处理阶段...");
    
    // 创建连接器
    int connector_capacity = 10; // 每个连接器的容量
    seg_to_mask_connector_ = std::make_unique<BatchConnector>(connector_capacity);
    mask_to_detection_connector_ = std::make_unique<BatchConnector>(connector_capacity);
    detection_to_tracking_connector_ = std::make_unique<BatchConnector>(connector_capacity);
    tracking_to_event_connector_ = std::make_unique<BatchConnector>(connector_capacity);
    
    // 初始化语义分割阶段
    if (config_.enable_segmentation) {
        semantic_seg_ = std::make_unique<BatchSemanticSegmentation>(config_.semantic_threads, &config_);
        LOG_INFO("✅ 批次语义分割阶段初始化完成");
    }
    
    // 初始化Mask后处理阶段
    if (config_.enable_mask_postprocess) {
        mask_postprocess_ = std::make_unique<BatchMaskPostProcess>(config_.mask_postprocess_threads);
        LOG_INFO("✅ 批次Mask后处理阶段初始化完成");
    }
    
    // 初始化目标检测阶段
    if (config_.enable_detection) {
        object_detection_ = std::make_unique<BatchObjectDetection>(config_.detection_threads, &config_);
        LOG_INFO("✅ 批次目标检测阶段初始化完成");
    }
    
    // 初始化目标跟踪阶段
    if (config_.enable_tracking) {
        object_tracking_ = std::make_unique<BatchObjectTracking>(config_.tracking_threads, &config_);
        LOG_INFO("✅ 批次目标跟踪阶段初始化完成");
    }
    
    // 初始化事件判定阶段
    if (config_.enable_event_determine) {
        event_determine_ = std::make_unique<BatchEventDetermine>(config_.event_determine_threads, &config_);
        LOG_INFO("✅ 批次事件判定阶段初始化完成");
    }
    
    return true;
}

void BatchPipelineManager::cleanup_stages() {
    semantic_seg_.reset();
    mask_postprocess_.reset();
    object_detection_.reset();
    object_tracking_.reset();
    event_determine_.reset();
    
    seg_to_mask_connector_.reset();
    mask_to_detection_connector_.reset();
    detection_to_tracking_connector_.reset();
    tracking_to_event_connector_.reset();
    final_result_connector_.reset();
}

void BatchPipelineManager::print_status() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    
    std::ostringstream status_stream;
    status_stream << "\n" << std::string(80, '=') << "\n";
    status_stream << "📊 批次流水线状态报告 (运行时间: " << runtime.count() << "s)\n";
    status_stream << std::string(80, '=') << "\n";
    
    // 基本统计
    auto stats = get_statistics();
    status_stream << std::fixed << std::setprecision(2);
    status_stream << "📈 总体统计:\n";
    status_stream << "  输入图像数: " << stats.total_images_input << "\n";
    status_stream << "  处理批次数: " << stats.total_batches_processed << "\n";
    status_stream << "  输出图像数: " << stats.total_images_output << "\n";
    status_stream << "  吞吐量: " << stats.throughput_images_per_second << " 图像/秒\n";
    status_stream << "  平均批次处理时间: " << stats.average_batch_processing_time_ms << " ms\n";
    
    // 队列状态
    status_stream << "\n📋 队列状态:\n";
    
    // 输入缓冲区状态，包含背压信息
    bool is_backpressure = input_buffer_->is_ready_queue_full();
    status_stream << "  输入缓冲区: " << input_buffer_->get_current_collecting_size() << "/32 (收集中), " 
              << input_buffer_->get_ready_batch_count() << "/" << input_buffer_->get_max_ready_batches() 
              << " 批次就绪";
    if (is_backpressure) {
        status_stream << " ⚠️ 背压激活";
    }
    status_stream << "\n";
    
    if (semantic_seg_) {
        status_stream << "  语义分割: " << semantic_seg_->get_queue_size() << " 批次等待\n";
    }
    if (mask_postprocess_) {
        status_stream << "  Mask后处理: " << mask_postprocess_->get_queue_size() << " 批次等待\n";
    }
    if (object_detection_) {
        status_stream << "  目标检测: " << object_detection_->get_queue_size() << " 批次等待\n";
    }
    if (object_tracking_) {
        status_stream << "  目标跟踪: " << object_tracking_->get_queue_size() << " 批次等待\n";
    }
    if (event_determine_) {
        status_stream << "  事件判定: " << event_determine_->get_queue_size() << " 批次等待\n";
    }
    
    status_stream << "  结果队列: " << stats.current_output_buffer_size << " 图像等待输出\n";
    
    // 性能指标
    status_stream << "\n⚡ 各阶段性能:\n";
    if (semantic_seg_) {
        status_stream << "  " << semantic_seg_->get_stage_name() << ": "
                  << semantic_seg_->get_processed_count() << " 批次, 平均 "
                  << semantic_seg_->get_average_processing_time() << " ms/批次\n";
    }
    if (mask_postprocess_) {
        status_stream << "  " << mask_postprocess_->get_stage_name() << ": "
                  << mask_postprocess_->get_processed_count() << " 批次, 平均 "
                  << mask_postprocess_->get_average_processing_time() << " ms/批次\n";
    }
    if (object_detection_) {
        status_stream << "  " << object_detection_->get_stage_name() << ": "
                  << object_detection_->get_processed_count() << " 批次, 平均 "
                  << object_detection_->get_average_processing_time() << " ms/批次\n";
    }
    if (object_tracking_) {
        status_stream << "  " << object_tracking_->get_stage_name() << ": "
                  << object_tracking_->get_processed_count() << " 批次, 平均 "
                  << object_tracking_->get_average_processing_time() << " ms/批次\n";
    }
    if (event_determine_) {
        status_stream << "  " << event_determine_->get_stage_name() << ": "
                  << event_determine_->get_processed_count() << " 批次, 平均 "
                  << event_determine_->get_average_processing_time() << " ms/批次\n";
    }
    
    status_stream << std::string(80, '=') << "\n\n";
    
    // 使用日志输出整个状态报告
    LOG_INFO(status_stream.str());
}

BatchPipelineManager::Statistics BatchPipelineManager::get_statistics() const {
    Statistics stats;
    
    stats.total_images_input = total_images_input_.load();
    stats.total_batches_processed = total_batches_processed_.load();
    stats.total_images_output = total_images_output_.load();
    
    // 计算吞吐量
    auto now = std::chrono::high_resolution_clock::now();
    auto runtime_seconds = std::chrono::duration<double>(now - start_time_).count();
    if (runtime_seconds > 0) {
        stats.throughput_images_per_second = (double)stats.total_images_output / runtime_seconds;
    } else {
        stats.throughput_images_per_second = 0.0;
    }
    
    // 计算平均处理时间
    stats.average_batch_processing_time_ms = 0.0;
    if (semantic_seg_) {
        stats.average_batch_processing_time_ms += semantic_seg_->get_average_processing_time();
    }
    if (mask_postprocess_) {
        stats.average_batch_processing_time_ms += mask_postprocess_->get_average_processing_time();
    }
    if (object_detection_) {
        stats.average_batch_processing_time_ms += object_detection_->get_average_processing_time();
    }
    if (object_tracking_) {
        stats.average_batch_processing_time_ms += object_tracking_->get_average_processing_time();
    }
    if (event_determine_) {
        stats.average_batch_processing_time_ms += event_determine_->get_average_processing_time();
    }
    
    // 当前队列大小
    stats.current_input_buffer_size = input_buffer_->get_ready_batch_count();
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(result_queue_mutex_));
        stats.current_output_buffer_size = result_image_queue_.size();
    }
    
    return stats;
}
