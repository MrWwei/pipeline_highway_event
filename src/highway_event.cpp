#include "highway_event.h"
#include <chrono>
#include <iostream>

HighwayEventDetector::HighwayEventDetector() {
    // 构造函数中不做实际初始化，等待用户调用initialize
}

HighwayEventDetector::~HighwayEventDetector() {
    stop();
}

bool HighwayEventDetector::initialize(const HighwayEventConfig& config) {
    if (is_initialized_.load()) {
        std::cerr << "❌ HighwayEventDetector 已经初始化过了" << std::endl;
        return false;
    }
    
    try {
        config_ = config;
        
        // 创建流水线配置
        PipelineConfig pipeline_config;
        pipeline_config.semantic_threads = config_.semantic_threads;
        pipeline_config.mask_postprocess_threads = config_.mask_threads;
        pipeline_config.detection_threads = config_.detection_threads;
        pipeline_config.tracking_threads = config_.tracking_threads;
        pipeline_config.box_filter_threads = config_.filter_threads;
        pipeline_config.seg_model_path = config_.seg_model_path;
        pipeline_config.seg_enable_show = config_.seg_enable_show;
        pipeline_config.seg_show_image_path = config_.seg_show_image_path;
        pipeline_config.det_algor_name = config_.det_algor_name;
        pipeline_config.det_model_path = config_.det_model_path;
        pipeline_config.det_img_size = config_.det_img_size;
        pipeline_config.det_conf_thresh = config_.det_conf_thresh;
        pipeline_config.det_iou_thresh = config_.det_iou_thresh;
        pipeline_config.det_max_batch_size = config_.det_max_batch_size;
        pipeline_config.det_min_opt = config_.det_min_opt;
        pipeline_config.det_mid_opt = config_.det_mid_opt;
        pipeline_config.det_max_opt = config_.det_max_opt;
        pipeline_config.det_is_ultralytics = config_.det_is_ultralytics;
        pipeline_config.det_gpu_id = config_.det_gpu_id;
        pipeline_config.box_filter_top_fraction = config_.box_filter_top_fraction;
        pipeline_config.box_filter_bottom_fraction = config_.box_filter_bottom_fraction;
        pipeline_config.final_result_queue_capacity = config_.result_queue_capacity;
        
        // 创建流水线管理器
        pipeline_manager_ = std::make_unique<PipelineManager>(pipeline_config);
        
        is_initialized_.store(true);
        
        if (config_.enable_debug_log) {
            std::cout << "✅ HighwayEventDetector 初始化成功" << std::endl;
            std::cout << "   语义分割线程: " << config_.semantic_threads << std::endl;
            std::cout << "   Mask后处理线程: " << config_.mask_threads << std::endl;
            std::cout << "   目标检测线程: " << config_.detection_threads << std::endl;
            std::cout << "   目标跟踪线程: " << config_.tracking_threads << std::endl;
            std::cout << "   目标框筛选线程: " << config_.filter_threads << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ HighwayEventDetector 初始化失败: " << e.what() << std::endl;
        return false;
    }
}

bool HighwayEventDetector::start() {
    if (!is_initialized_.load()) {
        std::cerr << "❌ 请先调用 initialize() 进行初始化" << std::endl;
        return false;
    }
    
    if (is_running_.load()) {
        if (config_.enable_debug_log) {
            std::cout << "⚠️ HighwayEventDetector 已经在运行中" << std::endl;
        }
        return true;
    }
    
    try {
        // 启动流水线管理器
        pipeline_manager_->start();
        
        // 启动内部结果处理线程
        result_thread_running_.store(true);
        result_thread_ = std::thread(&HighwayEventDetector::result_processing_thread, this);
        
        is_running_.store(true);
        
        if (config_.enable_debug_log) {
            std::cout << "✅ HighwayEventDetector 启动成功" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ HighwayEventDetector 启动失败: " << e.what() << std::endl;
        return false;
    }
}

int64_t HighwayEventDetector::add_frame(const cv::Mat& image_mat) {
    return add_frame_with_timeout(image_mat, config_.add_timeout_ms);
}

int64_t HighwayEventDetector::add_frame(cv::Mat&& image_mat) {
    if (!is_running_.load()) {
        std::cerr << "❌ 流水线未运行，请先调用 start()" << std::endl;
        return -1;
    }
    
    try {
        // 分配帧ID
        uint64_t frame_id = next_frame_id_.fetch_add(1);
        
        // 创建图像数据
        cv::Mat* frame_ptr = new cv::Mat(std::move(image_mat));
        ImageDataPtr img_data = std::make_shared<ImageData>(frame_ptr);
        img_data->frame_idx = frame_id;
        
        // 添加到流水线
        pipeline_manager_->add_image(img_data);
        
        if (config_.enable_debug_log) {
            std::cout << "📥 添加帧 " << frame_id << " 到流水线" << std::endl;
        }
        
        return static_cast<int64_t>(frame_id);
    } catch (const std::exception& e) {
        std::cerr << "❌ 添加帧失败: " << e.what() << std::endl;
        return -1;
    }
}

int64_t HighwayEventDetector::add_frame_with_timeout(const cv::Mat& image_mat, int timeout_ms) {
    if (!is_running_.load()) {
        std::cerr << "❌ 流水线未运行，请先调用 start()" << std::endl;
        return -1;
    }
    
    if (image_mat.empty()) {
        std::cerr << "❌ 输入图像为空" << std::endl;
        return -1;
    }
    
    try {
        // 分配帧ID
        uint64_t frame_id = next_frame_id_.fetch_add(1);
        
        // 创建图像数据（拷贝）
        cv::Mat* frame_ptr = new cv::Mat(image_mat.clone());
        ImageDataPtr img_data = std::make_shared<ImageData>(frame_ptr);
        img_data->frame_idx = frame_id;
        
        // TODO: 这里可以添加超时机制，检查第一阶段队列是否满
        // 当前直接添加到流水线，依赖PipelineManager内部的队列管理
        pipeline_manager_->add_image(img_data);
        
        if (config_.enable_debug_log) {
            std::cout << "📥 添加帧 " << frame_id << " 到流水线" << std::endl;
        }
        
        return static_cast<int64_t>(frame_id);
    } catch (const std::exception& e) {
        std::cerr << "❌ 添加帧失败: " << e.what() << std::endl;
        return -1;
    }
}

GetResultReturn HighwayEventDetector::get_result(uint64_t frame_id) {
    return get_result_with_timeout(frame_id, config_.get_result_timeout_ms);
}

GetResultReturn HighwayEventDetector::get_result_with_timeout(uint64_t frame_id, int timeout_ms) {
    if (!is_running_.load()) {
        return GetResultReturn(ResultStatus::PIPELINE_STOPPED);
    }
    
    std::unique_lock<std::mutex> lock(result_mutex_);
    
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(timeout_ms);
    
    while (is_running_.load()) {
        // 检查是否已有结果
        auto it = completed_results_.find(frame_id);
        if (it != completed_results_.end()) {
            ImageDataPtr result = it->second;
            completed_results_.erase(it);
            
            if (config_.enable_debug_log) {
                std::cout << "📤 获取帧 " << frame_id << " 的处理结果" << std::endl;
            }
            
            return GetResultReturn(ResultStatus::SUCCESS, result);
        }
        
        // 检查超时
        if (timeout_ms >= 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed >= timeout) {
                if (config_.enable_debug_log) {
                    std::cout << "⏰ 获取帧 " << frame_id << " 结果超时" << std::endl;
                }
                return GetResultReturn(ResultStatus::TIMEOUT);
            }
            
            // 等待一定时间
            auto remaining = timeout - elapsed;
            if (result_cv_.wait_for(lock, remaining) == std::cv_status::timeout) {
                return GetResultReturn(ResultStatus::TIMEOUT);
            }
        } else {
            // 无限等待
            result_cv_.wait(lock);
        }
    }
    
    return GetResultReturn(ResultStatus::PIPELINE_STOPPED);
}

GetResultReturn HighwayEventDetector::try_get_result(uint64_t frame_id) {
    if (!is_running_.load()) {
        return GetResultReturn(ResultStatus::PIPELINE_STOPPED);
    }
    
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    auto it = completed_results_.find(frame_id);
    if (it != completed_results_.end()) {
        ImageDataPtr result = it->second;
        completed_results_.erase(it);
        
        if (config_.enable_debug_log) {
            std::cout << "📤 获取帧 " << frame_id << " 的处理结果（非阻塞）" << std::endl;
        }
        
        return GetResultReturn(ResultStatus::SUCCESS, result);
    }
    
    return GetResultReturn(ResultStatus::NOT_FOUND);
}

bool HighwayEventDetector::stop() {
    if (!is_running_.load()) {
        return true;
    }
    
    if (config_.enable_debug_log) {
        std::cout << "🛑 正在停止 HighwayEventDetector..." << std::endl;
    }
    
    // 停止流水线
    is_running_.store(false);
    
    if (pipeline_manager_) {
        pipeline_manager_->stop();
    }
    
    // 停止结果处理线程
    result_thread_running_.store(false);
    result_cv_.notify_all();
    
    if (result_thread_.joinable()) {
        result_thread_.join();
    }
    
    // 清理结果
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        completed_results_.clear();
    }
    
    if (config_.enable_debug_log) {
        std::cout << "✅ HighwayEventDetector 已停止" << std::endl;
    }
    
    return true;
}

void HighwayEventDetector::print_status() const {
    if (!is_initialized_.load()) {
        std::cout << "❌ HighwayEventDetector 未初始化" << std::endl;
        return;
    }
    
    std::cout << "\n🔍 HighwayEventDetector 状态:" << std::endl;
    std::cout << "   初始化状态: " << (is_initialized_.load() ? "✅" : "❌") << std::endl;
    std::cout << "   运行状态: " << (is_running_.load() ? "🟢 运行中" : "🔴 已停止") << std::endl;
    std::cout << "   下一个帧ID: " << next_frame_id_.load() << std::endl;
    std::cout << "   已完成结果数: " << get_completed_result_count() << std::endl;
    
    if (pipeline_manager_ && is_running_.load()) {
        pipeline_manager_->print_status();
    }
}

size_t HighwayEventDetector::get_pending_frame_count() const {
    if (!pipeline_manager_ || !is_running_.load()) {
        return 0;
    }
    
    return pipeline_manager_->get_seg_queue_size() + 
           pipeline_manager_->get_mask_queue_size() +
           pipeline_manager_->get_det_queue_size() +
           pipeline_manager_->get_track_queue_size() +
           pipeline_manager_->get_filter_queue_size();
}

size_t HighwayEventDetector::get_completed_result_count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(result_mutex_));
    return completed_results_.size();
}

void HighwayEventDetector::cleanup_results_before(uint64_t before_frame_id) {
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    auto it = completed_results_.begin();
    while (it != completed_results_.end()) {
        if (it->first < before_frame_id) {
            it = completed_results_.erase(it);
        } else {
            ++it;
        }
    }
    
    if (config_.enable_debug_log) {
        std::cout << "🧹 清理帧ID " << before_frame_id << " 之前的结果" << std::endl;
    }
}

void HighwayEventDetector::result_processing_thread() {
    if (config_.enable_debug_log) {
        std::cout << "🔄 结果处理线程启动" << std::endl;
    }
    
    while (result_thread_running_.load()) {
        if (!pipeline_manager_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        ImageDataPtr result;
        if (pipeline_manager_->get_final_result(result)) {
            if (result) {
                // 存储结果
                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    completed_results_[result->frame_idx] = result;
                    
                    if (config_.enable_debug_log) {
                        std::cout << "📋 缓存帧 " << result->frame_idx << " 的处理结果" << std::endl;
                    }
                }
                
                // 通知等待的线程
                result_cv_.notify_all();
                
                // 定期清理旧结果（保留最近1000个结果）
                if (completed_results_.size() > 1000) {
                    cleanup_old_results();
                }
            }
        } else {
            // get_final_result返回false，说明流水线可能已关闭
            if (!is_running_.load()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    if (config_.enable_debug_log) {
        std::cout << "🔄 结果处理线程退出" << std::endl;
    }
}

void HighwayEventDetector::cleanup_old_results() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    if (completed_results_.size() <= 500) {
        return;
    }
    
    // 找到最小的帧ID，清理一半旧的结果
    uint64_t min_frame_id = UINT64_MAX;
    for (const auto& pair : completed_results_) {
        min_frame_id = std::min(min_frame_id, pair.first);
    }
    
    uint64_t cleanup_threshold = min_frame_id + completed_results_.size() / 2;
    
    auto it = completed_results_.begin();
    while (it != completed_results_.end()) {
        if (it->first < cleanup_threshold) {
            it = completed_results_.erase(it);
        } else {
            ++it;
        }
    }
    
    if (config_.enable_debug_log) {
        std::cout << "🧹 自动清理旧结果，保留 " << completed_results_.size() << " 个结果" << std::endl;
    }
}
