#include "batch_event_determine.h"
#include <iostream>
#include <algorithm>
// #include <execution>
#include <opencv2/imgproc.hpp>
#include <cmath>

BatchEventDetermine::BatchEventDetermine(int num_threads, const PipelineConfig* config)
    : num_threads_(num_threads), running_(false), stop_requested_(false),
      normal_direction_(1.0f, 0.0f) { // 默认水平向右为正常方向
    
    // 初始化配置
    if (config) {
        config_ = *config;
        
        // // 更新事件检测参数
        // if (config->illegal_parking_frames_threshold > 0) {
        //     event_params_.illegal_parking_frames_threshold = config->illegal_parking_frames_threshold;
        // }
        // if (config->abnormal_stay_frames_threshold > 0) {
        //     event_params_.abnormal_stay_frames_threshold = config->abnormal_stay_frames_threshold;
        // }
        // if (config->speed_limit_kmh > 0) {
        //     event_params_.speed_limit_kmh = config->speed_limit_kmh;
        // }
    }
    
    // 创建输入输出连接器
    input_connector_ = std::make_unique<BatchConnector>(10);
    output_connector_ = std::make_unique<BatchConnector>(10);
    
    // 初始化检测区域
    initialize_detection_zones();
}

BatchEventDetermine::~BatchEventDetermine() {
    stop();
}

void BatchEventDetermine::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    stop_requested_.store(false);
    
    // 启动连接器
    input_connector_->start();
    output_connector_->start();
    
    // 启动工作线程
    worker_threads_.clear();
    worker_threads_.reserve(num_threads_);
    
    for (int i = 0; i < num_threads_; ++i) {
        worker_threads_.emplace_back(&BatchEventDetermine::worker_thread_func, this);
    }
    
    std::cout << "✅ 批次事件判定已启动，使用 " << num_threads_ << " 个线程" << std::endl;
}

void BatchEventDetermine::stop() {
    if (!running_.load()) {
        return;
    }
    
    stop_requested_.store(true);
    running_.store(false);
    
    // 停止连接器
    input_connector_->stop();
    output_connector_->stop();
    
    // 等待所有工作线程结束
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "🛑 批次事件判定已停止" << std::endl;
}

bool BatchEventDetermine::add_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    return input_connector_->send_batch(batch);
}

bool BatchEventDetermine::get_processed_batch(BatchPtr& batch) {
    return output_connector_->receive_batch(batch);
}

bool BatchEventDetermine::process_batch(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "⚠️ 开始处理批次 " << batch->batch_id 
              << " 事件判定，包含 " << batch->actual_size << " 个图像" << std::endl;
    
    try {
        // 确保图像按帧序号排序
        std::sort(batch->images.begin(), batch->images.begin() + batch->actual_size,
                  [](const ImageDataPtr& a, const ImageDataPtr& b) {
                      return a->frame_idx < b->frame_idx;
                  });
        
        // 使用批次处理锁确保事件数据一致性
        std::lock_guard<std::mutex> batch_lock(batch_processing_mutex_);
        
        // 逐帧处理事件检测
        for (size_t i = 0; i < batch->actual_size; ++i) {
            process_image_events(batch->images[i]);
        }
        
        // 批次级事件分析
        analyze_batch_events(batch);
        
        // 事件筛选和去重
        filter_and_deduplicate_events(batch);
        
        // 标记批次完成
        batch->event_completed.store(true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 更新统计信息
        processed_batch_count_.fetch_add(1);
        total_processing_time_ms_.fetch_add(duration.count());
        total_images_processed_.fetch_add(batch->actual_size);
        
        // 统计检测到的事件数量
        uint64_t events_in_batch = 0;
        for (size_t i = 0; i < batch->actual_size; ++i) {
            // 这里需要根据实际的事件存储结构来统计
            // 暂时简化处理
        }
        total_events_detected_.fetch_add(events_in_batch);
        
        std::cout << "✅ 批次 " << batch->batch_id << " 事件判定完成，耗时: " 
                  << duration.count() << "ms，检测到 " << events_in_batch << " 个事件" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次 " << batch->batch_id << " 事件判定异常: " << e.what() << std::endl;
        return false;
    }
}

void BatchEventDetermine::worker_thread_func() {
    while (running_.load()) {
        BatchPtr batch;
        
        // 从输入连接器获取批次
        if (input_connector_->receive_batch(batch)) {
            if (batch) {
                // 处理批次
                bool success = process_batch(batch);
                
                if (success) {
                    // 发送到输出连接器
                    output_connector_->send_batch(batch);
                } else {
                    std::cerr << "❌ 批次 " << batch->batch_id << " 事件判定失败，丢弃" << std::endl;
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
}

void BatchEventDetermine::process_image_events(ImageDataPtr image) {
    if (!image || image->track_results.empty()) {
        return;
    }
    
    // 更新轨迹历史
    for (const auto& track_result : image->track_results) {
        update_trajectory_history(track_result, image->frame_idx);
        
        // 检测各种事件
        bool illegal_parking = detect_illegal_parking(track_result, image->frame_idx);
        bool abnormal_stay = detect_abnormal_stay(track_result, image->frame_idx);
        
        // 获取轨迹用于方向和速度分析
        std::vector<cv::Point> trajectory = get_trajectory_points(track_result.track_id);
        if (trajectory.size() >= event_params_.min_trajectory_points) {
            bool wrong_direction = detect_wrong_direction(trajectory);
            bool speed_violation = detect_speed_violation(trajectory, 1.0); // 1秒时间窗口
            
            // 记录事件（这里需要根据实际需求添加事件存储逻辑）
            if (illegal_parking || abnormal_stay || wrong_direction || speed_violation) {
                std::cout << "⚠️ 检测到事件 - 帧:" << image->frame_idx 
                          << ", 轨迹ID:" << track_result.track_id;
                if (illegal_parking) std::cout << " [违停]";
                if (abnormal_stay) std::cout << " [异常停留]";
                if (wrong_direction) std::cout << " [逆行]";
                if (speed_violation) std::cout << " [超速]";
                std::cout << std::endl;
            }
        }
    }
}

void BatchEventDetermine::analyze_batch_events(BatchPtr batch) {
    // 批次级事件分析，例如：
    // 1. 跨帧事件连续性分析
    // 2. 批次内事件模式识别
    // 3. 事件严重程度评估
    
    std::map<int, std::vector<size_t>> track_appearances;
    
    // 收集批次内每个轨迹的出现情况
    for (size_t i = 0; i < batch->actual_size; ++i) {
        for (const auto& track_result : batch->images[i]->track_results) {
            track_appearances[track_result.track_id].push_back(i);
        }
    }
    
    // 分析持续性事件
    for (const auto& [track_id, frame_indices] : track_appearances) {
        if (frame_indices.size() > 5) { // 在批次中出现超过5帧
            std::cout << "🔍 轨迹 " << track_id << " 在批次 " << batch->batch_id 
                      << " 中持续出现 " << frame_indices.size() << " 帧" << std::endl;
        }
    }
}

void BatchEventDetermine::filter_and_deduplicate_events(BatchPtr batch) {
    // 事件筛选和去重逻辑
    // 1. 移除重复的短时事件
    // 2. 合并连续的同类事件
    // 3. 过滤置信度低的事件
    
    // 这里可以添加具体的筛选逻辑
}

bool BatchEventDetermine::detect_illegal_parking(const ImageData::BoundingBox& track_result, uint64_t frame_idx) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    int track_id = track_result.track_id;
    if (trajectory_histories_.find(track_id) == trajectory_histories_.end()) {
        return false;
    }
    
    auto& history = trajectory_histories_[track_id];
    
    // 如果已经检测到违停事件，避免重复报告
    if (history.has_illegal_parking_event) {
        return false;
    }
    
    // 检查目标是否在违停区域
    cv::Point center = get_bounding_box_center(track_result);
    bool in_illegal_zone = false;
    
    for (const auto& zone : illegal_parking_zones_) {
        if (point_in_polygon(center, zone)) {
            in_illegal_zone = true;
            break;
        }
    }
    
    if (!in_illegal_zone) {
        return false;
    }
    
    // 检查停留时间
    uint64_t stay_frames = frame_idx - history.first_appearance + 1;
    if (stay_frames >= event_params_.illegal_parking_frames_threshold) {
        history.has_illegal_parking_event = true;
        return true;
    }
    
    return false;
}

bool BatchEventDetermine::detect_wrong_direction(const std::vector<cv::Point>& trajectory) {
    if (trajectory.size() < event_params_.min_trajectory_points) {
        return false;
    }
    
    // 计算轨迹方向向量
    cv::Point start = trajectory.front();
    cv::Point end = trajectory.back();
    cv::Point2f trajectory_direction(end.x - start.x, end.y - start.y);
    
    // 归一化
    float length = std::sqrt(trajectory_direction.x * trajectory_direction.x + 
                           trajectory_direction.y * trajectory_direction.y);
    if (length < 1.0f) return false; // 移动距离太小
    
    trajectory_direction.x /= length;
    trajectory_direction.y /= length;
    
    // 计算与正常方向的角度差
    double angle_diff = calculate_angle_difference(trajectory_direction, normal_direction_);
    
    return angle_diff > event_params_.wrong_direction_angle_threshold;
}

bool BatchEventDetermine::detect_abnormal_stay(const ImageData::BoundingBox& track_result, uint64_t frame_idx) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    int track_id = track_result.track_id;
    if (trajectory_histories_.find(track_id) == trajectory_histories_.end()) {
        return false;
    }
    
    auto& history = trajectory_histories_[track_id];
    
    // 如果已经检测到异常停留事件，避免重复报告
    if (history.has_abnormal_stay_event) {
        return false;
    }
    
    // 检查停留时间
    uint64_t stay_frames = frame_idx - history.first_appearance + 1;
    if (stay_frames < event_params_.abnormal_stay_frames_threshold) {
        return false;
    }
    
    // 检查移动距离
    if (history.points.size() >= 2) {
        cv::Point first_point = history.points.front();
        cv::Point last_point = history.points.back();
        double distance = std::sqrt(std::pow(last_point.x - first_point.x, 2) + 
                                  std::pow(last_point.y - first_point.y, 2));
        
        if (distance < event_params_.movement_threshold) {
            history.has_abnormal_stay_event = true;
            return true;
        }
    }
    
    return false;
}

bool BatchEventDetermine::detect_speed_violation(const std::vector<cv::Point>& trajectory, double time_span) {
    if (trajectory.size() < 2) {
        return false;
    }
    
    double speed = calculate_speed(trajectory, time_span);
    return speed > event_params_.speed_limit_kmh;
}

void BatchEventDetermine::update_trajectory_history(const ImageData::BoundingBox& track_result, uint64_t frame_idx) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    int track_id = track_result.track_id;
    cv::Point center = get_bounding_box_center(track_result);
    
    if (trajectory_histories_.find(track_id) == trajectory_histories_.end()) {
        // 创建新的轨迹历史
        TrajectoryHistory new_history;
        new_history.first_appearance = frame_idx;
        new_history.last_update = frame_idx;
        new_history.points.push_back(center);
        new_history.frame_indices.push_back(frame_idx);
        
        trajectory_histories_[track_id] = new_history;
    } else {
        // 更新现有轨迹历史
        auto& history = trajectory_histories_[track_id];
        history.last_update = frame_idx;
        history.points.push_back(center);
        history.frame_indices.push_back(frame_idx);
        
        // 限制历史长度，避免内存过度使用
        const size_t max_history_length = 1000;
        if (history.points.size() > max_history_length) {
            history.points.erase(history.points.begin());
            history.frame_indices.erase(history.frame_indices.begin());
        }
    }
}

std::vector<cv::Point> BatchEventDetermine::get_trajectory_points(int track_id) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    if (trajectory_histories_.find(track_id) != trajectory_histories_.end()) {
        return trajectory_histories_[track_id].points;
    }
    
    return {};
}

double BatchEventDetermine::calculate_speed(const std::vector<cv::Point>& trajectory, double time_span) {
    if (trajectory.size() < 2 || time_span <= 0) {
        return 0.0;
    }
    
    cv::Point start = trajectory.front();
    cv::Point end = trajectory.back();
    
    double distance_pixels = std::sqrt(std::pow(end.x - start.x, 2) + std::pow(end.y - start.y, 2));
    double distance_meters = distance_pixels / event_params_.pixels_per_meter;
    double speed_ms = distance_meters / time_span;
    double speed_kmh = speed_ms * 3.6; // 转换为km/h
    
    return speed_kmh;
}

bool BatchEventDetermine::point_in_polygon(const cv::Point& point, const std::vector<cv::Point>& polygon) {
    return cv::pointPolygonTest(polygon, point, false) >= 0;
}

double BatchEventDetermine::calculate_angle_difference(const cv::Point2f& dir1, const cv::Point2f& dir2) {
    double dot_product = dir1.x * dir2.x + dir1.y * dir2.y;
    double angle_radians = std::acos(std::clamp(dot_product, -1.0, 1.0));
    return angle_radians * 180.0 / CV_PI;
}

cv::Point BatchEventDetermine::get_bounding_box_center(const ImageData::BoundingBox& box) {
    return cv::Point((box.left + box.right) / 2, (box.top + box.bottom) / 2);
}

void BatchEventDetermine::initialize_detection_zones() {
    // 初始化违停检测区域（示例）
    // 实际使用时应该从配置文件加载
    
    std::vector<cv::Point> zone1 = {
        cv::Point(100, 100), cv::Point(300, 100), 
        cv::Point(300, 200), cv::Point(100, 200)
    };
    illegal_parking_zones_.push_back(zone1);
    
    std::vector<cv::Point> zone2 = {
        cv::Point(500, 300), cv::Point(700, 300),
        cv::Point(700, 400), cv::Point(500, 400)
    };
    illegal_parking_zones_.push_back(zone2);
    
    std::cout << "✅ 初始化了 " << illegal_parking_zones_.size() << " 个违停检测区域" << std::endl;
}

// BatchStage接口实现
std::string BatchEventDetermine::get_stage_name() const {
    return "批次事件判定";
}

size_t BatchEventDetermine::get_processed_count() const {
    return processed_batch_count_.load();
}

double BatchEventDetermine::get_average_processing_time() const {
    auto count = processed_batch_count_.load();
    if (count == 0) return 0.0;
    return (double)total_processing_time_ms_.load() / count;
}

size_t BatchEventDetermine::get_queue_size() const {
    return input_connector_->get_queue_size();
}
