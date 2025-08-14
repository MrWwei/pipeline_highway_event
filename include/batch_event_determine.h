#pragma once

#include "batch_data.h"
#include "event_type.h"
#include "pipeline_config.h"
#include <thread>
#include <atomic>
#include <map>
#include <vector>

/**
 * 批次事件判定器
 * 继承自BatchStage，负责对跟踪结果进行批次事件分析和判定
 * 支持多种事件类型：违停、逆行、异常停留等
 */
class BatchEventDetermine : public BatchStage {
public:
    explicit BatchEventDetermine(int num_threads = 2, const PipelineConfig* config = nullptr);
    virtual ~BatchEventDetermine();
    
    // BatchStage接口实现
    bool process_batch(BatchPtr batch) override;
    std::string get_stage_name() const override;
    size_t get_processed_count() const override;
    double get_average_processing_time() const override;
    size_t get_queue_size() const override;
    void start() override;
    void stop() override;
    
    // 获取输入批次
    bool add_batch(BatchPtr batch);
    
    // 获取处理完成的批次
    bool get_processed_batch(BatchPtr& batch);

private:
    // 工作线程函数
    void worker_thread_func();
    
    // 处理单个图像的事件判定
    void process_image_events(ImageDataPtr image);
    
    // 批次级事件分析
    void analyze_batch_events(BatchPtr batch);
    
    // 各种事件检测方法
    bool detect_illegal_parking(const ImageData::BoundingBox& track_result, uint64_t frame_idx);
    bool detect_wrong_direction(const std::vector<cv::Point>& trajectory);
    bool detect_abnormal_stay(const ImageData::BoundingBox& track_result, uint64_t frame_idx);
    bool detect_speed_violation(const std::vector<cv::Point>& trajectory, double time_span);
    
    // 轨迹分析
    void update_trajectory_history(const ImageData::BoundingBox& track_result, uint64_t frame_idx);
    std::vector<cv::Point> get_trajectory_points(int track_id);
    double calculate_speed(const std::vector<cv::Point>& trajectory, double time_span);
    
    // 事件筛选和去重
    void filter_and_deduplicate_events(BatchPtr batch);

private:
    // 基本配置
    int num_threads_;
    PipelineConfig config_;
    
    // 线程管理
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // 批次队列
    std::unique_ptr<BatchConnector> input_connector_;
    std::unique_ptr<BatchConnector> output_connector_;
    
    // 性能统计
    std::atomic<size_t> processed_batch_count_{0};
    std::atomic<uint64_t> total_processing_time_ms_{0};
    std::atomic<uint64_t> total_images_processed_{0};
    std::atomic<uint64_t> total_events_detected_{0};
    
    // 事件检测参数
    struct EventParams {
        // 违停检测参数
        int illegal_parking_frames_threshold = 150;  // 5秒@30fps
        double illegal_parking_area_threshold = 0.3; // 30%重叠认为在违停区域
        
        // 异常停留参数
        int abnormal_stay_frames_threshold = 300;    // 10秒@30fps
        double movement_threshold = 5.0;             // 5像素移动阈值
        
        // 逆行检测参数
        double wrong_direction_angle_threshold = 135.0; // 角度阈值
        int min_trajectory_points = 10;              // 最少轨迹点数
        
        // 超速检测参数
        double speed_limit_kmh = 60.0;               // 限速60km/h
        double pixels_per_meter = 10.0;              // 像素/米比例
    } event_params_;
    
    // 轨迹历史数据
    struct TrajectoryHistory {
        std::vector<cv::Point> points;
        std::vector<uint64_t> frame_indices;
        uint64_t first_appearance;
        uint64_t last_update;
        bool has_illegal_parking_event = false;
        bool has_wrong_direction_event = false;
        bool has_abnormal_stay_event = false;
    };
    
    std::map<int, TrajectoryHistory> trajectory_histories_;
    std::mutex trajectory_mutex_;
    
    // 违停区域定义（多边形）
    std::vector<std::vector<cv::Point>> illegal_parking_zones_;
    
    // 正常行驶方向（道路方向向量）
    cv::Point2f normal_direction_;
    
    // 批次处理同步
    std::mutex batch_processing_mutex_;
    
    // 工具方法
    bool point_in_polygon(const cv::Point& point, const std::vector<cv::Point>& polygon);
    double calculate_angle_difference(const cv::Point2f& dir1, const cv::Point2f& dir2);
    cv::Point get_bounding_box_center(const ImageData::BoundingBox& box);
    void initialize_detection_zones();
};
