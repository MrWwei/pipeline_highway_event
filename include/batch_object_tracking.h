#pragma once

#include "batch_data.h"
#include "byte_track.h"
#include "vehicle_parking_detect.h"
#include "pipeline_config.h"
#include <thread>
#include <atomic>
#include <map>
#include <mutex>

/**
 * 批次目标跟踪器
 * 继承自BatchStage，负责对检测结果进行批次目标跟踪
 * 支持跨帧目标关联和轨迹管理
 */
class BatchObjectTracking : public BatchStage {
public:
    explicit BatchObjectTracking(int num_threads = 2, const PipelineConfig* config = nullptr);
    virtual ~BatchObjectTracking();
    
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
    
    // 处理单个图像的目标跟踪
    void process_image_tracking(ImageDataPtr image, int thread_id);
    
    // 执行目标跟踪算法
    void perform_object_tracking(ImageDataPtr image, int thread_id);
    
    // 初始化跟踪模型
    bool initialize_tracking_models();
    
    // 清理跟踪模型
    void cleanup_tracking_models();
    
private:
    // 基本配置
    int num_threads_;
    PipelineConfig config_;
    
    // 线程管理
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // 跟踪模型实例 - 每个线程独立的模型实例
    std::vector<std::unique_ptr<xtkj::ITracker>> track_instances_;
    VehicleParkingDetect* vehicle_parking_instance_;
    
    // 批次队列
    std::unique_ptr<BatchConnector> input_connector_;
    std::unique_ptr<BatchConnector> output_connector_;
    
    // 性能统计
    std::atomic<size_t> processed_batch_count_{0};
    std::atomic<uint64_t> total_processing_time_ms_{0};
    std::atomic<uint64_t> total_images_processed_{0};
    
    // 跟踪参数配置
    float tracking_confidence_threshold_;
    int max_disappeared_frames_;
    float iou_threshold_;
    
    // 全局轨迹数据库（跨批次共享）
    struct TrajectoryInfo {
        int track_id;
        cv::Rect last_bbox;
        uint64_t last_frame_idx;
        int disappeared_count;
        bool is_active;
    };
    
    std::map<int, TrajectoryInfo> trajectory_database_;
    std::mutex trajectory_mutex_;
    int next_track_id_;
    
    // 批次处理同步
    std::mutex batch_processing_mutex_;
};
