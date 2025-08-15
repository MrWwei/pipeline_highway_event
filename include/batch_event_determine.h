#pragma once

#include "batch_data.h"
#include "event_type.h"
#include "event_utils.h"
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
    void perform_event_determination(ImageDataPtr image);
    int calculate_box_width(const ImageData::BoundingBox& box) const;
    bool is_box_in_region(const ImageData::BoundingBox& box, 
                                 int region_top, int region_bottom) const;
    ImageData::BoundingBox* find_min_width_box_in_region(
    const std::vector<ImageData::BoundingBox>& boxes,
    int region_top, int region_bottom) const;
    void drawEmergencyLaneQuarterPoints(cv::Mat &image,
                                 const EmergencyLaneResult &emergency_lane);
    ObjectStatus determineObjectStatus(const ImageData::BoundingBox &box,
                        const EmergencyLaneResult &emergency_lane);


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

    float top_fraction_ = 0.25f; // 事件判定区域上边界占图像高度的比例
    float bottom_fraction_ = 0.75f; // 事件判定区域下边界占图像高度的比例
    float times_car_width_ = 3.0f; // 车宽倍数，用于计算车道线位置

    std::string lane_show_image_path_; // 车道线可视化图像保存路径
    
    // 性能统计
    std::atomic<size_t> processed_batch_count_{0};
    std::atomic<uint64_t> total_processing_time_ms_{0};
    std::atomic<uint64_t> total_images_processed_{0};
    std::atomic<uint64_t> total_events_detected_{0};
    
    // 批次处理同步
    std::mutex batch_processing_mutex_;
    
};
