#pragma once

#include "batch_data.h"
#include "batch_semantic_segmentation.h"
#include "batch_mask_postprocess.h"
#include "batch_object_detection.h"
#include "batch_object_tracking.h"
#include "batch_event_determine.h"
#include "pipeline_config.h"
#include "memory_monitor.h"
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

/**
 * 批次流水线管理器
 * 管理整个批次处理流水线：输入收集 -> 语义分割 -> Mask后处理 -> 目标检测 -> 目标跟踪 -> 事件判定
 * 每个阶段处理完整的32个图像批次后再传递给下一阶段
 */
class BatchPipelineManager {
public:
    explicit BatchPipelineManager(const PipelineConfig& config);
    virtual ~BatchPipelineManager();
    
    // 启动批次流水线
    void start();
    
    // 停止批次流水线
    void stop();
    
    // 添加单个图像（自动组装成批次）
    bool add_image(ImageDataPtr image);
    
    // 获取处理完成的批次结果
    bool get_result_batch(BatchPtr& batch);
    
    // 获取处理完成的单个图像结果
    bool get_result_image(ImageDataPtr& image);
    
    // 打印流水线状态
    void print_status() const;
    
    // 获取统计信息
    struct Statistics {
        uint64_t total_images_input;
        uint64_t total_batches_processed;
        uint64_t total_images_output;
        double average_batch_processing_time_ms;
        double throughput_images_per_second;
        size_t current_input_buffer_size;
        size_t current_output_buffer_size;
    };
    
    Statistics get_statistics() const;
    
    // 内存监控相关方法
    void start_memory_monitoring();
    void stop_memory_monitoring();
    void print_memory_report();
    bool is_memory_leak_detected();
    void set_memory_leak_threshold(double threshold_mb_per_min);
    MemoryStats get_current_memory_stats();

private:
    // 配置
    PipelineConfig config_;
    
    // 运行状态
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // 批次收集器
    std::unique_ptr<BatchBuffer> input_buffer_;
    
    // 处理阶段
    std::unique_ptr<BatchSemanticSegmentation> semantic_seg_;
    std::unique_ptr<BatchMaskPostProcess> mask_postprocess_;
    std::unique_ptr<BatchObjectDetection> object_detection_;
    std::unique_ptr<BatchObjectTracking> object_tracking_;
    std::unique_ptr<BatchEventDetermine> event_determine_;
    
    // 阶段连接器
    std::unique_ptr<BatchConnector> seg_to_mask_connector_;
    std::unique_ptr<BatchConnector> mask_to_detection_connector_;
    std::unique_ptr<BatchConnector> detection_to_tracking_connector_;
    std::unique_ptr<BatchConnector> tracking_to_event_connector_;
    
    // 结果收集
    std::unique_ptr<BatchConnector> final_result_connector_;
    std::queue<ImageDataPtr> result_image_queue_;
    std::mutex result_queue_mutex_;
    std::condition_variable result_queue_cv_;
    
    // 流水线协调线程
    std::thread seg_coordinator_thread_;
    std::thread mask_coordinator_thread_;
    std::thread detection_coordinator_thread_;
    std::thread tracking_coordinator_thread_;
    std::thread event_coordinator_thread_;
    std::thread result_collector_thread_;
    
    // 性能统计
    std::atomic<uint64_t> total_images_input_{0};
    std::atomic<uint64_t> total_batches_processed_{0};
    std::atomic<uint64_t> total_images_output_{0};
    std::chrono::high_resolution_clock::time_point start_time_;
    
        // 状态监控线程
    std::thread status_monitor_thread_;
    std::chrono::seconds status_print_interval_;
    
    // 内存监控器
    std::unique_ptr<MemoryMonitor> memory_monitor_;
    
    // 初始化和清理方法
    
    // 协调线程函数
    void seg_coordinator_func();      // 语义分割协调
    void mask_coordinator_func();     // Mask后处理协调
    void detection_coordinator_func(); // 目标检测协调
    void tracking_coordinator_func();  // 目标跟踪协调
    void event_coordinator_func();     // 事件判定协调
    void result_collector_func();      // 结果收集
    
    // 状态监控函数
    void status_monitor_func();
    
    // 工具函数
    void decompose_batch_to_images(BatchPtr batch);
    bool initialize_stages();
    void cleanup_stages();
};
