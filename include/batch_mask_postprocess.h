#pragma once

#include "batch_data.h"
#include "pipeline_config.h"
#include "thread_pool.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <future>

/**
 * 批次Mask后处理器
 * 继承自BatchStage，负责对语义分割结果进行批次后处理
 * 使用线程池并发处理批次中的每个图像数据
 * 如去除小的白色区域、形态学操作等
 */
class BatchMaskPostProcess : public BatchStage {
public:
    explicit BatchMaskPostProcess(int num_threads = 4);
    virtual ~BatchMaskPostProcess();
    
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
    
    // 处理单个图像的mask后处理
    void process_image_mask(ImageDataPtr image);
    
    // 使用线程池并发处理批次中的所有图像
    bool process_batch_with_threadpool(BatchPtr batch);

private:
    // 基本配置
    int num_threads_;
    
    // 线程池
    std::unique_ptr<ThreadPool> thread_pool_;
    
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
    
    // Mask后处理参数
    int min_area_threshold_;           // 最小区域阈值
    int morphology_kernel_size_;       // 形态学操作核大小
    double roi_expansion_ratio_;       // ROI扩展比例
};
