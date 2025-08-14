#pragma once

#include "batch_data.h"
#include "detect.h"
#include "pipeline_config.h"
#include <thread>
#include <atomic>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

/**
 * 批次目标检测器
 * 继承自BatchStage，负责对图像批次进行目标检测
 * 支持车辆检测和行人检测
 */
class BatchObjectDetection : public BatchStage {
public:
    explicit BatchObjectDetection(int num_threads = 4, const PipelineConfig* config = nullptr);
    virtual ~BatchObjectDetection();
    
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
    
    // 处理单个图像的目标检测
    void process_image_detection(ImageDataPtr image, int thread_id);
    
    // 执行目标检测算法
    void perform_object_detection(ImageDataPtr image, int thread_id);
    
    // 初始化检测模型
    bool initialize_detection_models();
    
    // 清理检测模型
    void cleanup_detection_models();

private:
    // 基本配置
    int num_threads_;
    PipelineConfig config_;
    
    // 线程管理
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // 检测模型实例 - 每个线程独立的模型实例
    std::vector<std::unique_ptr<xtkj::IDetect>> car_detect_instances_;
    std::vector<std::unique_ptr<xtkj::IDetect>> personal_detect_instances_;
    
    // 批次队列
    std::unique_ptr<BatchConnector> input_connector_;
    std::unique_ptr<BatchConnector> output_connector_;
    
    // 性能统计
    std::atomic<size_t> processed_batch_count_{0};
    std::atomic<uint64_t> total_processing_time_ms_{0};
    std::atomic<uint64_t> total_images_processed_{0};
    
    // CUDA优化相关
    bool cuda_available_;
    mutable std::mutex gpu_mutex_;
    cv::cuda::GpuMat gpu_src_cache_;
    cv::cuda::GpuMat gpu_dst_cache_;
    
    // 检测参数配置
    float confidence_threshold_;
    float nms_threshold_;
    bool enable_car_detection_;
    bool enable_person_detection_;
};
