#pragma once

#include "batch_data.h"
#include "trt_seg_model.h"
#include "pipeline_config.h"
#include "thread_pool.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <future>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

/**
 * 批次语义分割处理器
 * 继承自BatchStage，专门负责对32个图像批次进行语义分割处理
 * 支持多线程并发处理批次内的图像
 */
class BatchSemanticSegmentation : public BatchStage {
public:
    explicit BatchSemanticSegmentation(
        int num_threads = 4, 
        const PipelineConfig* config = nullptr
    );
    
    virtual ~BatchSemanticSegmentation();
    
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
    
    // 更新配置参数
    void change_params(const PipelineConfig& config);

private:
    // 工作线程函数
    void worker_thread_func();
    
    // 批次预处理（使用线程池并发处理）
    void preprocess_batch(BatchPtr batch);
    
    // 使用线程池并发预处理批次中的所有图像
    bool preprocess_batch_with_threadpool(BatchPtr batch);
    
    // 批次语义分割推理
    bool inference_batch(BatchPtr batch);
    
    // 批次后处理
    void postprocess_batch(BatchPtr batch);
    
    // 单个图像预处理（在批次中）
    void preprocess_image(ImageDataPtr image, int thread_id);
    
    // 保存分割结果（如果启用）
    void save_segmentation_result(ImageDataPtr image);

private:
    // 基本配置
    int num_threads_;
    PipelineConfig config_;
    
    // 线程池
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // 线程管理
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // 模型实例 - 每个线程独立的模型实例
    std::vector<std::unique_ptr<PureTRTPPSeg>> seg_instances_;
    
    // 批次队列
    std::unique_ptr<BatchConnector> input_connector_;
    std::unique_ptr<BatchConnector> output_connector_;
    
    // 性能统计
    std::atomic<size_t> processed_batch_count_{0};
    std::atomic<uint64_t> total_processing_time_ms_{0};
    std::atomic<uint64_t> total_images_processed_{0};
    
    // CUDA优化相关
    bool cuda_available_;
    cv::cuda::GpuMat gpu_src_cache_;
    cv::cuda::GpuMat gpu_dst_cache_;
    mutable std::mutex gpu_mutex_;
    
    // 分割结果保存配置
    bool enable_seg_show_;
    std::string seg_show_image_path_;
    int seg_show_interval_;
    
    // 线程同步 - 用于批次内多线程协作
    struct BatchContext {
        BatchPtr batch;
        std::atomic<int> completed_threads{0};
        std::atomic<bool> preprocessing_done{false};
        std::atomic<bool> inference_done{false};
        std::atomic<bool> postprocessing_done{false};
        std::mutex context_mutex;
        std::condition_variable context_cv;
        std::chrono::high_resolution_clock::time_point start_time;
    };
    
    std::queue<std::shared_ptr<BatchContext>> processing_queue_;
    std::mutex processing_queue_mutex_;
    std::condition_variable processing_queue_cv_;
    
    // 内部工具方法
    bool initialize_seg_models();
    void cleanup_seg_models();
    std::shared_ptr<BatchContext> create_batch_context(BatchPtr batch);
    void process_batch_context(std::shared_ptr<BatchContext> context, int thread_id);
};
