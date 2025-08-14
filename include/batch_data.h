#pragma once

#include "image_data.h"
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>

/**
 * 批次数据容器 - 包含32个图像数据的批次
 */
struct ImageBatch {
    static constexpr size_t BATCH_SIZE = 32;
    
    std::vector<ImageDataPtr> images;                           // 32个图像数据
    uint64_t batch_id;                                          // 批次ID
    size_t actual_size;                                         // 实际图像数量（可能小于32）
    std::chrono::high_resolution_clock::time_point created_time; // 创建时间
    std::chrono::high_resolution_clock::time_point start_time;   // 开始处理时间
    
    // 处理状态跟踪
    std::atomic<size_t> completed_stages{0};                   // 已完成的阶段数
    std::atomic<bool> is_processing{false};                    // 是否正在处理
    std::atomic<bool> is_completed{false};                     // 是否处理完成
    
    // 批次处理阶段标记
    std::atomic<bool> segmentation_completed{false};
    std::atomic<bool> mask_postprocess_completed{false};
    std::atomic<bool> detection_completed{false};
    std::atomic<bool> tracking_completed{false};
    std::atomic<bool> event_completed{false};
    
    // 构造函数
    ImageBatch() : batch_id(0), actual_size(0) {
        images.reserve(BATCH_SIZE);
        created_time = std::chrono::high_resolution_clock::now();
    }
    
    explicit ImageBatch(uint64_t id) : batch_id(id), actual_size(0) {
        images.reserve(BATCH_SIZE);
        created_time = std::chrono::high_resolution_clock::now();
    }
    
    // 添加图像到批次
    bool add_image(ImageDataPtr image) {
        if (actual_size >= BATCH_SIZE) {
            return false;
        }
        images.push_back(image);
        actual_size++;
        return true;
    }
    
    // 检查批次是否已满
    bool is_full() const {
        return actual_size >= BATCH_SIZE;
    }
    
    // 检查批次是否为空
    bool is_empty() const {
        return actual_size == 0;
    }
    
    // 获取批次处理耗时
    double get_processing_time_ms() const {
        if (start_time.time_since_epoch().count() == 0) {
            return 0.0;
        }
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }
    
    // 开始处理（设置开始时间）
    void start_processing() {
        start_time = std::chrono::high_resolution_clock::now();
        is_processing.store(true);
    }
    
    // 完成处理
    void complete_processing() {
        is_processing.store(false);
        is_completed.store(true);
    }
};

using BatchPtr = std::shared_ptr<ImageBatch>;

/**
 * 批次缓冲区 - 负责收集单个图像并组装成批次
 * 支持背压机制，防止内存无限增长
 */
class BatchBuffer {
public:
    explicit BatchBuffer(
        std::chrono::milliseconds flush_timeout = std::chrono::milliseconds(100),
        size_t max_ready_batches = 50  // 最大就绪批次数量，实现背压
    );
    ~BatchBuffer();
    
    // 启动批次收集
    void start();
    
    // 停止批次收集
    void stop();
    
    // 添加单个图像，自动组装成批次
    bool add_image(ImageDataPtr image);
    
    // 获取就绪的批次
    bool get_ready_batch(BatchPtr& batch);
    
    // 强制刷新当前收集的批次
    void flush_current_batch();
    
    // 获取统计信息
    size_t get_ready_batch_count() const;
    size_t get_current_collecting_size() const;
    uint64_t get_total_batches_created() const;
    size_t get_max_ready_batches() const;
    bool is_ready_queue_full() const;

private:
    // 批次收集相关
    mutable std::mutex collect_mutex_;
    BatchPtr current_collecting_batch_;
    uint64_t next_batch_id_;
    
    // 就绪批次队列
    mutable std::mutex ready_mutex_;
    std::queue<BatchPtr> ready_batches_;
    std::condition_variable ready_cv_;
    size_t max_ready_batches_;  // 就绪批次队列的最大大小
    
    // 自动刷新机制
    std::thread flush_thread_;
    std::chrono::milliseconds flush_timeout_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // 统计信息
    std::atomic<uint64_t> total_batches_created_{0};
    std::atomic<uint64_t> total_images_received_{0};
    
    // 内部方法
    void flush_thread_func();
    void move_batch_to_ready(BatchPtr batch);
};

/**
 * 批次阶段接口 - 定义批次处理阶段的统一接口
 */
class BatchStage {
public:
    virtual ~BatchStage() = default;
    
    // 处理批次数据
    virtual bool process_batch(BatchPtr batch) = 0;
    
    // 获取阶段名称
    virtual std::string get_stage_name() const = 0;
    
    // 获取处理的批次数量
    virtual size_t get_processed_count() const = 0;
    
    // 获取平均处理时间（毫秒）
    virtual double get_average_processing_time() const = 0;
    
    // 获取当前队列大小
    virtual size_t get_queue_size() const = 0;
    
    // 启动处理阶段
    virtual void start() = 0;
    
    // 停止处理阶段
    virtual void stop() = 0;
};

/**
 * 批次连接器 - 连接两个批次处理阶段
 */
class BatchConnector {
public:
    explicit BatchConnector(size_t max_queue_size = 10);
    ~BatchConnector();
    
    // 启动连接器
    void start();
    
    // 停止连接器
    void stop();
    
    // 向连接器发送批次
    bool send_batch(BatchPtr batch);
    
    // 从连接器接收批次
    bool receive_batch(BatchPtr& batch);
    
    // 获取队列状态
    size_t get_queue_size() const;
    size_t get_max_queue_size() const;
    bool is_full() const;
    
private:
    mutable std::mutex queue_mutex_;
    std::queue<BatchPtr> batch_queue_;
    std::condition_variable queue_cv_;
    
    size_t max_queue_size_;
    std::atomic<bool> running_;
    
    // 统计信息
    std::atomic<uint64_t> total_sent_{0};
    std::atomic<uint64_t> total_received_{0};
};
