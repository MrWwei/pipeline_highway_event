#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <memory>

/**
 * 图像缓冲区内存池 - 减少频繁的内存分配和释放
 * 用于复用cv::Mat对象，避免重复分配大块内存
 */
class ImageBufferPool {
private:
    std::queue<cv::Mat*> available_buffers_;
    std::mutex pool_mutex_;
    const size_t max_pool_size_;
    size_t allocated_count_;
    
public:
    explicit ImageBufferPool(size_t max_size = 50) 
        : max_pool_size_(max_size), allocated_count_(0) {}
    
    ~ImageBufferPool() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        while (!available_buffers_.empty()) {
            delete available_buffers_.front();
            available_buffers_.pop();
        }
    }
    
    /**
     * 获取一个指定尺寸的Mat缓冲区
     * 如果池中有合适的缓冲区则复用，否则分配新的
     */
    cv::Mat* acquire(int rows, int cols, int type) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // 尝试从池中获取可复用的缓冲区
        if (!available_buffers_.empty()) {
            cv::Mat* mat = available_buffers_.front();
            available_buffers_.pop();
            
            // 检查尺寸是否匹配，不匹配则重新创建
            if (mat->rows != rows || mat->cols != cols || mat->type() != type) {
                mat->create(rows, cols, type);
            }
            return mat;
        }
        
        // 池中没有可用缓冲区，分配新的
        allocated_count_++;
        return new cv::Mat(rows, cols, type);
    }
    
    /**
     * 归还缓冲区到池中
     * 如果池未满则保留供后续复用，否则直接释放
     */
    void release(cv::Mat* mat) {
        if (!mat) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (available_buffers_.size() < max_pool_size_) {
            // 池未满，归还供复用
            available_buffers_.push(mat);
        } else {
            // 池已满，直接释放
            delete mat;
            allocated_count_--;
        }
    }
    
    /**
     * 获取池的统计信息
     */
    struct PoolStats {
        size_t available_count;
        size_t allocated_count;
        size_t max_pool_size;
    };
    
    PoolStats get_stats() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return {available_buffers_.size(), allocated_count_, max_pool_size_};
    }
};

/**
 * 检测结果缓冲区池 - 复用BoundingBox向量
 */
class DetectionResultPool {
private:
    std::queue<std::vector<cv::Rect>*> available_vectors_;
    std::mutex pool_mutex_;
    const size_t max_pool_size_;
    
public:
    explicit DetectionResultPool(size_t max_size = 20) 
        : max_pool_size_(max_size) {}
    
    ~DetectionResultPool() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        while (!available_vectors_.empty()) {
            delete available_vectors_.front();
            available_vectors_.pop();
        }
    }
    
    std::vector<cv::Rect>* acquire() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (!available_vectors_.empty()) {
            auto* vec = available_vectors_.front();
            available_vectors_.pop();
            vec->clear(); // 清空之前的内容
            return vec;
        }
        
        auto* vec = new std::vector<cv::Rect>();
        vec->reserve(100); // 预分配空间
        return vec;
    }
    
    void release(std::vector<cv::Rect>* vec) {
        if (!vec) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (available_vectors_.size() < max_pool_size_) {
            available_vectors_.push(vec);
        } else {
            delete vec;
        }
    }
};

/**
 * 自定义删除器，用于智能指针自动归还到内存池
 */
struct BufferPoolDeleter {
    ImageBufferPool* pool;
    
    explicit BufferPoolDeleter(ImageBufferPool* p) : pool(p) {}
    
    void operator()(cv::Mat* mat) {
        if (pool) {
            pool->release(mat);
        } else {
            delete mat;
        }
    }
};

// 使用智能指针管理的Mat类型
using PooledMat = std::unique_ptr<cv::Mat, BufferPoolDeleter>;

/**
 * 全局内存池实例 - 单例模式
 */
class GlobalMemoryPools {
private:
    static std::unique_ptr<ImageBufferPool> image_pool_;
    static std::unique_ptr<DetectionResultPool> detection_pool_;
    static std::once_flag init_flag_;
    
    static void initialize() {
        image_pool_ = std::make_unique<ImageBufferPool>(50);
        detection_pool_ = std::make_unique<DetectionResultPool>(20);
    }
    
public:
    static ImageBufferPool& image_pool() {
        std::call_once(init_flag_, initialize);
        return *image_pool_;
    }
    
    static DetectionResultPool& detection_pool() {
        std::call_once(init_flag_, initialize);
        return *detection_pool_;
    }
    
    // 创建池管理的Mat
    static PooledMat create_pooled_mat(int rows, int cols, int type) {
        auto* mat = image_pool().acquire(rows, cols, type);
        return PooledMat(mat, BufferPoolDeleter(&image_pool()));
    }
};

// 静态成员定义（需要在.cpp文件中定义）
// std::unique_ptr<ImageBufferPool> GlobalMemoryPools::image_pool_;
// std::unique_ptr<DetectionResultPool> GlobalMemoryPools::detection_pool_;
// std::once_flag GlobalMemoryPools::init_flag_;

/**
 * 内存优化的ImageData - 使用内存池
 * 这是一个重构建议，展示如何集成内存池
 */
/*
struct OptimizedImageData {
    PooledMat imageMat;           // 使用池管理的原始图像
    PooledMat segInResizeMat;     // 使用池管理的调整尺寸图像
    
    uint64_t frame_idx;
    int width, height, channels;
    
    // 其他成员保持不变...
    std::vector<uint8_t> label_map;
    // ...
    
    OptimizedImageData(int img_rows, int img_cols, int img_type) 
        : imageMat(GlobalMemoryPools::create_pooled_mat(img_rows, img_cols, img_type))
        , segInResizeMat(GlobalMemoryPools::create_pooled_mat(1024, 1024, img_type))
        , frame_idx(0), width(img_cols), height(img_rows), channels(CV_MAT_CN(img_type)) {
        
        // 预分配缓冲区
        label_map.reserve(1024 * 1024);
    }
};
*/
