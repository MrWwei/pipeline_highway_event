#include "batch_data.h"
#include <iostream>
#include <algorithm>

// BatchBuffer implementation

BatchBuffer::BatchBuffer(std::chrono::milliseconds flush_timeout, size_t max_ready_batches)
    : next_batch_id_(1), flush_timeout_(flush_timeout), max_ready_batches_(max_ready_batches),
      running_(false), stop_requested_(false) {
    current_collecting_batch_ = nullptr;
}

BatchBuffer::~BatchBuffer() {
    stop();
}

void BatchBuffer::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    stop_requested_.store(false);
    
    // 启动自动刷新线程
    flush_thread_ = std::thread(&BatchBuffer::flush_thread_func, this);
    
    std::cout << "✅ BatchBuffer 已启动，刷新超时: " << flush_timeout_.count() << "ms" << std::endl;
}

void BatchBuffer::stop() {
    if (!running_.load()) {
        return;
    }
    
    stop_requested_.store(true);
    running_.store(false);
    
    // 刷新当前批次
    flush_current_batch();
    
    // 等待刷新线程结束
    if (flush_thread_.joinable()) {
        flush_thread_.join();
    }
    
    std::cout << "🛑 BatchBuffer 已停止" << std::endl;
}

bool BatchBuffer::add_image(ImageDataPtr image) {
    if (!running_.load() || !image) {
        return false;
    }

    // 背压检查：如果就绪队列已满，则阻塞等待
    {
        std::unique_lock<std::mutex> ready_lock(ready_mutex_);
        ready_cv_.wait(ready_lock, [this]() {
            return ready_batches_.size() < max_ready_batches_ || !running_.load();
        });
        
        if (!running_.load()) {
            return false;
        }
    }
    
    std::lock_guard<std::mutex> lock(collect_mutex_);
    
    // 如果当前没有收集批次，创建新的
    if (!current_collecting_batch_) {
        current_collecting_batch_ = std::make_shared<ImageBatch>(next_batch_id_++);
    }
    
    // 添加图像到当前批次
    bool added = current_collecting_batch_->add_image(image);
    if (!added) {
        std::cerr << "❌ 无法添加图像到批次，批次可能已满" << std::endl;
        return false;
    }
    
    total_images_received_.fetch_add(1);
    
    // 如果批次已满，移动到就绪队列
    if (current_collecting_batch_->is_full()) {
        move_batch_to_ready(current_collecting_batch_);
        current_collecting_batch_ = nullptr;
    }
    
    return true;
}

bool BatchBuffer::get_ready_batch(BatchPtr& batch) {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    
    // 等待就绪批次
    ready_cv_.wait(lock, [this]() {
        return !ready_batches_.empty() || !running_.load();
    });
    
    if (!running_.load() && ready_batches_.empty()) {
        return false;
    }
    
    if (!ready_batches_.empty()) {
        batch = ready_batches_.front();
        ready_batches_.pop();
        
        // 通知等待的add_image线程
        lock.unlock();
        ready_cv_.notify_one();
        
        return true;
    }
    
    return false;
}

bool BatchBuffer::try_get_ready_batch(BatchPtr& batch) {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    
    if (!ready_batches_.empty()) {
        batch = ready_batches_.front();
        ready_batches_.pop();
        
        // 通知等待的add_image线程
        lock.unlock();
        ready_cv_.notify_one();
        
        return true;
    }
    
    return false;
}

void BatchBuffer::flush_current_batch() {
    std::lock_guard<std::mutex> lock(collect_mutex_);
    
    if (current_collecting_batch_ && !current_collecting_batch_->is_empty()) {
        std::cout << "🚿 强制刷新批次 " << current_collecting_batch_->batch_id 
                  << "，包含 " << current_collecting_batch_->actual_size << " 个图像" << std::endl;
        move_batch_to_ready(current_collecting_batch_);
        current_collecting_batch_ = nullptr;
    }
}

size_t BatchBuffer::get_ready_batch_count() const {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    return ready_batches_.size();
}

size_t BatchBuffer::get_current_collecting_size() const {
    std::lock_guard<std::mutex> lock(collect_mutex_);
    return current_collecting_batch_ ? current_collecting_batch_->actual_size : 0;
}

uint64_t BatchBuffer::get_total_batches_created() const {
    return total_batches_created_.load();
}

size_t BatchBuffer::get_max_ready_batches() const {
    return max_ready_batches_;
}

bool BatchBuffer::is_ready_queue_full() const {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    return ready_batches_.size() >= max_ready_batches_;
}

void BatchBuffer::flush_thread_func() {
    while (running_.load()) {
        std::this_thread::sleep_for(flush_timeout_);
        
        if (!running_.load()) {
            break;
        }
        // 检查是否需要刷新当前批次
        {
            std::lock_guard<std::mutex> lock(collect_mutex_);
            if (current_collecting_batch_ && !current_collecting_batch_->is_empty()) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - current_collecting_batch_->created_time);
                
                if (elapsed >= flush_timeout_) {
                    std::cout << "⏰ 超时刷新批次 " << current_collecting_batch_->batch_id 
                              << "，包含 " << current_collecting_batch_->actual_size 
                              << " 个图像，等待时间: " << elapsed.count() << "ms" << std::endl;
                    std::cout << "规定超时时间是 " << flush_timeout_.count() << " ms" << std::endl;
                    move_batch_to_ready(current_collecting_batch_);
                    current_collecting_batch_ = nullptr;
                }
            }
        }
    }
}

void BatchBuffer::move_batch_to_ready(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(ready_mutex_);
        
        // 检查是否会超过限制
        if (ready_batches_.size() >= max_ready_batches_) {
            std::cout << "⚠️ 就绪队列已满 (" << ready_batches_.size() 
                      << "/" << max_ready_batches_ << ")，批次 " << batch->batch_id 
                      << " 将丢弃" << std::endl;
            return;
        }
        
        ready_batches_.push(batch);
        total_batches_created_.fetch_add(1);
    }
    ready_cv_.notify_one();
    
    // std::cout << "📦 批次 " << batch->batch_id << " 已就绪，包含 " 
    //           << batch->actual_size << " 个图像，队列大小: " 
    //           << ready_batches_.size() << "/" << max_ready_batches_ << std::endl;
}

// BatchConnector implementation

BatchConnector::BatchConnector(size_t max_queue_size)
    : max_queue_size_(max_queue_size), running_(false) {
}

BatchConnector::~BatchConnector() {
    stop();
}

void BatchConnector::start() {
    running_.store(true);
    std::cout << "✅ BatchConnector 已启动，最大队列大小: " << max_queue_size_ << std::endl;
}

void BatchConnector::stop() {
    running_.store(false);
    queue_cv_.notify_all();
    std::cout << "🛑 BatchConnector 已停止" << std::endl;
}

bool BatchConnector::send_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // 等待队列有空间
    queue_cv_.wait(lock, [this]() {
        return batch_queue_.size() < max_queue_size_ || !running_.load();
    });
    
    if (!running_.load()) {
        return false;
    }
    
    batch_queue_.push(batch);
    total_sent_.fetch_add(1);
    
    lock.unlock();
    queue_cv_.notify_one();
    
    return true;
}

bool BatchConnector::receive_batch(BatchPtr& batch) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // 等待有批次可用
    queue_cv_.wait(lock, [this]() {
        return !batch_queue_.empty() || !running_.load();
    });
    
    if (!running_.load() && batch_queue_.empty()) {
        return false;
    }
    
    if (!batch_queue_.empty()) {
        batch = batch_queue_.front();
        batch_queue_.pop();
        total_received_.fetch_add(1);
        
        lock.unlock();
        queue_cv_.notify_one(); // 通知可能等待发送的线程
        
        return true;
    }
    
    return false;
}

bool BatchConnector::try_receive_batch(BatchPtr& batch) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    if (!batch_queue_.empty()) {
        batch = batch_queue_.front();
        batch_queue_.pop();
        total_received_.fetch_add(1);
        
        lock.unlock();
        queue_cv_.notify_one(); // 通知可能等待发送的线程
        
        return true;
    }
    
    return false;
}

size_t BatchConnector::get_queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return batch_queue_.size();
}

size_t BatchConnector::get_max_queue_size() const {
    return max_queue_size_;
}

bool BatchConnector::is_full() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return batch_queue_.size() >= max_queue_size_;
}
