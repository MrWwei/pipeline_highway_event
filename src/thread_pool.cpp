#include "thread_pool.h"
#include "logger_manager.h"
#include <iostream>

ThreadPool::ThreadPool(size_t threads) : running_(true) {
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4; // 默认4个线程
    }
    
    workers_.reserve(threads);
    
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    this->condition_.wait(lock, [this] { 
                        return !this->running_.load() || !this->tasks_.empty(); 
                    });
                    
                    if (!this->running_.load() && this->tasks_.empty()) {
                        return;
                    }
                    
                    if (!this->tasks_.empty()) {
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    } else {
                        continue;
                    }
                }

                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "ThreadPool任务执行异常: " << e.what() << std::endl;
                } catch (...) {
                    LOG_ERROR("ThreadPool任务执行未知异常");
                }
            }
        });
    }
    
    std::cout << "✅ ThreadPool启动，线程数: " << threads 
              << "，最大队列大小: " << MAX_QUEUE_SIZE << std::endl;
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::stop() {
    if (!running_.load()) {
        return;
    }
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        running_.store(false);
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers_.clear();
    
    // 清空剩余任务
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        std::queue<std::function<void()>> empty_queue;
        tasks_.swap(empty_queue);
    }
    
    LOG_INFO("🛑 ThreadPool已停止");
}

size_t ThreadPool::get_queue_size() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}
