#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>

/**
 * 高速公路事件检测专用线程池类
 * 支持任意类型的任务提交和执行
 * 最大任务队列大小为64
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    // 提交任务到线程池
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    // 获取线程池信息
    size_t get_thread_count() const { return workers_.size(); }
    size_t get_queue_size() const;
    bool is_running() const { return running_.load(); }
    
    // 停止线程池
    void stop();

private:
    // 工作线程
    std::vector<std::thread> workers_;
    
    // 任务队列
    std::queue<std::function<void()>> tasks_;
    
    // 同步原语
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    
    // 控制标志
    std::atomic<bool> running_;
    
    // 最大队列大小
    static constexpr size_t MAX_QUEUE_SIZE = 64;
};

// 模板函数实现
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // 检查线程池是否已停止
        if (!running_.load()) {
            throw std::runtime_error("ThreadPool已停止，无法提交新任务");
        }

        // 检查队列是否已满
        if (tasks_.size() >= MAX_QUEUE_SIZE) {
            throw std::runtime_error("ThreadPool任务队列已满");
        }

        tasks_.emplace([task](){ (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}
