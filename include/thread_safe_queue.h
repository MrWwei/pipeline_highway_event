#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

/**
 * 线程安全的队列，用于在不同线程间传递数据
 * 支持最大容量限制，防止内存无限增长
 */
template <typename T> class ThreadSafeQueue {
private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable not_full_condition_;
  size_t max_size_;

public:
  // 构造函数，可指定最大容量，默认为100
  explicit ThreadSafeQueue(size_t max_size = 100) : max_size_(max_size) {}
  // 添加元素到队列（阻塞版本，如果队列满则等待）
  void push(const T &item) {
    std::unique_lock<std::mutex> lock(mutex_);
    // 等待直到队列不满
    while (queue_.size() >= max_size_) {
      not_full_condition_.wait(lock);
    }
    queue_.push(item);
    condition_.notify_one();
  }

  // 添加元素到队列（移动语义，阻塞版本）
  void push(T &&item) {
    std::unique_lock<std::mutex> lock(mutex_);
    // 等待直到队列不满
    while (queue_.size() >= max_size_) {
      not_full_condition_.wait(lock);
    }
    queue_.push(std::move(item));
    condition_.notify_one();
  }

  // 非阻塞添加元素到队列，如果队列满则返回false
  bool try_push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= max_size_) {
      return false;
    }
    queue_.push(item);
    condition_.notify_one();
    return true;
  }

  // 非阻塞添加元素到队列（移动语义），如果队列满则返回false
  bool try_push(T &&item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= max_size_) {
      return false;
    }
    queue_.push(std::move(item));
    condition_.notify_one();
    return true;
  }

  // 非阻塞方式获取元素
  bool try_pop(T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = queue_.front();
    queue_.pop();
    not_full_condition_.notify_one(); // 通知可能等待的生产者
    return true;
  }

  // 阻塞方式获取元素
  void wait_and_pop(T &item) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty()) {
      condition_.wait(lock);
    }
    item = queue_.front();
    queue_.pop();
    not_full_condition_.notify_one(); // 通知可能等待的生产者
  }

  // 检查队列是否为空
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  // 获取队列大小
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  // 检查队列是否已满
  bool full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() >= max_size_;
  }

  // 获取最大容量
  size_t max_size() const { return max_size_; }

  // 获取剩余容量
  size_t remaining_capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_size_ - queue_.size();
  }
};
