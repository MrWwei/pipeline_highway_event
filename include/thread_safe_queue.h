#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class ThreadSafeQueue {
private:
  mutable std::mutex mtx_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  std::queue<T> q_;
  const size_t cap_;
  std::atomic<bool> shutdown_{false};

public:
  explicit ThreadSafeQueue(size_t max_size = 100) : cap_(max_size) {}

  /*-------------------------------------------------
   * 阻塞 push（拷贝）
   *------------------------------------------------*/
  void push(const T &value) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_full_.wait(lk, [this] { return q_.size() < cap_ || shutdown_.load(); });
    
    if (shutdown_.load()) {
      return; // 如果已关闭，直接返回
    }

    try {
      q_.push(value);
    } catch (...) {
      lk.unlock();
      cv_not_empty_.notify_all(); // 异常也唤醒
      cv_not_full_.notify_all();
      throw;
    }
    lk.unlock();
    cv_not_empty_.notify_one();
  }

  /*-------------------------------------------------
   * 阻塞 push（移动）
   *------------------------------------------------*/
  void push(T &&value) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_full_.wait(lk, [this] { return q_.size() < cap_ || shutdown_.load(); });
    
    if (shutdown_.load()) {
      return; // 如果已关闭，直接返回
    }

    try {
      q_.push(std::move(value));
    } catch (...) {
      lk.unlock();
      cv_not_empty_.notify_all();
      cv_not_full_.notify_all();
      throw;
    }
    lk.unlock();
    cv_not_empty_.notify_one();
  }

  /*-------------------------------------------------
   * 完美转发 emplace（可构造任意参数）
   *------------------------------------------------*/
  template <typename... Args> void emplace(Args &&... args) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_full_.wait(lk, [this] { return q_.size() < cap_ || shutdown_.load(); });
    
    if (shutdown_.load()) {
      return; // 如果已关闭，直接返回
    }

    try {
      q_.emplace(std::forward<Args>(args)...);
    } catch (...) {
      lk.unlock();
      cv_not_empty_.notify_all();
      cv_not_full_.notify_all();
      throw;
    }
    lk.unlock();
    cv_not_empty_.notify_one();
  }

  /*-------------------------------------------------
   * 阻塞 pop
   *------------------------------------------------*/
  bool wait_and_pop(T &out) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_empty_.wait(lk, [this] { return !q_.empty() || shutdown_.load(); });
    
    if (shutdown_.load() && q_.empty()) {
      return false; // 已关闭且队列为空，返回false表示失败
    }

    try {
      out = std::move(q_.front());
      q_.pop();
    } catch (...) {
      lk.unlock();
      cv_not_empty_.notify_all();
      cv_not_full_.notify_all();
      throw;
    }
    lk.unlock();
    cv_not_full_.notify_one();
    return true; // 成功获取数据
  }

  /*-------------------------------------------------
   * 非阻塞 try_pop 方法
   *------------------------------------------------*/
  bool try_pop(T &out) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (q_.empty()) {
      return false;
    }
    
    try {
      out = std::move(q_.front());
      q_.pop();
    } catch (...) {
      cv_not_empty_.notify_all();
      cv_not_full_.notify_all();
      throw;
    }
    cv_not_full_.notify_one();
    return true;
  }

  /*-------------------------------------------------
   * 只读查询
   *------------------------------------------------*/
  bool empty() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return q_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return q_.size();
  }

  bool full() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return q_.size() >= cap_;
  }

  size_t max_size() const { return cap_; }

  size_t remaining_capacity() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return cap_ - q_.size();
  }

  /*-------------------------------------------------
   * 清空队列
   *------------------------------------------------*/
  void clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    while (!q_.empty()) {
      q_.pop();
    }
    cv_not_full_.notify_all();
  }

  /*-------------------------------------------------
   * 关闭队列，唤醒所有等待的线程
   *------------------------------------------------*/
  void shutdown() {
    std::lock_guard<std::mutex> lk(mtx_);
    shutdown_.store(true);
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }

  /*-------------------------------------------------
   * 重置队列，允许重新使用
   *------------------------------------------------*/
  void reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    shutdown_.store(false);
    while (!q_.empty()) {
      q_.pop();
    }
  }

  /*-------------------------------------------------
   * 检查是否已关闭
   *------------------------------------------------*/
  bool is_shutdown() const {
    return shutdown_.load();
  }
};