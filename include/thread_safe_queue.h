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

public:
  explicit ThreadSafeQueue(size_t max_size = 100) : cap_(max_size) {}

  /*-------------------------------------------------
   * 阻塞 push（拷贝）
   *------------------------------------------------*/
  void push(const T &value) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_full_.wait(lk, [this] { return q_.size() < cap_; });

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
    cv_not_full_.wait(lk, [this] { return q_.size() < cap_; });

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
    cv_not_full_.wait(lk, [this] { return q_.size() < cap_; });

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
  void wait_and_pop(T &out) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_empty_.wait(lk, [this] { return !q_.empty(); });

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
};