#pragma once
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>

/**
 * 高性能无锁环形队列 (Lock-Free Ring Buffer Queue)
 * 基于原子操作实现的多生产者多消费者队列
 * 相比传统的基于锁的队列，具有更低的延迟和更高的吞吐量
 */
template <typename T> 
class ThreadSafeQueue {
private:
  // 环形缓冲区节点
  struct Node {
    std::atomic<T*> data{nullptr};
    std::atomic<size_t> sequence{0};
    
    Node() = default;
    ~Node() {
      T* ptr = data.load();
      if (ptr) {
        delete ptr;
      }
    }
  };

  // 缓冲区大小，必须是2的幂次方以便使用位掩码优化
  size_t capacity_;
  size_t mask_;
  
  // 环形缓冲区
  std::unique_ptr<Node[]> buffer_;
  
  // 生产者和消费者位置指针
  alignas(64) std::atomic<size_t> write_pos_{0};  // 缓存行对齐避免伪共享
  alignas(64) std::atomic<size_t> read_pos_{0};   // 缓存行对齐避免伪共享
  
  // 关闭标志
  std::atomic<bool> shutdown_{false};
  
  // 将输入大小向上舍入到最近的2的幂次方
  static size_t round_up_to_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
  }

public:
  explicit ThreadSafeQueue(size_t max_size = 100) 
    : capacity_(round_up_to_power_of_2(max_size))
    , mask_(capacity_ - 1)
    , buffer_(std::make_unique<Node[]>(capacity_)) {
    
    // 初始化序列号
    for (size_t i = 0; i < capacity_; ++i) {
      buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }
  }

  /*-------------------------------------------------
   * 阻塞 push（拷贝）- 高性能无锁实现
   *------------------------------------------------*/
  void push(const T &value) {
    T* new_item = new T(value);
    if (!try_push_impl(new_item)) {
      // 如果推送失败，进入退避重试机制
      delete new_item;
      if (shutdown_.load(std::memory_order_acquire)) {
        return;
      }
      
      // 指数退避重试
      size_t backoff = 1;
      while (!shutdown_.load(std::memory_order_acquire)) {
        new_item = new T(value);
        if (try_push_impl(new_item)) {
          return;
        }
        delete new_item;
        
        // 指数退避，避免忙等
        std::this_thread::sleep_for(std::chrono::nanoseconds(backoff));
        backoff = std::min(backoff * 2, static_cast<size_t>(1000)); // 最大1微秒
      }
    }
  }

  /*-------------------------------------------------
   * 阻塞 push（移动）- 高性能无锁实现
   *------------------------------------------------*/
  void push(T &&value) {
    T* new_item = new T(std::move(value));
    if (!try_push_impl(new_item)) {
      delete new_item;
      if (shutdown_.load(std::memory_order_acquire)) {
        return;
      }
      
      // 指数退避重试
      size_t backoff = 1;
      while (!shutdown_.load(std::memory_order_acquire)) {
        new_item = new T(std::move(value));
        if (try_push_impl(new_item)) {
          return;
        }
        delete new_item;
        
        std::this_thread::sleep_for(std::chrono::nanoseconds(backoff));
        backoff = std::min(backoff * 2, static_cast<size_t>(1000));
      }
    }
  }

  /*-------------------------------------------------
   * 完美转发 emplace - 高性能无锁实现
   *------------------------------------------------*/
  template <typename... Args> 
  void emplace(Args &&... args) {
    T* new_item = new T(std::forward<Args>(args)...);
    if (!try_push_impl(new_item)) {
      delete new_item;
      if (shutdown_.load(std::memory_order_acquire)) {
        return;
      }
      
      // 指数退避重试
      size_t backoff = 1;
      while (!shutdown_.load(std::memory_order_acquire)) {
        new_item = new T(std::forward<Args>(args)...);
        if (try_push_impl(new_item)) {
          return;
        }
        delete new_item;
        
        std::this_thread::sleep_for(std::chrono::nanoseconds(backoff));
        backoff = std::min(backoff * 2, static_cast<size_t>(1000));
      }
    }
  }

  /*-------------------------------------------------
   * 阻塞 pop - 高性能无锁实现
   *------------------------------------------------*/
  bool wait_and_pop(T &out) {
    T* item = nullptr;
    
    // 先尝试非阻塞获取
    if (try_pop_impl(item)) {
      out = std::move(*item);
      delete item;
      return true;
    }
    
    // 如果队列为空且已关闭，直接返回
    if (shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    
    // 进入等待模式，使用指数退避
    size_t backoff = 1;
    while (!shutdown_.load(std::memory_order_acquire)) {
      if (try_pop_impl(item)) {
        out = std::move(*item);
        delete item;
        return true;
      }
      
      // 指数退避等待
      std::this_thread::sleep_for(std::chrono::nanoseconds(backoff));
      backoff = std::min(backoff * 2, static_cast<size_t>(10000)); // 最大10微秒
    }
    
    return false; // 已关闭且队列为空
  }

  /*-------------------------------------------------
   * 非阻塞 try_pop 方法 - 高性能无锁实现
   *------------------------------------------------*/
  bool try_pop(T &out) {
    T* item = nullptr;
    if (try_pop_impl(item)) {
      out = std::move(*item);
      delete item;
      return true;
    }
    return false;
  }

  /*-------------------------------------------------
   * 只读查询 - 无锁实现
   *------------------------------------------------*/
  bool empty() const {
    size_t read_pos = read_pos_.load(std::memory_order_acquire);
    size_t write_pos = write_pos_.load(std::memory_order_acquire);
    return read_pos == write_pos;
  }

  size_t size() const {
    size_t write_pos = write_pos_.load(std::memory_order_acquire);
    size_t read_pos = read_pos_.load(std::memory_order_acquire);
    return write_pos - read_pos;
  }

  bool full() const {
    size_t write_pos = write_pos_.load(std::memory_order_acquire);
    size_t read_pos = read_pos_.load(std::memory_order_acquire);
    return (write_pos - read_pos) >= capacity_;
  }

  size_t max_size() const { 
    return capacity_; 
  }

  size_t remaining_capacity() const {
    size_t write_pos = write_pos_.load(std::memory_order_acquire);
    size_t read_pos = read_pos_.load(std::memory_order_acquire);
    return capacity_ - (write_pos - read_pos);
  }

  /*-------------------------------------------------
   * 清空队列 - 无锁实现
   *------------------------------------------------*/
  void clear() {
    T dummy;
    while (try_pop(dummy)) {
      // 继续清空直到队列为空
    }
  }

  /*-------------------------------------------------
   * 关闭队列，停止所有操作
   *------------------------------------------------*/
  void shutdown() {
    shutdown_.store(true, std::memory_order_release);
  }

  /*-------------------------------------------------
   * 重置队列，允许重新使用
   *------------------------------------------------*/
  void reset() {
    shutdown_.store(false, std::memory_order_release);
    clear();
    // 重置位置指针
    read_pos_.store(0, std::memory_order_release);
    write_pos_.store(0, std::memory_order_release);
    
    // 重新初始化序列号
    for (size_t i = 0; i < capacity_; ++i) {
      buffer_[i].sequence.store(i, std::memory_order_relaxed);
      T* old_data = buffer_[i].data.exchange(nullptr, std::memory_order_acq_rel);
      if (old_data) {
        delete old_data;
      }
    }
  }

  /*-------------------------------------------------
   * 检查是否已关闭
   *------------------------------------------------*/
  bool is_shutdown() const {
    return shutdown_.load(std::memory_order_acquire);
  }

private:
  /*-------------------------------------------------
   * 内部无锁推送实现
   *------------------------------------------------*/
  bool try_push_impl(T* item) {
    size_t current_write = write_pos_.load(std::memory_order_relaxed);
    
    for (;;) {
      Node& node = buffer_[current_write & mask_];
      size_t node_seq = node.sequence.load(std::memory_order_acquire);
      
      if (node_seq == current_write) {
        // 找到可用节点，尝试写入
        if (write_pos_.compare_exchange_weak(current_write, current_write + 1, 
                                           std::memory_order_relaxed)) {
          // 成功获取写入位置，设置数据
          node.data.store(item, std::memory_order_release);
          node.sequence.store(current_write + 1, std::memory_order_release);
          return true;
        }
      } else if (node_seq < current_write) {
        // 节点还被消费者占用，队列可能满了
        size_t read_pos = read_pos_.load(std::memory_order_acquire);
        if (current_write - read_pos >= capacity_) {
          return false; // 队列满
        }
        // 重新读取写位置
        current_write = write_pos_.load(std::memory_order_relaxed);
      } else {
        // 其他生产者已经写入了这个位置，重新读取写位置
        current_write = write_pos_.load(std::memory_order_relaxed);
      }
    }
  }

  /*-------------------------------------------------
   * 内部无锁弹出实现
   *------------------------------------------------*/
  bool try_pop_impl(T*& item) {
    size_t current_read = read_pos_.load(std::memory_order_relaxed);
    
    for (;;) {
      Node& node = buffer_[current_read & mask_];
      size_t node_seq = node.sequence.load(std::memory_order_acquire);
      
      if (node_seq == current_read + 1) {
        // 找到可读节点，尝试读取
        if (read_pos_.compare_exchange_weak(current_read, current_read + 1,
                                          std::memory_order_relaxed)) {
          // 成功获取读取位置，获取数据
          item = node.data.load(std::memory_order_acquire);
          node.data.store(nullptr, std::memory_order_release);
          node.sequence.store(current_read + capacity_, std::memory_order_release);
          return item != nullptr;
        }
      } else if (node_seq < current_read + 1) {
        // 队列为空
        return false;
      } else {
        // 其他消费者已经读取了这个位置，重新读取读位置
        current_read = read_pos_.load(std::memory_order_relaxed);
      }
    }
  }
};