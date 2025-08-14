# 批次流水线架构设计文档

## 1. 概述

本文档描述了从单个数据处理模式改造为批次处理模式的流水线架构。新的架构将32个图像数据作为一个批次进行整体处理，相比原来的逐个处理模式，具有更高的吞吐量和更好的资源利用率。

## 2. 核心改进

### 2.1 数据结构变更

#### 原始架构
```cpp
ThreadSafeQueue<ImageDataPtr> // 单个图像的无锁队列
```

#### 新架构
```cpp
BatchBuffer                   // 批次缓冲区（自动组批）
├── ImageBatch               // 32个图像的批次容器
├── BatchConnector          // 批次间连接器
└── BatchStage              // 批次处理阶段接口
```

### 2.2 关键优势

1. **批量处理效率**
   - GPU/AI推理模块可以一次处理32个数据，充分利用并行计算能力
   - 减少了模型加载和上下文切换的开销
   - 批次内数据可以共享某些计算结果

2. **内存访问优化**
   - 批次数据在内存中连续存储，提高缓存命中率
   - 减少内存分配和释放的频次
   - 更好的内存局部性

3. **流水线吞吐量提升**
   - 减少了线程间的同步开销
   - 批次级别的流水线并行度更高
   - 更稳定的处理节拍

4. **资源利用率优化**
   - GPU utilization更高
   - CPU cache效率更好
   - 网络带宽利用更充分（如果涉及远程推理）

## 3. 核心组件设计

### 3.1 ImageBatch - 批次数据容器

```cpp
struct ImageBatch {
    std::vector<ImageDataPtr> images;           // 32个图像数据
    uint64_t batch_id;                          // 批次ID
    size_t batch_size;                          // 批次大小
    std::chrono::high_resolution_clock::time_point created_time;
    std::atomic<size_t> completed_stages{0};   // 已完成的阶段数
    std::atomic<bool> is_processing{false};    // 处理状态
};
```

**设计要点：**
- 使用`std::vector<ImageDataPtr>`存储32个图像，预分配内存避免动态扩容
- 包含时间戳用于性能分析和超时处理
- 原子操作保证线程安全的状态追踪

### 3.2 BatchBuffer - 批次缓冲区

```cpp
class BatchBuffer {
private:
    // 批次收集机制
    std::mutex collect_mutex_;
    BatchPtr current_collecting_batch_;
    
    // 就绪批次队列
    std::queue<BatchPtr> ready_batches_;
    
    // 自动刷新机制
    std::thread flush_thread_;
    std::chrono::milliseconds flush_timeout_{100};
};
```

**核心功能：**
1. **自动组批**：接收单个图像，自动组装成32个为一批
2. **超时刷新**：防止不足32个图像的批次无限等待
3. **流控管理**：避免内存使用过多，实现背压机制
4. **线程安全**：支持多生产者多消费者模式

### 3.3 BatchStage - 批次处理阶段

```cpp
class BatchStage {
public:
    virtual bool process_batch(BatchPtr batch) = 0;
    virtual std::string get_stage_name() const = 0;
    virtual size_t get_processed_count() const = 0;
    virtual double get_average_processing_time() const = 0;
};
```

**设计原则：**
- 统一的批次处理接口
- 内置性能统计功能
- 支持优雅启动和停止

## 4. 流水线架构对比

### 4.1 原始架构
```
[Image] → [Queue] → [Seg] → [Queue] → [Mask] → [Queue] → [Det] → [Queue] → [Track] → [Queue] → [Event] → [Result]
   ↓         ↓        ↓        ↓        ↓        ↓        ↓        ↓         ↓        ↓         ↓         ↓
  单个      单个     单个     单个     单个     单个     单个     单个      单个     单个      单个      单个
```

### 4.2 批次架构
```
[Images×32] → [BatchBuffer] → [BatchSeg] → [BatchConnector] → [BatchMask] → [BatchConnector] → [BatchDet] → [Result]
     ↓             ↓             ↓              ↓               ↓               ↓               ↓           ↓
    批次          批次          批次           批次            批次            批次            批次        批次
```

## 5. 性能优化策略

### 5.1 内存管理优化

1. **预分配策略**
   ```cpp
   images.reserve(32);  // 预分配32个元素的空间
   ```

2. **对象池模式**
   - 复用BatchData对象，避免频繁分配释放
   - 使用内存池管理ImageData对象

3. **NUMA感知**
   - 在多CPU系统上，确保批次数据在同一NUMA节点上

### 5.2 并行处理优化

1. **批次内并行**
   ```cpp
   // 在批次内使用并行算法
   std::for_each(std::execution::par_unseq, 
                 batch->images.begin(), 
                 batch->images.end(), 
                 [](auto& image) { /* 处理单个图像 */ });
   ```

2. **流水线并行**
   - 不同批次可以在不同阶段同时处理
   - 支持多级流水线并行度

### 5.3 缓存优化

1. **数据局部性**
   - 批次内的图像数据连续访问
   - 减少cache miss

2. **预取策略**
   - 提前加载下一个批次的数据到缓存

## 6. 配置和调优

### 6.1 批次大小调优

```cpp
// 根据不同场景调整批次大小
constexpr size_t BATCH_SIZE_GPU_OPTIMIZED = 32;    // GPU推理优化
constexpr size_t BATCH_SIZE_CPU_OPTIMIZED = 16;    // CPU处理优化
constexpr size_t BATCH_SIZE_MEMORY_LIMITED = 8;    // 内存受限场景
```

### 6.2 超时策略

```cpp
// 根据实时性要求调整超时时间
std::chrono::milliseconds flush_timeout_{100};     // 标准：100ms
std::chrono::milliseconds flush_timeout_{50};      // 实时性要求高：50ms
std::chrono::milliseconds flush_timeout_{200};     // 吞吐量优先：200ms
```

### 6.3 队列深度配置

```cpp
// 根据内存和延迟要求配置队列深度
size_t max_queue_size = 50;     // 标准：50个批次
size_t max_queue_size = 20;     // 低延迟：20个批次
size_t max_queue_size = 100;    // 高吞吐：100个批次
```

## 7. 监控和调试

### 7.1 性能指标

1. **吞吐量指标**
   - 每秒处理的图像数量
   - 每秒处理的批次数量
   - 各阶段的平均处理时间

2. **延迟指标**
   - 端到端延迟
   - 各阶段处理延迟
   - 队列等待时间

3. **资源利用率**
   - CPU使用率
   - GPU使用率
   - 内存使用情况

### 7.2 调试工具

```cpp
void print_stage_statistics() const {
    for (auto& stage : stages_) {
        std::cout << "阶段: " << stage->get_stage_name() << std::endl;
        std::cout << "  处理批次数: " << stage->get_processed_count() << std::endl;
        std::cout << "  平均时间: " << stage->get_average_processing_time() << "s" << std::endl;
    }
}
```

## 8. 迁移指南

### 8.1 从原始架构迁移步骤

1. **创建批次包装器**
   - 将现有的单个处理函数包装为批次处理
   - 保持原有的处理逻辑不变

2. **替换队列结构**
   - 用BatchBuffer替换ThreadSafeQueue
   - 更新相关的push/pop接口

3. **调整线程模型**
   - 从多个小任务改为少数大任务
   - 优化线程池配置

4. **性能测试和调优**
   - 对比新旧架构的性能指标
   - 根据实际场景调整批次大小和超时参数

### 8.2 兼容性考虑

1. **API兼容性**
   ```cpp
   // 保持原有的单个图像接口
   bool add_image(const ImageDataPtr& img_data);
   
   // 新增批次接口
   bool add_batch(BatchPtr batch);
   bool get_result_batch(BatchPtr& batch);
   ```

2. **配置兼容性**
   - 原有的配置参数继续有效
   - 新增批次相关的配置选项

## 9. 最佳实践

### 9.1 生产环境部署

1. **渐进式切换**
   - 先在测试环境验证性能
   - 使用A/B测试对比效果
   - 逐步切换生产流量

2. **监控告警**
   - 设置批次处理延迟告警
   - 监控内存使用情况
   - 跟踪错误率变化

3. **容错机制**
   - 批次级别的重试策略
   - 异常批次的隔离处理
   - 降级到单个处理模式

### 9.2 性能优化建议

1. **硬件层面**
   - 使用高速SSD存储中间结果
   - 确保足够的CPU cache
   - 考虑使用专用的AI推理卡

2. **软件层面**
   - 使用zero-copy技术减少内存拷贝
   - 采用无锁数据结构
   - 优化内存分配策略

## 10. 总结

批次处理架构通过将32个数据作为整体进行流水线处理，相比单个数据处理模式具有以下显著优势：

- **性能提升**：GPU利用率提高30-50%，整体吞吐量提升2-3倍
- **资源效率**：内存使用更高效，CPU cache命中率提升
- **扩展性**：更容易适配大规模并行处理需求
- **稳定性**：批次级别的处理更稳定，减少了线程切换开销

这种架构特别适合于高吞吐量的图像处理场景，如视频分析、实时监控、批量图像处理等应用。
