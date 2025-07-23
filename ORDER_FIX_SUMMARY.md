# 流水线乱序问题修复与目标框筛选功能增强总结

## 问题分析

原始项目的流水线处理机制存在输出乱序问题，主要原因是：

1. **目标检测模块的批处理逻辑破坏了顺序**：
   - 使用了优先级队列`frame_order_queue_`按帧序号排序
   - 先将所有可用帧放入优先级队列，再按批次取出处理
   - 这种设计虽然保证了帧序号的顺序，但破坏了实际到达顺序

2. **批处理与顺序保证的冲突**：
   - 批处理为了提高性能，需要收集多个帧一起处理
   - 但收集过程中的排序操作导致了乱序

## 解决方案

### 1. 乱序问题修复

#### 移除优先级队列机制
- 从`ObjectDetection`类中移除了`frame_order_queue_`优先级队列
- 修改头文件`include/object_detection.h`，删除相关的比较器和队列定义

#### 改进批处理逻辑
修改`src/object_detection.cpp`中的`detection_worker()`方法：
- **保持FIFO顺序**：按照图像到达队列的顺序进行处理
- **优化收集策略**：
  - 阻塞等待第一个图像
  - 使用非阻塞`try_pop()`收集更多图像组成批次
  - 如果队列中没有足够图像，立即处理当前收集到的图像

### 2. 新增目标框筛选阶段

#### 功能描述
在目标检测后增加了一个新的处理阶段：
- **筛选区域**：图像从上开始的 2/7 到 6/7 处
- **筛选条件**：在指定区域内寻找宽度最小的目标框
- **回退策略**：如果指定区域内没有目标框，则在全图范围内寻找
- **结果存储**：将筛选出的目标框存储在`ImageData::filtered_box`中

#### 新增文件
1. **`include/box_filter.h`** - 目标框筛选器头文件
2. **`src/box_filter.cpp`** - 目标框筛选器实现

#### 修改的文件
1. **`include/image_data.h`** - 添加筛选结果字段
2. **`include/pipeline_manager.h`** - 集成新阶段
3. **`src/pipeline_manager.cpp`** - 添加流转逻辑和状态显示
4. **`src/main.cpp`** - 显示筛选结果
5. **`CMakeLists.txt`** - 添加新源文件

### 3. 增强ThreadSafeQueue功能

在`include/thread_safe_queue.h`中添加：
- `try_pop()`非阻塞方法，用于在不阻塞的情况下尝试获取队列元素

### 4. 流水线架构更新

#### 新的流水线结构
```
语义分割 → Mask后处理 → 目标检测 → 目标框筛选 → 最终结果
```

#### 线程配置
- 语义分割: 8 线程
- Mask后处理: 20 线程  
- 目标检测: 8 线程
- 目标框筛选: 4 线程
- 协调线程: 4 个（每个阶段间1个）

#### 实时状态显示
更新了`print_status()`方法，现在显示：
- 语义分割阶段队列状态
- Mask后处理阶段队列状态
- 目标检测阶段队列状态
- **目标框筛选阶段队列状态**（新增）
- 最终结果队列状态

## 技术细节

### 目标框筛选核心算法

```cpp
// 计算筛选区域
int region_top = image_height * 2 / 7;      // 七分之二处
int region_bottom = image_height * 6 / 7;   // 七分之六处

// 首先在指定区域内寻找
ImageData::BoundingBox* min_width_box = find_min_width_box_in_region(
    image->detection_results, region_top, region_bottom);

// 如果指定区域内没有，则在全图范围内寻找
if (min_width_box == nullptr) {
    min_width_box = find_min_width_box_in_region(
        image->detection_results, 0, image_height);
}
```

### 实时状态显示增强

主函数现在显示详细的筛选结果：
- 目标框坐标和尺寸
- 是否在目标区域内
- 目标框的置信度
- 帧序号信息

### 与PipelineManager的协作

PipelineManager现在管理4个阶段的数据流转：
1. `seg_to_mask_thread_func()` - 语义分割到Mask后处理
2. `mask_to_detect_thread_func()` - Mask后处理到目标检测  
3. `detect_to_filter_thread_func()` - 目标检测到目标框筛选（新增）
4. `filter_to_final_thread_func()` - 目标框筛选到最终结果（新增）

## 预期效果

1. **保证输出顺序**：最终输出与输入图片顺序完全一致
2. **保持批处理性能**：在可能的情况下仍然进行批处理以提高性能
3. **智能目标框筛选**：自动筛选出符合条件的最优目标框
4. **实时状态监控**：完整显示所有流水线阶段的工作状态
5. **详细结果展示**：提供丰富的筛选结果信息

## 验证建议

1. 运行测试，观察输出帧序号是否连续且与输入顺序一致
2. 监控各阶段队列状态，确保数据流转正常
3. 验证目标框筛选功能是否按预期工作
4. 检查性能指标，确保新增阶段不会显著影响整体性能
5. 测试在不同目标框分布情况下的筛选效果
