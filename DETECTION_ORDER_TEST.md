# 目标检测输入顺序验证

## 添加的调试日志

为了验证目标检测模块是否按正确顺序接收和处理图像，我在以下位置添加了详细的日志：

### 1. 目标检测模块 (`src/object_detection.cpp`)

#### 接收图像日志
```cpp
void ObjectDetection::process_image(ImageDataPtr image, int thread_id) {
  std::cout << "📥 目标检测接收图像: 帧 " << image->frame_idx << " (线程 " << thread_id << ")" << std::endl;
  detection_queue_->push(image);
}
```

#### 批处理开始日志
```cpp
void ObjectDetection::detection_worker() {
  // 当开始处理一个新批次时
  std::cout << "🔄 目标检测开始处理批次，首帧: " << first_img->frame_idx << std::endl;
  
  // 显示批次中所有帧的序号
  std::cout << "📊 目标检测批次处理 [批次大小: " << batch_images.size() << "] 帧序号: ";
  for (const auto& img : batch_images) {
    std::cout << img->frame_idx << " ";
  }
  std::cout << std::endl;
}
```

#### 批处理完成日志
```cpp
// 批量处理完成
std::cout << "✅ 批量目标检测完成，耗时: " << duration.count() << "ms, 完成帧: ";
for (const auto& img : batch_images) {
  std::cout << img->frame_idx << " ";
}
std::cout << std::endl;

// 单图处理完成
std::cout << "✅ 目标检测完成 (单图)，帧 " << image->frame_idx 
          << "，耗时: " << duration.count() << "ms" << std::endl;
```

### 2. Mask后处理模块 (`src/mask_postprocess.cpp`)

```cpp
void MaskPostProcess::on_processing_start(ImageDataPtr image, int thread_id) {
  std::cout << "🔍 Mask后处理准备开始 (线程 " << thread_id << ", 帧 " << image->frame_idx << ")" << std::endl;
}

void MaskPostProcess::on_processing_complete(ImageDataPtr image, int thread_id) {
  std::cout << "🔍 Mask后处理完成 (线程 " << thread_id << ", 帧 " << image->frame_idx << ")" << std::endl;
}
```

### 3. 流水线管理器 (`src/pipeline_manager.cpp`)

#### Mask后处理到目标检测的数据流转
```cpp
void PipelineManager::mask_to_detect_thread_func() {
  while (mask_postprocess_->get_processed_image(mask_result)) {
    std::cout << "🔄 PipelineManager: Mask后处理 → 目标检测, 帧 " << mask_result->frame_idx << std::endl;
    object_det_->add_image(mask_result);
  }
}
```

#### 目标检测到目标框筛选的数据流转
```cpp
void PipelineManager::detect_to_filter_thread_func() {
  while (object_det_->get_processed_image(detect_result)) {
    std::cout << "🔄 PipelineManager: 目标检测 → 目标框筛选, 帧 " << detect_result->frame_idx << std::endl;
    box_filter_->add_image(detect_result);
  }
}
```

## 如何验证输入顺序

运行程序后，观察日志输出中的帧序号模式：

### 1. 正确的顺序应该显示：
```
📥 目标检测接收图像: 帧 0 (线程 X)
📥 目标检测接收图像: 帧 1 (线程 Y)
📥 目标检测接收图像: 帧 2 (线程 Z)
...

🔄 目标检测开始处理批次，首帧: 0
📊 目标检测批次处理 [批次大小: 8] 帧序号: 0 1 2 3 4 5 6 7
✅ 批量目标检测完成，耗时: XXXms, 完成帧: 0 1 2 3 4 5 6 7

🔄 目标检测开始处理批次，首帧: 8
📊 目标检测批次处理 [批次大小: 8] 帧序号: 8 9 10 11 12 13 14 15
...
```

### 2. 需要关注的问题：

#### 🚨 乱序接收
如果看到类似这样的日志：
```
📥 目标检测接收图像: 帧 0
📥 目标检测接收图像: 帧 2  ← 跳过了帧1
📥 目标检测接收图像: 帧 1  ← 帧1延后到达
```

#### 🚨 批次内乱序
如果看到类似这样的日志：
```
📊 目标检测批次处理 [批次大小: 8] 帧序号: 0 2 1 4 3 6 5 7  ← 批次内顺序错乱
```

#### 🚨 跨批次乱序
如果看到类似这样的日志：
```
✅ 批量目标检测完成，耗时: XXXms, 完成帧: 8 9 10 11 12 13 14 15
✅ 批量目标检测完成，耗时: XXXms, 完成帧: 0 1 2 3 4 5 6 7  ← 第二批比第一批先完成
```

## 预期行为

基于我们的修复：

1. **接收顺序**：目标检测应该按帧序号顺序接收图像 (0, 1, 2, 3, ...)
2. **批次内顺序**：每个批次内的帧序号应该是连续的
3. **处理顺序**：批次应该按接收顺序处理，不会出现后面的批次先完成的情况
4. **FIFO保证**：由于使用ThreadSafeQueue的FIFO特性，顺序应该得到保证

## 测试建议

1. 运行程序并观察前100帧的日志
2. 重点检查：
   - 📥 接收日志的帧序号是否连续
   - 📊 批次处理日志的帧序号是否连续且有序
   - ✅ 完成日志的帧序号是否按预期顺序
3. 如果发现乱序，记录具体的日志模式以便进一步分析

这些日志将帮助我们确认目标检测模块是否正确地按顺序接收和处理图像。
