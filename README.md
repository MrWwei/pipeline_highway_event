# 高速公路事件检测流水线系统

## 项目概述

这是一个基于C++的高性能图像处理流水线系统，专为高速公路场景的事件检测而设计。系统集成了语义分割和目标检测两个核心功能，采用非阻塞并行处理架构，能够高效处理实时图像流。

## 系统架构

### 核心组件

1. **ImageData** - 图像数据结构
   - 存储原始图像数据
   - 管理语义分割和目标检测结果
   - 跟踪处理状态

2. **ThreadSafeQueue** - 线程安全队列
   - 支持多线程并发访问
   - 提供阻塞和非阻塞操作
   - 用于模块间数据传递

3. **SemanticSegmentation** - 语义分割处理器
   - 独立工作线程
   - 模拟2秒处理时间
   - 生成像素级分类结果

4. **ObjectDetection** - 目标检测处理器
   - 独立工作线程
   - 模拟1.5秒处理时间
   - 生成边界框和置信度

5. **PipelineManager** - 流水线管理器
   - 协调各个处理模块
   - 管理流水线生命周期
   - 合并处理结果

### 流水线设计特点

- **非阻塞处理**: 语义分割和目标检测并行执行
- **线程安全**: 所有数据传递都通过线程安全队列
- **智能协调**: 自动合并两个模块的处理结果
- **实时监控**: 提供处理状态和队列信息
- **资源管理**: 自动管理线程生命周期

## 编译和运行

### 环境要求

- C++17 或更高版本
- CMake 3.16 或更高版本
- 支持 pthread 的编译器

### 编译步骤

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake ..

# 编译
make

# 运行
./PipelineHighwayEvent
```

## 使用示例

程序启动后会自动演示流水线处理流程：

1. 初始化各个处理模块
2. 启动工作线程
3. 添加5个测试图像
4. 实时显示处理进度
5. 输出最终结果
6. 优雅关闭系统

### 输出示例

```
=== 高速公路事件检测流水线系统 ===
🚗 整合语义分割和目标检测的非阻塞流水线处理

🔍 语义分割模块初始化完成
🔍 目标检测模块初始化完成
🏗️  流水线管理器初始化完成
🚀 语义分割处理线程启动
🚀 目标检测处理线程启动
🚀 流水线启动完成

📸 开始添加测试图像到流水线...
✓ 图像数据加载完成: highway_scene_001.jpg (640x480x3)
📥 图像添加到语义分割队列: highway_scene_001.jpg
📥 图像添加到目标检测队列: highway_scene_001.jpg
📤 图像添加到流水线: highway_scene_001.jpg

🔄 开始语义分割处理: highway_scene_001.jpg
🔄 开始目标检测处理: highway_scene_001.jpg
...
```

## 扩展说明

### 替换为真实算法

要将模拟算法替换为真实的深度学习模型：

1. **语义分割**: 集成如 DeepLabV3+、U-Net 等模型
2. **目标检测**: 集成如 YOLO、SSD、Faster R-CNN 等模型
3. **依赖库**: 添加 OpenCV、TensorFlow/PyTorch C++ API
4. **GPU加速**: 支持 CUDA/OpenCL 加速

### 性能优化

- 使用内存池减少内存分配开销
- 实现 GPU 流水线处理
- 添加批处理支持
- 优化数据传输格式

### 功能扩展

- 支持视频流输入
- 添加结果后处理
- 集成事件检测逻辑
- 添加可视化输出

## 技术特性

- ✅ 多线程并行处理
- ✅ 线程安全数据结构
- ✅ RAII 资源管理
- ✅ 智能指针内存管理
- ✅ 现代C++特性
- ✅ 可扩展架构设计

## 许可证

MIT License
