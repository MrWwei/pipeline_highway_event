# 高速公路事件检测器 - 纯净接口

本项目提供了一个简洁的高速公路事件检测接口，包含三个核心功能：

## 🎯 核心功能

1. **初始化** - 流水线初始化
2. **添加数据** - 向流水线中添加数据并返回帧序号
3. **获取结果** - 获取指定帧序号的处理结果

## 🚀 快速开始

### 基本使用流程

```cpp
#include "highway_event.h"

int main() {
    // 1. 创建检测器实例
    HighwayEventDetector detector;
    
    // 2. 配置参数
    HighwayEventConfig config;
    config.semantic_threads = 2;
    config.detection_threads = 2;
    config.enable_debug_log = true;
    
    // 3. 初始化流水线
    if (!detector.initialize(config)) {
        std::cerr << "初始化失败" << std::endl;
        return -1;
    }
    
    // 4. 添加图像数据，获取帧序号
    cv::Mat image = cv::imread("test.jpg");
    int64_t frame_id = detector.add_frame(image);
    
    // 5. 获取处理结果
    ProcessResult result = detector.get_result_with_timeout(frame_id, 30000);
    
    if (result.status == ResultStatus::SUCCESS) {
        std::cout << "检测到 " << result.detections.size() << " 个目标" << std::endl;
    }
    
    return 0;
}
```

## 📋 接口说明

### HighwayEventDetector 类

#### 核心方法

- `initialize(config)` - 初始化流水线
- `add_frame(image)` - 添加图像，返回帧序号
- `get_result(frame_id)` - 获取结果
- `get_result_with_timeout(frame_id, timeout_ms)` - 获取结果（带超时）

#### 状态查询

- `is_initialized()` - 是否已初始化
- `is_running()` - 是否正在运行
- `get_config()` - 获取当前配置
- `get_pipeline_status()` - 获取流水线状态信息

### 配置参数 (HighwayEventConfig)

#### 线程配置
- `semantic_threads` - 语义分割线程数 (默认: 2)
- `mask_threads` - Mask后处理线程数 (默认: 1)
- `detection_threads` - 目标检测线程数 (默认: 2)
- `tracking_threads` - 目标跟踪线程数 (默认: 1)
- `filter_threads` - 目标框筛选线程数 (默认: 1)

#### 模型配置
- `seg_model_path` - 语义分割模型路径 (默认: "seg_model")
- `det_model_path` - 目标检测模型路径 (默认: "car_detect.onnx")

#### 检测配置
- `det_img_size` - 输入图像尺寸 (默认: 640)
- `det_conf_thresh` - 置信度阈值 (默认: 0.25)
- `det_iou_thresh` - NMS IoU阈值 (默认: 0.2)

#### 调试配置
- `enable_debug_log` - 启用调试日志 (默认: false)
- `seg_enable_show` - 启用分割结果可视化 (默认: false)

### 处理结果 (ProcessResult)

#### 状态信息
- `status` - 处理状态 (SUCCESS, PENDING, TIMEOUT, NOT_FOUND, ERROR)
- `frame_id` - 帧序号

#### 检测结果
- `detections` - 所有检测到的目标列表
- `filtered_box` - 筛选出的最佳目标框
- `has_filtered_box` - 是否有筛选结果
- `mask` - 语义分割掩码
- `roi` - 感兴趣区域

### 检测框 (DetectionBox)

- `left, top, right, bottom` - 边界框坐标
- `confidence` - 置信度
- `class_id` - 类别ID
- `track_id` - 跟踪ID
- `status` - 目标状态 (NORMAL, PARKING_LANE, HIGHWAY_JAM等)

## 📁 文件结构

```
include/
  highway_event.h          # 纯净接口头文件
src/
  highway_event.cpp        # 接口实现
simple_usage_example.cpp  # 简单使用示例
example_usage.cpp         # 详细使用示例
```

## 🔧 编译和运行

### 编译

```bash
# 使用现有的构建脚本
./build.sh

# 或者手动编译示例
g++ -std=c++17 simple_usage_example.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lhighway_event_sdk_v1.0 -o simple_example
```

### 运行示例

```bash
# 确保有测试图片
cp DJI_20250501091406_0001_V_frame_4460.jpg test.jpg

# 运行简单示例
./simple_example
```

## 🎯 设计特点

### 1. 简洁的接口
- 只有三个核心方法：初始化、添加数据、获取结果
- 隐藏复杂的内部实现细节
- 使用Pimpl模式确保ABI稳定性

### 2. 异步处理
- 添加数据立即返回帧序号，不阻塞
- 流水线在后台异步处理图像
- 可以并发添加多帧数据

### 3. 按序号获取结果
- 通过帧序号精确获取对应的处理结果
- 支持超时机制，避免无限等待
- 自动清理已获取的结果，防止内存泄漏

### 4. 完整的错误处理
- 详细的状态码和错误信息
- 优雅的资源清理
- 线程安全的设计

## 🔍 状态监控

使用 `get_pipeline_status()` 可以获取详细的流水线状态：

```cpp
std::cout << detector.get_pipeline_status() << std::endl;
```

输出示例：
```
=== 流水线状态 ===
语义分割队列: 0 帧
Mask后处理队列: 0 帧
目标检测队列: 0 帧
目标跟踪队列: 0 帧
目标框筛选队列: 0 帧
最终结果队列: 0 帧
下一帧ID: 1
缓存结果数量: 0 帧
```

## ⚠️ 注意事项

1. **模型文件** - 确保 `seg_model` 和 `car_detect.onnx` 文件存在
2. **内存管理** - 及时获取处理结果，避免结果积累
3. **线程安全** - 接口是线程安全的，可以在多线程环境中使用
4. **资源清理** - 析构函数会自动清理资源，也可以手动调用 `stop()`

## 🎉 优势

- **简单易用** - 只需要关心三个核心方法
- **高性能** - 多线程流水线处理，充分利用硬件资源
- **灵活配置** - 丰富的配置选项，适应不同场景
- **健壮性** - 完善的错误处理和资源管理
- **可扩展** - 基于现有的模块化架构，易于扩展功能
