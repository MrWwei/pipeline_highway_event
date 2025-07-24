# Highway Event Detector Demo

这是一个全面的高速公路事件检测系统 (`HighwayEventDetector`) 接口测试程序，包含多种测试场景来验证系统的功能和性能。

## 🚀 快速开始

### 1. 构建项目

```bash
# 自动构建和运行 (推荐)
chmod +x run_demo.sh
./run_demo.sh

# 或者手动构建
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. 运行测试

```bash
# 在build目录下运行
./HighwayEventDemo [测试类型] [可选参数]
```

## 📋 测试类型

### 1. API功能完整性测试 (`api`)
测试所有API接口的基本功能：
- 初始化和启动流水线
- 不同方式添加帧 (拷贝、移动、带超时)
- 不同方式获取结果 (阻塞、非阻塞、带超时)
- 状态查询和结果清理
- 正确停止流水线

```bash
./HighwayEventDemo api
```

### 2. 单张图片处理测试 (`single`)
测试基础的单张图片处理功能：
- 处理测试图片 `test.jpg`
- 测量处理延迟
- 分析检测结果

```bash
./HighwayEventDemo single
```

**注意**: 需要在项目根目录放置名为 `test.jpg` 的测试图片。

### 3. 批量处理测试 (`batch`)
测试批量图片处理的性能和稳定性：
- 处理多张不同尺寸的模拟图片
- 测量吞吐量和成功率
- 验证内存管理

```bash
./HighwayEventDemo batch
```

### 4. 视频处理测试 (`video`)
测试连续视频帧的处理能力：
- 解析视频文件信息
- 连续处理视频帧
- 统计检测结果和性能指标

```bash
./HighwayEventDemo video /path/to/your/video.mp4
```

### 5. 压力测试 (`stress`)
测试多线程并发场景下的系统稳定性：
- 多线程并发添加帧
- 测试队列管理和线程安全
- 评估系统最大吞吐量

```bash
./HighwayEventDemo stress
```

### 6. 全面测试 (`all`)
依次运行所有测试 (除视频测试外，需要单独提供视频文件):

```bash
./HighwayEventDemo all
```

## 🔧 配置参数

测试程序使用的主要配置参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `semantic_threads` | 8 | 语义分割线程数 |
| `mask_threads` | 8 | Mask后处理线程数 |
| `detection_threads` | 8 | 目标检测线程数 |
| `tracking_threads` | 1 | 目标跟踪线程数 |
| `filter_threads` | 4 | 目标框筛选线程数 |
| `input_queue_capacity` | 100-500 | 输入队列容量 |
| `result_queue_capacity` | 500-1000 | 结果队列容量 |

## 📊 输出信息说明

### 状态指示符
- ✅ 成功操作
- ❌ 失败操作
- ⏰ 超时
- 📥 添加帧
- 📤 获取结果
- 🔄 处理中
- 📊 统计信息
- 🎯 测试开始
- 🎉 测试完成

### 性能指标
- **处理延迟**: 单帧从添加到获取结果的时间
- **吞吐量**: 每秒处理的帧数
- **成功率**: 成功处理的帧数百分比
- **检测统计**: 每帧检测到的目标数量

## 📁 文件说明

```
项目根目录/
├── highway_event_demo.cpp     # 主要的demo测试程序
├── run_demo.sh               # 自动构建和运行脚本
├── example_usage.cpp         # 原有的使用示例
├── test.jpg                  # 单张图片测试用的测试图片
└── build/
    └── HighwayEventDemo      # 编译生成的demo可执行文件
```

## 🐛 故障排除

### 1. 构建失败
- 检查所有依赖库是否正确安装 (OpenCV, TensorRT, CUDA等)
- 确认路径配置在CMakeLists.txt中正确
- 检查编译器版本兼容性

### 2. 运行时错误
- 确保GPU驱动和CUDA正确安装
- 检查模型文件是否存在
- 验证输入图片/视频文件格式

### 3. 性能问题
- 调整线程数配置以匹配硬件
- 检查GPU内存使用情况
- 考虑减少队列容量

### 4. 内存问题
- 监控系统内存使用
- 及时调用`cleanup_results_before()`清理旧结果
- 调整队列容量参数

## 🎯 使用建议

1. **首次运行**: 建议先运行API测试确保基本功能正常
2. **性能评估**: 使用批量测试和压力测试评估系统性能
3. **实际应用**: 使用视频测试模拟真实使用场景
4. **调优参考**: 根据测试结果调整配置参数

## 📝 开发建议

如果需要扩展或修改demo：

1. **添加新测试**: 在`HighwayEventDemo`类中添加新的测试方法
2. **修改配置**: 调整各个测试中的`HighwayEventConfig`参数
3. **自定义输入**: 修改测试图片或视频路径
4. **结果分析**: 增加更详细的结果分析和可视化

## 📞 技术支持

如遇到问题，请检查：
1. 系统环境配置
2. 依赖库版本
3. 模型文件完整性
4. 硬件资源是否充足
