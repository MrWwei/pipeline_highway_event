#!/bin/bash

# 高速公路事件检测系统 Demo 测试脚本

set -e

echo "🚗 高速公路事件检测系统 Demo 构建和测试"
echo "======================================="

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 错误: 请在项目根目录下运行此脚本"
    exit 1
fi

# 创建构建目录
echo "📁 创建构建目录..."
mkdir -p build
cd build
FASTDEPLOY_INSTALL_DIR="/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk"
# 构建项目
echo "🔨 构建项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Release -DFASTDEPLOY_INSTALL_DIR="$FASTDEPLOY_INSTALL_DIR"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 构建失败"
    exit 1
fi

echo "✅ 构建成功!"

# 检查可执行文件
if [ ! -f "HighwayEventDemo" ]; then
    echo "❌ Demo可执行文件未找到"
    exit 1
fi

echo "🎯 Demo程序已准备就绪"

export THIRD_PARTY=/home/ubuntu/ThirdParty
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/lib:\
/home/ubuntu/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib:\
/usr/local/cuda/lib64:\
$THIRD_PARTY/FFmpeg-n6.0/install/lib:\
$THIRD_PARTY/TensorRT-8.5.1.7/lib:\
/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/onnxruntime/lib:\
/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/lib:\
/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/opencv/lib64:\
/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/openvino/runtime/lib:\
/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/paddle2onnx/lib:\
/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/paddle_inference/paddle/lib:\
$LD_LIBRARY_PATH
cd ..
./build/HighwayEventDemo video /home/ubuntu/Desktop/DJI_20250501091406_0001.mp4