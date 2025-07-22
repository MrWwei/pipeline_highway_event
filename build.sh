#!/bin/bash

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

# 高速公路事件检测流水线系统构建脚本

echo "=== 高速公路事件检测流水线系统 构建脚本 ==="
echo

# 创建构建目录
if [ ! -d "build" ]; then
    echo "📁 创建构建目录..."
    mkdir build
fi

cd build

# 设置FastDeploy路径（需要根据实际情况修改）
FASTDEPLOY_INSTALL_DIR="/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk"

# 配置项目
echo "⚙️  配置CMake项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DFASTDEPLOY_INSTALL_DIR="$FASTDEPLOY_INSTALL_DIR"

# 检查配置是否成功
if [ $? -ne 0 ]; then
    echo "❌ CMake配置失败!"
    exit 1
fi

# 编译项目
echo "🔨 编译项目..."
make -j$(nproc)

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "❌ 编译失败!"
    exit 1
fi

echo "✅ 编译成功!"
echo

# 询问是否运行程序
read -p "🚀 是否立即运行程序? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏃‍♂️ 运行程序..."
    cd ..
    echo
    ./build/PipelineHighwayEvent
fi

echo
echo "✨ 构建脚本执行完成!"
