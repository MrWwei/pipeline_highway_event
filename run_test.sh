#!/bin/bash

# 设置LD_LIBRARY_PATH来解决库依赖问题
export LD_LIBRARY_PATH="/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/lib:/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/onnxruntime/lib:/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/paddle_inference/paddle/lib:/home/ubuntu/wtwei/FastDeploy-release-1.1.0/build/compiled_fastdeploy_sdk/third_libs/install/openvino/runtime/lib:$LD_LIBRARY_PATH"

echo "启动流水线测试..."
cd /home/ubuntu/wtwei/pipeline_highway_event
timeout 60s ./build/PipelineHighwayEvent
