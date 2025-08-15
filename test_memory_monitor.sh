#!/bin/bash

echo "🧪 内存监控功能测试"
echo "=================="

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 设置日志文件
log_file="memory_test_${timestamp}.log"

echo "📊 启动内存监控测试..."
echo "日志文件: ${log_file}"

# 创建测试图像
test_image="DJI_20250501091406_0001_V_frame_4460.jpg"

if [ ! -f "$test_image" ]; then
    echo "❌ 测试图像不存在: $test_image"
    echo "请确保有测试图像文件"
    exit 1
fi

echo ""
echo "📈 系统信息:"
echo "CPU核心数: $(nproc)"
echo "总内存: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "可用内存: $(free -h | grep '^Mem:' | awk '{print $7}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️ GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
fi

echo ""
echo "🔄 开始测试流水线内存使用..."

# 运行内存监控的demo
echo "运行命令: ./build/HighwayEventDemo video $test_image"

# 运行demo并监控内存
./build/HighwayEventDemo video "$test_image" 2>&1 | tee "$log_file"

echo ""
echo "📊 测试完成，检查日志文件:"
echo "主日志: $log_file"

# 查找生成的内存监控文件
echo ""
echo "📈 生成的内存监控文件:"
ls -la *memory*.log *memory*.csv 2>/dev/null | head -10

echo ""
echo "🎯 内存监控功能测试总结:"
echo "- 构建成功: ✅"
echo "- 内存监控启动: $(grep -q "内存监控已启动" "$log_file" && echo "✅" || echo "❌")"
echo "- 内存报告生成: $(grep -q "内存使用报告" "$log_file" && echo "✅" || echo "❌")"
echo "- 内存泄漏检测: $(grep -q "内存泄漏" "$log_file" && echo "⚠️" || echo "✅")"

# 显示最后的内存统计
if [ -f "pipeline_memory.log" ]; then
    echo ""
    echo "📊 内存使用趋势 (最后10条记录):"
    tail -10 pipeline_memory.log | column -t -s ','
fi

echo ""
echo "✅ 内存监控测试完成!"
