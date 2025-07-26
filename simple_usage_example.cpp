#include "highway_event.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

/**
 * 高速公路事件检测器纯净接口简单示例
 * 演示三个核心功能：初始化、添加数据、获取结果
 */
int main() {
    std::cout << "=== 高速公路事件检测器纯净接口示例 ===" << std::endl;
    
    // ========== 1. 初始化流水线 ==========
    std::cout << "\n🔧 步骤1: 初始化流水线" << std::endl;
    
    HighwayEventDetector detector;
    
    // 配置参数
    HighwayEventConfig config;
    config.semantic_threads = 2;
    config.mask_threads = 1;
    config.detection_threads = 2;
    config.tracking_threads = 1;
    config.filter_threads = 1;
    config.enable_debug_log = true;  // 启用调试日志
    config.seg_enable_show = false;  // 不保存可视化结果
    
    // 初始化
    if (!detector.initialize(config)) {
        std::cerr << "❌ 初始化失败" << std::endl;
        return -1;
    }
    
    std::cout << "✅ 流水线初始化成功" << std::endl;
    
    // ========== 2. 添加数据到流水线 ==========
    std::cout << "\n📥 步骤2: 添加图像数据" << std::endl;
    
    // 读取测试图片
    cv::Mat test_image = cv::imread("test.jpg");
    if (test_image.empty()) {
        std::cerr << "❌ 无法读取测试图片 test.jpg" << std::endl;
        std::cerr << "   请确保当前目录下有 test.jpg 文件" << std::endl;
        return -1;
    }
    
    std::cout << "📷 读取图片成功，尺寸: " << test_image.cols << "x" << test_image.rows << std::endl;
    
    // 添加图像到流水线，获取帧序号
    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t frame_id = detector.add_frame(test_image);
    
    if (frame_id < 0) {
        std::cerr << "❌ 添加图像失败" << std::endl;
        return -1;
    }
    
    std::cout << "📌 图像已添加到流水线，分配的帧序号: " << frame_id << std::endl;
    
    // ========== 3. 获取处理结果 ==========
    std::cout << "\n📤 步骤3: 获取处理结果" << std::endl;
    
    // 获取指定帧序号的处理结果（带30秒超时）
    ProcessResult result = detector.get_result_with_timeout(frame_id, 30000);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "⏱️  总处理时间: " << duration.count() << " ms" << std::endl;
    
    // 分析处理结果
    std::cout << "\n📋 处理结果分析:" << std::endl;
    std::cout << "   帧序号: " << result.frame_id << std::endl;
    
    switch (result.status) {
    case ResultStatus::SUCCESS:
        std::cout << "   状态: ✅ 处理成功" << std::endl;
        std::cout << "   检测到目标数量: " << result.detections.size() << std::endl;
        
        // 显示检测到的目标
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& box = result.detections[i];
            std::cout << "   目标 " << i+1 << ": [" 
                      << box.left << "," << box.top << "," 
                      << box.right << "," << box.bottom << "] "
                      << "置信度:" << box.confidence 
                      << " 类别:" << box.class_id 
                      << " 跟踪ID:" << box.track_id << std::endl;
        }
        
        // 显示筛选结果
        if (result.has_filtered_box) {
            const auto& box = result.filtered_box;
            std::cout << "   筛选目标: [" 
                      << box.left << "," << box.top << "," 
                      << box.right << "," << box.bottom << "] "
                      << "置信度:" << box.confidence << std::endl;
        } else {
            std::cout << "   无筛选目标" << std::endl;
        }
        
        // 显示ROI信息
        std::cout << "   感兴趣区域: [" 
                  << result.roi.x << "," << result.roi.y << "," 
                  << result.roi.width << "," << result.roi.height << "]" << std::endl;
        
        break;
        
    case ResultStatus::TIMEOUT:
        std::cout << "   状态: ⏰ 处理超时" << std::endl;
        break;
        
    case ResultStatus::NOT_FOUND:
        std::cout << "   状态: ❓ 帧未找到" << std::endl;
        break;
        
    case ResultStatus::ERROR:
        std::cout << "   状态: ❌ 处理错误" << std::endl;
        break;
        
    default:
        std::cout << "   状态: ⏳ 处理中" << std::endl;
        break;
    }
    
    // 显示流水线状态
    std::cout << "\n📊 流水线状态信息:" << std::endl;
    std::cout << detector.get_pipeline_status() << std::endl;
    
    // ========== 批量处理示例 ==========
    std::cout << "\n🔄 批量处理示例:" << std::endl;
    
    std::vector<int64_t> frame_ids;
    const int batch_size = 3;
    
    // 添加多帧图像
    for (int i = 0; i < batch_size; ++i) {
        int64_t fid = detector.add_frame(test_image);
        if (fid >= 0) {
            frame_ids.push_back(fid);
            std::cout << "📥 添加批次帧 " << fid << std::endl;
        }
    }
    
    // 获取所有结果
    for (int64_t fid : frame_ids) {
        ProcessResult batch_result = detector.get_result_with_timeout(fid, 15000);
        std::cout << "📤 帧 " << fid << " 处理状态: " 
                  << (batch_result.status == ResultStatus::SUCCESS ? "成功" : "失败") 
                  << std::endl;
    }
    
    std::cout << "\n🎉 示例运行完成！" << std::endl;
    std::cout << "流水线将自动停止和清理资源..." << std::endl;
    
    // detector析构时会自动调用stop()进行清理
    return 0;
}
