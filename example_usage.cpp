#include "highway_event.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main() {
    // 1. 创建检测器实例
    HighwayEventDetector detector;
    
    // 2. 配置参数
    HighwayEventConfig config;
    config.semantic_threads = 8;
    config.mask_threads = 8;
    config.detection_threads = 8;
    config.tracking_threads = 1;
    config.filter_threads = 4;
    config.add_timeout_ms = 5000;
    config.get_result_timeout_ms = 10000;
    config.enable_debug_log = true;
    config.enable_status_print = true;
    
    // 3. 初始化
    if (!detector.initialize(config)) {
        std::cerr << "❌ 初始化失败" << std::endl;
        return -1;
    }
    
    // 4. 启动流水线
    if (!detector.start()) {
        std::cerr << "❌ 启动失败" << std::endl;
        return -1;
    }
    
    // 5. 打开视频文件
    cv::VideoCapture cap("/home/ubuntu/Desktop/DJI_20250501091406_0001.mp4");
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频文件" << std::endl;
        return -1;
    }
    
    std::cout << "✅ 视频文件打开成功" << std::endl;
    
    // 6. 处理视频帧
    cv::Mat frame;
    std::vector<int64_t> frame_ids;
    const int max_frames = 100; // 处理100帧作为演示
    
    std::cout << "🎬 开始处理视频帧..." << std::endl;
    
    // 添加帧到流水线
    for (int i = 0; i < max_frames && cap.read(frame); ++i) {
        if (frame.empty()) {
            std::cerr << "⚠️ 读取到空帧，跳过" << std::endl;
            continue;
        }
        
        // 添加帧到流水线
        int64_t frame_id = detector.add_frame(frame);
        if (frame_id >= 0) {
            frame_ids.push_back(frame_id);
            std::cout << "📥 添加帧 " << frame_id << " (总共: " << frame_ids.size() << ")" << std::endl;
        } else {
            std::cerr << "❌ 添加帧失败" << std::endl;
        }
        
        // 每10帧打印一次状态
        if ((i + 1) % 10 == 0) {
            detector.print_status();
        }
    }
    
    std::cout << "📊 完成添加 " << frame_ids.size() << " 帧到流水线" << std::endl;
    
    // 7. 获取处理结果
    std::cout << "📤 开始获取处理结果..." << std::endl;
    
    int success_count = 0;
    int timeout_count = 0;
    int error_count = 0;
    
    for (size_t i = 0; i < frame_ids.size(); ++i) {
        int64_t frame_id = frame_ids[i];
        
        std::cout << "⏳ 等待帧 " << frame_id << " 的结果..." << std::endl;
        
        // 获取结果（带超时）
        auto result = detector.get_result_with_timeout(frame_id, 15000); // 15秒超时
        
        switch (result.status) {
        case ResultStatus::SUCCESS:
            std::cout << "✅ 帧 " << frame_id << " 处理成功";
            if (result.result) {
                std::cout << " (检测到 " << result.result->track_results.size() << " 个目标)";
                if (result.result->has_filtered_box) {
                    std::cout << " [筛选目标: 置信度=" << result.result->filtered_box.confidence << "]";
                }
            }
            std::cout << std::endl;
            success_count++;
            break;
            
        case ResultStatus::TIMEOUT:
            std::cout << "⏰ 帧 " << frame_id << " 处理超时" << std::endl;
            timeout_count++;
            break;
            
        case ResultStatus::NOT_FOUND:
            std::cout << "❓ 帧 " << frame_id << " 结果未找到" << std::endl;
            error_count++;
            break;
            
        case ResultStatus::PIPELINE_STOPPED:
            std::cout << "🛑 流水线已停止" << std::endl;
            error_count++;
            break;
            
        case ResultStatus::ERROR:
            std::cout << "❌ 帧 " << frame_id << " 处理错误" << std::endl;
            error_count++;
            break;
        }
        
        // 每10个结果打印一次进度
        if ((i + 1) % 10 == 0) {
            std::cout << "📊 进度: " << (i + 1) << "/" << frame_ids.size() 
                      << " (成功: " << success_count 
                      << ", 超时: " << timeout_count 
                      << ", 错误: " << error_count << ")" << std::endl;
            detector.print_status();
        }
    }
    
    // 8. 打印最终统计
    std::cout << "\n📈 处理完成统计:" << std::endl;
    std::cout << "   总帧数: " << frame_ids.size() << std::endl;
    std::cout << "   成功: " << success_count << std::endl;
    std::cout << "   超时: " << timeout_count << std::endl;
    std::cout << "   错误: " << error_count << std::endl;
    std::cout << "   成功率: " << (success_count * 100.0 / frame_ids.size()) << "%" << std::endl;
    
    // 9. 清理资源
    std::cout << "🧹 清理资源..." << std::endl;
    cap.release();
    
    // 停止流水线
    if (!detector.stop()) {
        std::cerr << "❌ 停止流水线失败" << std::endl;
        return -1;
    }
    
    std::cout << "✅ 程序完成!" << std::endl;
    return 0;
}
