#include "highway_event.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <random>
#include <fstream>
#include <iomanip>
#include <queue>
#include <sys/resource.h>

class HighwayEventDemo {
private:
    std::unique_ptr<HighwayEventDetector> detector_;
    
    // 内存监控函数
    size_t get_memory_usage_mb() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024; // 转换为MB (Linux)
    }
    
public:
    HighwayEventDemo() {
        detector_ = create_highway_event_detector();
    }
    
    /**
     * 视频文件批量处理测试 - 每32帧为一批
     */
    void test_video_batch_processing(const std::string& video_path) {
        std::cout << "\n=== 🎬 视频批量处理测试 (每32帧一批) ===" << std::endl;
        
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "❌ 无法打开视频文件: " << video_path << std::endl;
            return;
        }
        
        // 获取视频信息
        int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        std::cout << "📹 视频信息:" << std::endl;
        std::cout << "   尺寸: " << width << "x" << height << std::endl;
        std::cout << "   FPS: " << fps << std::endl;
        std::cout << "   总帧数: " << frame_count << std::endl;
        
        // 配置高性能参数
        HighwayEventConfig config;
        config.semantic_threads = 6;
        config.mask_threads = 6;
        config.detection_threads = 6;
        config.tracking_threads = 1;
        config.filter_threads = 3;
        config.result_queue_capacity = 50; // 适合批量处理的队列大小
        config.enable_debug_log = false;
        config.get_timeout_ms = 30000; // 增加超时时间适应批量处理
        config.enable_detection = true;
        config.enable_tracking = true;
        config.enable_box_filter = true;
        config.enable_mask_postprocess = true;
        
        if (!detector_->initialize(config) || !detector_->start()) {
            std::cerr << "❌ 初始化或启动失败" << std::endl;
            cap.release();
            return;
        }
        
        // 批量处理参数
        const int BATCH_SIZE = 32;
        int total_frames_processed = 0;
        int total_successful = 0;
        int total_detections = 0;
        int batch_number = 0;
        
        auto process_start = std::chrono::high_resolution_clock::now();
        size_t initial_memory = get_memory_usage_mb();
        std::cout << "🧠 初始内存使用: " << initial_memory << " MB" << std::endl;
        
        std::cout << "🎬 开始批量处理视频，每批 " << BATCH_SIZE << " 帧..." << std::endl;
        
        cv::Mat frame;
        std::vector<cv::Mat> batch_frames;
        std::vector<int64_t> batch_frame_ids;
        
        while (cap.read(frame) && !frame.empty()) {
            batch_frames.push_back(frame.clone());
            
            // 当达到批量大小或者是最后的帧时，处理这一批
            if (batch_frames.size() == BATCH_SIZE || 
                total_frames_processed + batch_frames.size() >= frame_count) {
                
                batch_number++;
                int current_batch_size = batch_frames.size();
                
                std::cout << "\n📦 ========== 处理第 " << batch_number << " 批 ========== " << std::endl;
                std::cout << "📊 批次信息: " << current_batch_size << " 帧 (总进度: " 
                          << total_frames_processed << "/" << frame_count << ")" << std::endl;
                
                auto batch_start = std::chrono::high_resolution_clock::now();
                
                // 步骤1: 批量添加帧到流水线
                std::cout << "📥 步骤1: 批量添加 " << current_batch_size << " 帧到流水线..." << std::endl;
                batch_frame_ids.clear();
                
                for (int i = 0; i < current_batch_size; ++i) {
                    int64_t frame_id = detector_->add_frame(std::move(batch_frames[i]));
                    if (frame_id >= 0) {
                        batch_frame_ids.push_back(frame_id);
                        if ((i + 1) % 8 == 0 || i == current_batch_size - 1) {
                            std::cout << "   已添加 " << (i + 1) << "/" << current_batch_size << " 帧" << std::endl;
                        }
                    } else {
                        std::cout << "⚠️ 第 " << i << " 帧添加失败" << std::endl;
                    }
                }
                
                auto add_end = std::chrono::high_resolution_clock::now();
                auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - batch_start);
                std::cout << "✅ 添加完成，耗时: " << add_duration.count() << " ms，成功添加: " 
                          << batch_frame_ids.size() << "/" << current_batch_size << " 帧" << std::endl;
                
                // 步骤2: 等待所有帧处理完成并获取结果
                std::cout << "🔄 步骤2: 等待批量处理完成并获取结果..." << std::endl;
                int batch_successful = 0;
                int batch_detections = 0;
                
                for (size_t i = 0; i < batch_frame_ids.size(); ++i) {
                    auto result = detector_->get_result_with_timeout(batch_frame_ids[i], 30000);
                    
                    if (result.status == ResultStatus::SUCCESS) {
                        batch_successful++;
                        batch_detections += result.detections.size();
                        
                        if ((i + 1) % 8 == 0 || i == batch_frame_ids.size() - 1) {
                            std::cout << "   已获取 " << (i + 1) << "/" << batch_frame_ids.size() 
                                      << " 个结果 (成功: " << batch_successful << ")" << std::endl;
                        }
                    } else if (result.status == ResultStatus::TIMEOUT) {
                        std::cout << "⏰ 帧 " << batch_frame_ids[i] << " 处理超时" << std::endl;
                    } else {
                        std::cout << "❌ 帧 " << batch_frame_ids[i] << " 处理失败" << std::endl;
                    }
                }
                
                auto batch_end = std::chrono::high_resolution_clock::now();
                auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
                auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - add_end);
                
                // 更新统计信息
                total_frames_processed += current_batch_size;
                total_successful += batch_successful;
                total_detections += batch_detections;
                
                // 获取当前内存使用
                size_t current_memory = get_memory_usage_mb();
                
                // 输出批次统计
                std::cout << "📊 批次 " << batch_number << " 统计:" << std::endl;
                std::cout << "   处理帧数: " << current_batch_size << std::endl;
                std::cout << "   成功帧数: " << batch_successful << std::endl;
                std::cout << "   成功率: " << (batch_successful * 100.0 / current_batch_size) << "%" << std::endl;
                std::cout << "   检测目标数: " << batch_detections << std::endl;
                std::cout << "   平均检测数: " << (batch_successful > 0 ? batch_detections / (double)batch_successful : 0) << " 个/帧" << std::endl;
                std::cout << "⏱️  批次耗时:" << std::endl;
                std::cout << "   总耗时: " << batch_duration.count() << " ms" << std::endl;
                std::cout << "   添加耗时: " << add_duration.count() << " ms" << std::endl;
                std::cout << "   处理耗时: " << process_duration.count() << " ms" << std::endl;
                std::cout << "   平均处理时间: " << (process_duration.count() / current_batch_size) << " ms/帧" << std::endl;
                std::cout << "🧠 内存使用: " << current_memory << " MB (增长: " 
                          << (current_memory - initial_memory) << " MB)" << std::endl;
                
                // 清空当前批次的帧数据
                batch_frames.clear();
                
                // 输出累计统计
                double overall_progress = (total_frames_processed * 100.0) / frame_count;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - process_start);
                
                std::cout << "📈 累计统计:" << std::endl;
                std::cout << "   总进度: " << std::fixed << std::setprecision(1) << overall_progress << "% "
                          << "(" << total_frames_processed << "/" << frame_count << ")" << std::endl;
                std::cout << "   总成功率: " << (total_successful * 100.0 / total_frames_processed) << "%" << std::endl;
                std::cout << "   总运行时间: " << elapsed.count() << " 秒" << std::endl;
                std::cout << "   平均处理速度: " << (total_frames_processed / (double)elapsed.count()) << " 帧/秒" << std::endl;
                std::cout << "   检测目标总数: " << total_detections << std::endl;
                
                std::cout << "📊 流水线状态: " << detector_->get_pipeline_status() << std::endl;
                
                // 短暂休息，让系统稳定
                if (total_frames_processed < frame_count) {
                    std::cout << "😴 批次间休息 2 秒..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
        }
        
        auto process_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
        size_t final_memory = get_memory_usage_mb();
        
        std::cout << "\n🎉 ========== 视频批量处理完成 ==========" << std::endl;
        std::cout << "📊 最终统计:" << std::endl;
        std::cout << "   视频总帧数: " << frame_count << std::endl;
        std::cout << "   处理帧数: " << total_frames_processed << std::endl;
        std::cout << "   成功帧数: " << total_successful << std::endl;
        std::cout << "   总成功率: " << (total_successful * 100.0 / total_frames_processed) << "%" << std::endl;
        std::cout << "   批次数量: " << batch_number << std::endl;
        std::cout << "   平均批次大小: " << (total_frames_processed / (double)batch_number) << " 帧" << std::endl;
        std::cout << "⏱️  时间统计:" << std::endl;
        std::cout << "   总处理时间: " << total_duration.count() << " ms (" << (total_duration.count() / 1000.0) << " 秒)" << std::endl;
        std::cout << "   平均处理时间: " << (total_duration.count() / total_frames_processed) << " ms/帧" << std::endl;
        std::cout << "   实际吞吐量: " << (total_frames_processed * 1000.0 / total_duration.count()) << " 帧/秒" << std::endl;
        std::cout << "   相对原视频速度: " << (total_frames_processed * 1000.0 / total_duration.count() / fps) << "x" << std::endl;
        std::cout << "🎯 检测统计:" << std::endl;
        std::cout << "   检测目标总数: " << total_detections << std::endl;
        std::cout << "   平均检测数: " << (total_successful > 0 ? total_detections / (double)total_successful : 0) << " 个/帧" << std::endl;
        std::cout << "🧠 内存统计:" << std::endl;
        std::cout << "   初始内存: " << initial_memory << " MB" << std::endl;
        std::cout << "   最终内存: " << final_memory << " MB" << std::endl;
        std::cout << "   内存增长: " << (final_memory - initial_memory) << " MB" << std::endl;
        
        cap.release();
        detector_->stop();
    }
};

void print_usage() {
    std::cout << "用法: ./highway_event_demo video [视频文件路径]" << std::endl;
    std::cout << "\n功能说明:" << std::endl;
    std::cout << "  此程序对视频文件进行批量处理，每32帧为一批" << std::endl;
    std::cout << "  等待每批处理完成后再进行下一批，确保内存稳定" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  ./highway_event_demo video /path/to/video.mp4" << std::endl;
    std::cout << "  ./highway_event_demo video /home/ubuntu/Desktop/test_video.mp4" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🚗 高速公路事件检测系统 - 视频批量处理程序" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string test_type = argv[1];
    HighwayEventDemo demo;
    
    try {
        if (test_type == "video") {
            if (argc < 3) {
                std::cerr << "❌ 视频测试需要提供视频文件路径" << std::endl;
                print_usage();
                return 1;
            }
            demo.test_video_batch_processing(argv[2]);
        }
        else {
            std::cerr << "❌ 未知的测试类型: " << test_type << std::endl;
            std::cerr << "💡 当前版本只支持视频批量处理" << std::endl;
            print_usage();
            return 1;
        }
        
        std::cout << "\n🎉 视频批量处理完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 处理过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
