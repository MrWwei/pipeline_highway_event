#include "highway_event.h"
#include "memory_monitor.h"
#include "logger_manager.h"
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
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sys/resource.h>

class HighwayEventDemo {
private:
    std::unique_ptr<HighwayEventDetector> detector_;
    std::unique_ptr<MemoryMonitor> memory_monitor_;
    
    // 内存监控函数
    size_t get_memory_usage_mb() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024; // 转换为MB (Linux)
    }
    
public:
    HighwayEventDemo() {
        detector_ = create_highway_event_detector();
        
        // 初始化内存监控器
        memory_monitor_ = std::make_unique<MemoryMonitor>("highway_event_demo_memory.log", 500);
        
        // 设置内存告警回调
        memory_monitor_->set_memory_warning_callback([](const MemoryStats& stats) {
            std::cout << "⚠️ 内存告警: 进程内存 " << stats.process_memory_mb << " MB, "
                      << "系统内存使用率 " << std::fixed << std::setprecision(1) 
                      << stats.memory_usage_percent << "%" << std::endl;
        });
        
        // 设置内存泄漏检测阈值为20MB/分钟
        memory_monitor_->set_leak_detection_threshold(20.0);
        
        // 启动内存监控
        memory_monitor_->start();
        std::cout << "📊 内存监控已启动" << std::endl;
    }
    
    ~HighwayEventDemo() {
        if (memory_monitor_) {
            std::cout << "\n📊 最终内存报告:" << std::endl;
            memory_monitor_->print_memory_report();
            memory_monitor_->stop();
        }
    }
    
    /**
     * 视频文件阻塞式处理测试 - 解码出帧直接添加到流水线，阻塞获取结果
     */
    void test_video_stream_processing(const std::string& video_path) {
        SCOPED_MEMORY_MONITOR_WITH_MONITOR("视频流处理测试", memory_monitor_.get());
        std::cout << "\n=== 🎬 视频阻塞式处理测试 (仅目标检测) ===" << std::endl;
        
        MEMORY_CHECKPOINT(memory_monitor_.get(), "开始视频处理");
        
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
        
        MEMORY_CHECKPOINT(memory_monitor_.get(), "视频信息获取完成");
        
        // 配置高性能参数
        HighwayEventConfig config;
        
        config.semantic_threads = 1;
        config.mask_threads = 8;
        config.detection_threads = 1;
        config.tracking_threads = 1;
        config.filter_threads = 1;
        config.result_queue_capacity = 50; // 适合流式处理的队列大小
        config.enable_debug_log = false;
        config.enable_segmentation = true; // 关闭语义分割
        config.enable_mask_postprocess = true; // 关闭mask后处理
        config.enable_detection = true;
        config.enable_tracking = true; // 关闭目标跟踪模块
        config.enable_event_determine = true;   // 关闭事件判定

        config.seg_model_path = "/home/ubuntu/wtwei/seg_trt/pidnet_resize.onnx"; // 语义分割模型路径
        config.car_det_model_path = "car_detect.onnx"; // 车辆检测模型路径
        config.pedestrian_det_model_path = "Pedestrain_TAG1_yl_S640_V1.2.onnx"; // 行人检测模型路径

        config.enable_seg_show = false;
        config.seg_show_image_path = "./segmentation_results/"; // 分割结果图像保存路径
        config.get_timeout_ms = 100000; // 阻塞处理使用较长超时

        config.times_car_width = 1.2f; // 车宽倍数
        config.enable_lane_show = false; // 关闭车道线可视化
        config.lane_show_image_path = "./lane_results/"; // 车道线结果
        config.enable_pedestrian_detect = false;
        
        
        if (!detector_->initialize(config) || !detector_->start()) {
            std::cerr << "❌ 初始化或启动失败" << std::endl;
            cap.release();
            return;
        }
        
        MEMORY_CHECKPOINT(memory_monitor_.get(), "检测器初始化完成");
        
        // 流式处理参数
        std::atomic<int> total_frames_processed{0};
        std::atomic<int> total_successful{0};
        std::atomic<int> total_detections{0};
        std::atomic<int> frame_number{0};
        std::atomic<bool> processing_finished{false};
        
        // 用于存储待处理的frame_id队列
        std::queue<int64_t> pending_frame_ids;
        std::mutex pending_mutex;
        std::condition_variable pending_cv;
        
        auto process_start = std::chrono::high_resolution_clock::now();
        
        MEMORY_CHECKPOINT(memory_monitor_.get(), "开始视频处理循环");
        
        // 创建结果获取线程
        std::thread result_thread([&]() {
            std::cout << "🔄 结果获取线程启动" << std::endl;
            
            while (!processing_finished.load() || !pending_frame_ids.empty()) {
                std::unique_lock<std::mutex> lock(pending_mutex);
                
                // 等待有frame_id可处理，或者处理完成
                pending_cv.wait(lock, [&]() { 
                    return !pending_frame_ids.empty() || processing_finished.load(); 
                });
                
                if (pending_frame_ids.empty()) {
                    if (processing_finished.load()) {
                        break;
                    }
                    continue;
                }
                
                // 取出一个frame_id
                int64_t frame_id = pending_frame_ids.front();
                pending_frame_ids.pop();
                lock.unlock();
                
                // 阻塞方式获取结果
                auto result = detector_->get_result(frame_id);
                // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                
                if (result.status == ResultStatus::SUCCESS) {
                    // cv::Mat image = result.srcImage;
                    // cv::imwrite("src_outs/output_" + std::to_string(result.frame_id) + ".jpg", image);
                    // cv::Mat mask = result.mask;
                    // cv::imwrite("mask_outs/output_" + std::to_string(result.frame_id) + ".jpg", mask);
                    // for(auto box:result.detections){
                    //     cv::Scalar color(0, 255, 0);
                    //     if(box.status == ObjectStatus::OCCUPY_EMERGENCY_LANE){
                    //         color = cv::Scalar(0, 0, 255); // 红色表示占用应急车道
                    //     } 
                        // cv::rectangle(result.srcImage, 
                        //           cv::Point(box.left, box.top), 
                        //           cv::Point(box.right, box.bottom), 
                        //           color, 2);
                       
                        // cv::putText(result.srcImage,
                        //             std::to_string(box.track_id) +" " + std::to_string(int(box.confidence)),
                        //             cv::Point(box.left, box.top - 10),
                        //             cv::FONT_HERSHEY_SIMPLEX,
                        //             0.5,
                        //             color, 1);
                    // }
                    // cv::imwrite("src_outs/output_" + std::to_string(result.frame_id) + ".jpg", result.srcImage);
                    total_successful.fetch_add(1);
                    total_detections.fetch_add(result.detections.size());
                } else {
                    std::cout << "❌ 帧 " << frame_id << " 处理失败或超时" << std::endl;
                    std::cout << "   状态: " << static_cast<int>(result.status) << std::endl;
                }
            }
            
            std::cout << "🔄 结果获取线程结束" << std::endl;
        });
        
        cv::Mat frame;
        auto last_status_time = std::chrono::high_resolution_clock::now();
        
        while (cap.read(frame) && !frame.empty()) {

            // while(true){
                frame_number.fetch_add(1);
            
                // 添加帧到流水线
                int64_t frame_id = detector_->add_frame(frame.clone());
                if (frame_id >= 0) {
                    total_frames_processed.fetch_add(1);
                    
                    // 将frame_id添加到待处理队列
                    {
                        std::lock_guard<std::mutex> lock(pending_mutex);
                        pending_frame_ids.push(frame_id);
                    }
                    pending_cv.notify_one();
                    
                
                }
                
                // 每隔一定时间显示状态
                // auto current_time = std::chrono::high_resolution_clock::now();
                // if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_status_time).count() >= 2) {
                //     std::cout << "\n📊 流水线实时状态 (第 " << frame_number.load() << " 帧):" << std::endl;
                detector_->get_pipeline_status(); // 显示流水线各模块的队列状态
                //     std::cout << "📈 处理进度: 已提交 " << total_frames_processed.load() 
                //               << " 帧，已完成 " << total_successful.load() << " 帧" << std::endl;
                //     last_status_time = current_time;
                // }

            // }
            
        }
        
        // 标记处理完成，通知结果线程
        processing_finished.store(true);
        pending_cv.notify_all();
        
        // 等待结果线程完成
        std::cout << "⏳ 等待所有结果处理完成..." << std::endl;
        if (result_thread.joinable()) {
            result_thread.join();
        }
        
        // 等待所有剩余帧处理完成
        auto process_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
        
        // 显示最终统计信息
        std::cout << "\n📊 最终处理统计:" << std::endl;
        std::cout << "   总处理时间: " << total_duration.count() << " ms" << std::endl;
        std::cout << "   已提交帧数: " << total_frames_processed.load() << std::endl;
        std::cout << "   成功处理帧数: " << total_successful.load() << std::endl;
        std::cout << "   总检测目标数: " << total_detections.load() << std::endl;
        if (total_successful.load() > 0) {
            std::cout << "   平均每帧检测目标: " << (double)total_detections.load() / total_successful.load() << std::endl;
            std::cout << "   处理成功率: " << (double)total_successful.load() / total_frames_processed.load() * 100 << "%" << std::endl;
        }
        
        MEMORY_CHECKPOINT(memory_monitor_.get(), "视频处理完成");
        
        cap.release();
        detector_->stop();
        
        MEMORY_CHECKPOINT(memory_monitor_.get(), "检测器停止完成");
        
        // 显示内存使用总结
        std::cout << "\n📊 内存使用总结:" << std::endl;
        if (memory_monitor_->is_memory_leak_detected()) {
            std::cout << "⚠️  检测到内存泄漏!" << std::endl;
        } else {
            std::cout << "✅ 未检测到明显的内存泄漏" << std::endl;
        }
        memory_monitor_->print_memory_report();
    }
};

void print_usage() {
    std::cout << "用法: ./highway_event_demo video [视频文件路径]" << std::endl;
    std::cout << "\n功能说明:" << std::endl;
    std::cout << "  此程序对视频文件进行阻塞式处理，仅使用目标检测模块" << std::endl;
    std::cout << "  关闭目标跟踪模块，检测结果直接送到结果队列" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  ./highway_event_demo video /path/to/video.mp4" << std::endl;
    std::cout << "  ./highway_event_demo video /home/ubuntu/Desktop/test_video.mp4" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🚗 高速公路事件检测系统 - 阻塞式目标检测程序" << std::endl;
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
            demo.test_video_stream_processing(argv[2]);
        }
        else {
            std::cerr << "❌ 未知的测试类型: " << test_type << std::endl;
            std::cerr << "💡 当前版本只支持阻塞式目标检测处理" << std::endl;
            print_usage();
            return 1;
        }
        
        std::cout << "\n🎉 阻塞式目标检测处理完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 处理过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
