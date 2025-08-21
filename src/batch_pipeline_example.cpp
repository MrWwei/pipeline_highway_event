#include "batch_pipeline_manager.h"
#include "logger_manager.h"
#include "pipeline_config.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

/**
 * 批次流水线使用示例
 * 展示如何使用新的批次处理架构替代原来的队列架构
 */

void print_usage() {
    LOG_INFO("批次流水线使用示例");
    LOG_INFO("用法: ./batch_pipeline_example [选项]");
    LOG_INFO("选项:");
    LOG_INFO("  --help          显示此帮助信息");
    LOG_INFO("  --test-images   使用测试图像");
    LOG_INFO("  --duration N    运行N秒 (默认: 30)");
    LOG_INFO("  --fps N         输入帧率 (默认: 25)");
}

// 创建测试图像
cv::Mat create_test_image(int width = 1920, int height = 1080, int frame_idx = 0) {
    cv::Mat image(height, width, CV_8UC3);
    
    // 创建渐变背景
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int r = (x * 255) / width;
            int g = (y * 255) / height;
            int b = ((frame_idx * 5) % 255);
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    
    // 添加一些图形元素
    cv::circle(image, cv::Point(width/2, height/2), 50 + (frame_idx % 100), cv::Scalar(255, 255, 255), 2);
    cv::rectangle(image, cv::Point(100 + (frame_idx % 200), 100), cv::Point(300 + (frame_idx % 200), 300), cv::Scalar(0, 255, 0), 3);
    
    // 添加文本
    std::string text = "Frame: " + std::to_string(frame_idx);
    cv::putText(image, text, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    return image;
}

int main(int argc, char* argv[]) {
    LOG_INFO("🚀 批次流水线使用示例");
    
    // 解析命令行参数
    bool use_test_images = false;
    int duration_seconds = 30;
    int fps = 25;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--test-images") {
            use_test_images = true;
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_seconds = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = std::stoi(argv[++i]);
        }
    }
    
    // 配置流水线参数
    PipelineConfig config;
    
    // 启用各个阶段
    config.enable_segmentation = true;
    config.enable_mask_postprocess = true;
    config.enable_detection = true;
    config.enable_tracking = false;  // 暂时禁用，因为还未实现批次跟踪
    config.enable_event_determine = false;  // 暂时禁用
    
    // 设置线程数 - 针对批次处理优化
    config.semantic_threads = 4;      // 语义分割使用4个线程
    config.mask_postprocess_threads = 2;  // Mask后处理使用2个线程
    config.detection_threads = 4;     // 目标检测使用4个线程
    
    // 设置模型路径（这里使用占位符，实际使用时需要设置正确路径）
    config.seg_model_path = "ppseg_model.trt";
    config.car_model_path = "car_detect.trt";
    config.person_model_path = "Pedestrain_TAG1_yl_S640_V1.2.trt";
    
    // 检测参数
    config.detection_confidence_threshold = 0.5f;
    config.detection_nms_threshold = 0.4f;
    config.enable_car_detection = true;
    config.enable_person_detection = false;  // 暂时只启用车辆检测
    
    // 分割结果保存配置
    config.enable_seg_show = false;  // 禁用可视化以提高性能
    config.seg_show_image_path = "./seg_results/";
    
    LOG_INFO("📋 流水线配置:");
    std::cout << "  语义分割线程数: " << config.semantic_threads << std::endl;
    std::cout << "  Mask后处理线程数: " << config.mask_postprocess_threads << std::endl;
    std::cout << "  目标检测线程数: " << config.detection_threads << std::endl;
    std::cout << "  运行时长: " << duration_seconds << " 秒" << std::endl;
    std::cout << "  输入帧率: " << fps << " FPS" << std::endl;
    
    try {
        // 创建批次流水线管理器
        LOG_INFO("🏗️ 创建批次流水线管理器...");
        BatchPipelineManager pipeline(config);
        
        // 启动流水线
        LOG_INFO("🚀 启动批次流水线...");
        pipeline.start();
        
        // 输入数据线程
        std::thread input_thread([&]() {
            LOG_INFO("📥 输入线程已启动");
            
            uint64_t frame_idx = 0;
            auto frame_interval = std::chrono::milliseconds(1000 / fps);
            auto start_time = std::chrono::high_resolution_clock::now();
            auto end_time = start_time + std::chrono::seconds(duration_seconds);
            
            while (std::chrono::high_resolution_clock::now() < end_time) {
                auto frame_start = std::chrono::high_resolution_clock::now();
                
                // 创建图像数据
                ImageDataPtr image_data;
                if (use_test_images) {
                    // 使用测试图像
                    cv::Mat test_image = create_test_image(1920, 1080, frame_idx);
                    image_data = std::make_shared<ImageData>(std::move(test_image));
                } else {
                    // 创建空白图像（模拟实际输入）
                    cv::Mat blank_image = cv::Mat::zeros(1080, 1920, CV_8UC3);
                    image_data = std::make_shared<ImageData>(std::move(blank_image));
                }
                
                image_data->frame_idx = frame_idx++;
                
                // 添加到流水线
                if (!pipeline.add_image(image_data)) {
                    LOG_ERROR("❌ 无法添加图像到流水线");
                    break;
                }
                
                // 控制帧率
                auto frame_end = std::chrono::high_resolution_clock::now();
                auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
                if (frame_duration < frame_interval) {
                    std::this_thread::sleep_for(frame_interval - frame_duration);
                }
                
                if (frame_idx % 100 == 0) {
                    std::cout << "📥 已输入 " << frame_idx << " 帧" << std::endl;
                }
            }
            
            std::cout << "📥 输入线程结束，总共输入 " << frame_idx << " 帧" << std::endl;
        });
        
        // 输出结果线程
        std::thread output_thread([&]() {
            LOG_INFO("📤 输出线程已启动");
            
            uint64_t output_count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            while (true) {
                ImageDataPtr result_image;
                if (pipeline.get_result_image(result_image)) {
                    if (result_image) {
                        output_count++;
                        
                        if (output_count % 100 == 0) {
                            auto now = std::chrono::high_resolution_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                            double fps_out = 0.0;
                            if (elapsed.count() > 0) {
                                fps_out = (double)output_count / elapsed.count();
                            }
                            
                            std::cout << "📤 已输出 " << output_count << " 帧，平均输出帧率: " 
                                      << std::fixed << std::setprecision(2) << fps_out << " FPS" << std::endl;
                            
                            // 打印检测结果统计
                            if (!result_image->detection_results.empty()) {
                                std::cout << "  🎯 帧 " << result_image->frame_idx 
                                          << " 检测到 " << result_image->detection_results.size() << " 个目标" << std::endl;
                            }
                        }
                    }
                } else {
                    // 没有更多结果，流水线可能已停止
                    break;
                }
            }
            
            std::cout << "📤 输出线程结束，总共输出 " << output_count << " 帧" << std::endl;
        });
        
        // 主线程等待
        std::cout << "⏱️ 流水线运行中，等待 " << duration_seconds << " 秒..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
        
        // 停止流水线
        LOG_INFO("🛑 停止批次流水线...");
        pipeline.stop();
        
        // 等待线程结束
        if (input_thread.joinable()) input_thread.join();
        if (output_thread.joinable()) output_thread.join();
        
        // 打印最终统计信息
        LOG_INFO("\n📊 最终统计信息:");
        auto final_stats = pipeline.get_statistics();
        std::cout << "  总输入图像: " << final_stats.total_images_input << std::endl;
        std::cout << "  总处理批次: " << final_stats.total_batches_processed << std::endl;
        std::cout << "  总输出图像: " << final_stats.total_images_output << std::endl;
        std::cout << "  平均吞吐量: " << std::fixed << std::setprecision(2) 
                  << final_stats.throughput_images_per_second << " 图像/秒" << std::endl;
        std::cout << "  平均批次处理时间: " << final_stats.average_batch_processing_time_ms << " ms" << std::endl;
        
        double efficiency = 0.0;
        if (final_stats.total_images_input > 0) {
            efficiency = (double)final_stats.total_images_output / final_stats.total_images_input * 100.0;
        }
        std::cout << "  处理效率: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        
        LOG_INFO("✅ 批次流水线示例运行完成");
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次流水线示例运行失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
