#include "highway_event.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <random>

class HighwayEventDemo {
private:
    HighwayEventDetector detector_;
    
public:
    /**
     * 基础功能测试 - 单张图片处理
     */
    void test_single_image() {
        std::cout << "\n=== 🖼️  单张图片处理测试 ===" << std::endl;
        
        // 配置参数
        HighwayEventConfig config;
        config.semantic_threads = 4;
        config.mask_threads = 4;
        config.detection_threads = 4;
        config.tracking_threads = 1;
        config.filter_threads = 2;
        config.enable_debug_log = true;
        
        // 初始化并启动
        if (!detector_.initialize(config)) {
            std::cerr << "❌ 初始化失败" << std::endl;
            return;
        }
        
        if (!detector_.start()) {
            std::cerr << "❌ 启动失败" << std::endl;
            return;
        }
        
        // 读取测试图片
        cv::Mat test_image = cv::imread("test.jpg");
        if (test_image.empty()) {
            std::cerr << "❌ 无法读取测试图片 test.jpg" << std::endl;
            detector_.stop();
            return;
        }
        
        std::cout << "📷 图片尺寸: " << test_image.cols << "x" << test_image.rows << std::endl;
        
        // 添加图片到流水线
        auto start_time = std::chrono::high_resolution_clock::now();
        int64_t frame_id = detector_.add_frame(test_image);
        
        if (frame_id < 0) {
            std::cerr << "❌ 添加图片失败" << std::endl;
            detector_.stop();
            return;
        }
        
        std::cout << "📥 图片已添加，帧ID: " << frame_id << std::endl;
        
        // 获取处理结果
        auto result = detector_.get_result_with_timeout(frame_id, 30000); // 30秒超时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "⏱️  处理耗时: " << duration.count() << " ms" << std::endl;
        
        // 分析结果
        switch (result.status) {
        case ResultStatus::SUCCESS:
            std::cout << "✅ 处理成功!" << std::endl;
            if (result.result) {
                std::cout << "   检测到目标数量: " << result.result->track_results.size() << std::endl;
                if (result.result->has_filtered_box) {
                    std::cout << "   筛选目标置信度: " << result.result->filtered_box.confidence << std::endl;
                }
                std::cout << "   语义分割完成！ " << std::endl;
            }
            break;
        case ResultStatus::TIMEOUT:
            std::cout << "⏰ 处理超时" << std::endl;
            break;
        default:
            std::cout << "❌ 处理失败" << std::endl;
            break;
        }
        
        detector_.stop();
    }
    
    /**
     * 批量处理测试 - 多张图片连续处理
     */
    void test_batch_processing() {
        std::cout << "\n=== 📚 批量处理测试 ===" << std::endl;
        
        HighwayEventConfig config;
        config.semantic_threads = 8;
        config.mask_threads = 8;
        config.detection_threads = 8;
        config.tracking_threads = 1;
        config.filter_threads = 4;
        config.input_queue_capacity = 50;
        config.result_queue_capacity = 100;
        
        if (!detector_.initialize(config) || !detector_.start()) {
            std::cerr << "❌ 初始化或启动失败" << std::endl;
            return;
        }
        
        // 创建测试图片（模拟不同尺寸）
        std::vector<cv::Mat> test_images;
        std::vector<cv::Size> sizes = {{1920, 1080}, {1280, 720}, {640, 480}};
        
        for (int i = 0; i < 20; ++i) {
            cv::Size size = sizes[i % sizes.size()];
            cv::Mat img(size.height, size.width, CV_8UC3);
            cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            test_images.push_back(img);
        }
        
        std::cout << "🎯 准备处理 " << test_images.size() << " 张图片" << std::endl;
        
        // 批量添加
        std::vector<int64_t> frame_ids;
        auto add_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < test_images.size(); ++i) {
            int64_t frame_id = detector_.add_frame(test_images[i]);
            if (frame_id >= 0) {
                frame_ids.push_back(frame_id);
                std::cout << "📥 添加图片 " << i << ", 帧ID: " << frame_id << std::endl;
            }
            
            // 每5张图片打印一次状态
            if ((i + 1) % 5 == 0) {
                detector_.print_status();
            }
        }
        
        auto add_end = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - add_start);
        std::cout << "📊 添加完成，耗时: " << add_duration.count() << " ms" << std::endl;
        
        // 批量获取结果
        int success_count = 0;
        auto get_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < frame_ids.size(); ++i) {
            auto result = detector_.get_result_with_timeout(frame_ids[i], 10000);
            if (result.status == ResultStatus::SUCCESS) {
                success_count++;
                std::cout << "✅ 帧 " << frame_ids[i] << " 处理完成" << std::endl;
            } else {
                std::cout << "❌ 帧 " << frame_ids[i] << " 处理失败" << std::endl;
            }
        }
        
        auto get_end = std::chrono::high_resolution_clock::now();
        auto get_duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_end - get_start);
        
        std::cout << "📈 批量处理完成:" << std::endl;
        std::cout << "   成功率: " << (success_count * 100.0 / frame_ids.size()) << "%" << std::endl;
        std::cout << "   获取结果耗时: " << get_duration.count() << " ms" << std::endl;
        std::cout << "   平均处理时间: " << (get_duration.count() / frame_ids.size()) << " ms/帧" << std::endl;
        
        detector_.stop();
    }
    
    /**
     * 视频文件处理测试
     */
    void test_video_processing(const std::string& video_path) {
        std::cout << "\n=== 🎬 视频处理测试 ===" << std::endl;
        
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
        config.semantic_threads = 8;
        config.mask_threads = 8;
        config.detection_threads = 8;
        config.tracking_threads = 1;
        config.filter_threads = 4;
        config.input_queue_capacity = 200;
        config.result_queue_capacity = 500;
        config.enable_status_print = true;
        
        if (!detector_.initialize(config) || !detector_.start()) {
            std::cerr << "❌ 初始化或启动失败" << std::endl;
            cap.release();
            return;
        }
        
        // 处理视频帧
        cv::Mat frame;
        std::vector<int64_t> frame_ids;
        const int max_frames = std::min(1000, frame_count); // 最多处理50帧作为演示
        
        std::cout << "🎬 开始处理视频帧 (最多 " << max_frames << " 帧)..." << std::endl;
        
        auto process_start = std::chrono::high_resolution_clock::now();
        
        // 添加帧
        for (int i = 0; i < max_frames && cap.read(frame); ++i) {
            if (frame.empty()) continue;
            
            int64_t frame_id = detector_.add_frame(frame);
            if (frame_id >= 0) {
                frame_ids.push_back(frame_id);
                
                if ((i + 1) % 10 == 0) {
                    std::cout << "📥 已添加 " << (i + 1) << "/" << max_frames << " 帧" << std::endl;
                    detector_.print_status();
                }
            }
        }
        
        std::cout << "📊 完成添加 " << frame_ids.size() << " 帧" << std::endl;
        
        // 获取结果
        int success_count = 0;
        int total_detections = 0;
        
        for (size_t i = 0; i < frame_ids.size(); ++i) {
            auto result = detector_.get_result_with_timeout(frame_ids[i], 15000);
            
            if (result.status == ResultStatus::SUCCESS) {
                success_count++;
                if (result.result) {
                    total_detections += result.result->track_results.size();
                }
                
                if ((i + 1) % 10 == 0) {
                    std::cout << "📤 已获取 " << (i + 1) << "/" << frame_ids.size() << " 个结果" << std::endl;
                }
            }
        }
        
        auto process_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
        
        std::cout << "\n📈 视频处理统计:" << std::endl;
        std::cout << "   处理帧数: " << frame_ids.size() << std::endl;
        std::cout << "   成功率: " << (success_count * 100.0 / frame_ids.size()) << "%" << std::endl;
        std::cout << "   总处理时间: " << total_duration.count() << " ms" << std::endl;
        std::cout << "   平均处理时间: " << (total_duration.count() / frame_ids.size()) << " ms/帧" << std::endl;
        std::cout << "   检测目标总数: " << total_detections << std::endl;
        std::cout << "   平均检测数: " << (total_detections / (double)success_count) << " 个/帧" << std::endl;
        
        cap.release();
        detector_.stop();
    }
    
    /**
     * 压力测试 - 多线程并发添加
     */
    void test_stress_concurrent() {
        std::cout << "\n=== 💪 压力测试 (多线程并发) ===" << std::endl;
        
        HighwayEventConfig config;
        config.semantic_threads = 8;
        config.mask_threads = 8;
        config.detection_threads = 8;
        config.tracking_threads = 1;
        config.filter_threads = 4;
        config.input_queue_capacity = 500;
        config.result_queue_capacity = 1000;
        
        if (!detector_.initialize(config) || !detector_.start()) {
            std::cerr << "❌ 初始化或启动失败" << std::endl;
            return;
        }
        
        // 创建测试图片
        cv::Mat test_image(720, 1280, CV_8UC3);
        cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        
        const int num_threads = 4;
        const int frames_per_thread = 25;
        const int total_frames = num_threads * frames_per_thread;
        
        std::cout << "🚀 启动 " << num_threads << " 个线程，每个处理 " << frames_per_thread << " 帧" << std::endl;
        
        std::vector<std::future<std::vector<int64_t>>> futures;
        auto stress_start = std::chrono::high_resolution_clock::now();
        
        // 启动多个线程并发添加
        for (int t = 0; t < num_threads; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                std::vector<int64_t> thread_frame_ids;
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> delay_dist(1, 10);
                
                for (int i = 0; i < frames_per_thread; ++i) {
                    // 添加一点随机延迟模拟真实场景
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));
                    
                    int64_t frame_id = detector_.add_frame(test_image);
                    if (frame_id >= 0) {
                        thread_frame_ids.push_back(frame_id);
                        std::cout << "🧵 线程 " << t << " 添加帧 " << frame_id << std::endl;
                    }
                }
                return thread_frame_ids;
            }));
        }
        
        // 收集所有帧ID
        std::vector<int64_t> all_frame_ids;
        for (auto& future : futures) {
            auto thread_frame_ids = future.get();
            all_frame_ids.insert(all_frame_ids.end(), thread_frame_ids.begin(), thread_frame_ids.end());
        }
        
        auto add_end = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - stress_start);
        
        std::cout << "📊 并发添加完成:" << std::endl;
        std::cout << "   预期帧数: " << total_frames << std::endl;
        std::cout << "   实际添加: " << all_frame_ids.size() << std::endl;
        std::cout << "   添加耗时: " << add_duration.count() << " ms" << std::endl;
        
        // 获取所有结果
        int success_count = 0;
        auto get_start = std::chrono::high_resolution_clock::now();
        
        for (auto frame_id : all_frame_ids) {
            auto result = detector_.get_result_with_timeout(frame_id, 20000);
            if (result.status == ResultStatus::SUCCESS) {
                success_count++;
            }
        }
        
        auto get_end = std::chrono::high_resolution_clock::now();
        auto get_duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_end - get_start);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_end - stress_start);
        
        std::cout << "📈 压力测试结果:" << std::endl;
        std::cout << "   成功率: " << (success_count * 100.0 / all_frame_ids.size()) << "%" << std::endl;
        std::cout << "   总耗时: " << total_duration.count() << " ms" << std::endl;
        std::cout << "   吞吐量: " << (all_frame_ids.size() * 1000.0 / total_duration.count()) << " 帧/秒" << std::endl;
        
        detector_.stop();
    }
    
    /**
     * API功能完整性测试
     */
    void test_api_completeness() {
        std::cout << "\n=== 🔧 API功能完整性测试 ===" << std::endl;
        
        HighwayEventConfig config;
        config.enable_debug_log = true;
        
        // 测试初始化前的状态
        std::cout << "📋 测试初始状态..." << std::endl;
        std::cout << "   已初始化: " << (detector_.is_initialized() ? "是" : "否") << std::endl;
        std::cout << "   正在运行: " << (detector_.is_running() ? "是" : "否") << std::endl;
        
        // 测试初始化
        std::cout << "🔧 测试初始化..." << std::endl;
        bool init_success = detector_.initialize(config);
        std::cout << "   初始化结果: " << (init_success ? "成功" : "失败") << std::endl;
        std::cout << "   已初始化: " << (detector_.is_initialized() ? "是" : "否") << std::endl;
        
        if (!init_success) return;
        
        // 测试启动
        std::cout << "🚀 测试启动..." << std::endl;
        bool start_success = detector_.start();
        std::cout << "   启动结果: " << (start_success ? "成功" : "失败") << std::endl;
        std::cout << "   正在运行: " << (detector_.is_running() ? "是" : "否") << std::endl;
        
        if (!start_success) return;
        
        // 测试添加帧的不同方式
        cv::Mat test_image(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
        
        std::cout << "📥 测试不同的添加帧方式..." << std::endl;
        
        // 1. 基础添加
        int64_t frame_id1 = detector_.add_frame(test_image);
        std::cout << "   基础添加: 帧ID=" << frame_id1 << std::endl;
        
        // 2. 移动语义添加
        cv::Mat move_image = test_image.clone();
        int64_t frame_id2 = detector_.add_frame(std::move(move_image));
        std::cout << "   移动添加: 帧ID=" << frame_id2 << std::endl;
        
        // 3. 带超时添加
        int64_t frame_id3 = detector_.add_frame_with_timeout(test_image, 5000);
        std::cout << "   超时添加: 帧ID=" << frame_id3 << std::endl;
        
        // 测试获取结果的不同方式
        std::cout << "📤 测试不同的获取结果方式..." << std::endl;
        
        // 1. 非阻塞获取
        auto try_result = detector_.try_get_result(frame_id1);
        std::cout << "   非阻塞获取: " << (try_result.status == ResultStatus::SUCCESS ? "成功" : "未就绪") << std::endl;
        
        // 2. 带超时获取
        auto timeout_result = detector_.get_result_with_timeout(frame_id1, 10000);
        std::cout << "   超时获取: " << (timeout_result.status == ResultStatus::SUCCESS ? "成功" : "失败") << std::endl;
        
        // 3. 阻塞获取
        auto block_result = detector_.get_result(frame_id2);
        std::cout << "   阻塞获取: " << (block_result.status == ResultStatus::SUCCESS ? "成功" : "失败") << std::endl;
        
        // 测试状态查询
        std::cout << "📊 测试状态查询..." << std::endl;
        std::cout << "   待处理帧数: " << detector_.get_pending_frame_count() << std::endl;
        std::cout << "   完成结果数: " << detector_.get_completed_result_count() << std::endl;
        
        detector_.print_status();
        
        // 测试清理
        std::cout << "🧹 测试结果清理..." << std::endl;
        detector_.cleanup_results_before(frame_id2);
        std::cout << "   清理后完成结果数: " << detector_.get_completed_result_count() << std::endl;
        
        // 测试停止
        std::cout << "🛑 测试停止..." << std::endl;
        bool stop_success = detector_.stop();
        std::cout << "   停止结果: " << (stop_success ? "成功" : "失败") << std::endl;
        std::cout << "   正在运行: " << (detector_.is_running() ? "是" : "否") << std::endl;
        
        std::cout << "✅ API功能测试完成!" << std::endl;
    }
};

void print_usage() {
    std::cout << "用法: ./highway_event_demo [测试类型] [可选参数]" << std::endl;
    std::cout << "\n测试类型:" << std::endl;
    std::cout << "  single     - 单张图片处理测试" << std::endl;
    std::cout << "  batch      - 批量处理测试" << std::endl;
    std::cout << "  video      - 视频处理测试 (需要提供视频文件路径)" << std::endl;
    std::cout << "  stress     - 压力测试 (多线程并发)" << std::endl;
    std::cout << "  api        - API功能完整性测试" << std::endl;
    std::cout << "  all        - 运行所有测试" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  ./highway_event_demo single" << std::endl;
    std::cout << "  ./highway_event_demo video /path/to/video.mp4" << std::endl;
    std::cout << "  ./highway_event_demo all" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🚗 高速公路事件检测系统 Demo 测试程序" << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string test_type = argv[1];
    HighwayEventDemo demo;
    
    try {
        if (test_type == "single") {
            demo.test_single_image();
        }
        else if (test_type == "batch") {
            demo.test_batch_processing();
        }
        else if (test_type == "video") {
            if (argc < 3) {
                std::cerr << "❌ 视频测试需要提供视频文件路径" << std::endl;
                print_usage();
                return 1;
            }
            demo.test_video_processing(argv[2]);
        }
        else if (test_type == "stress") {
            demo.test_stress_concurrent();
        }
        else if (test_type == "api") {
            demo.test_api_completeness();
        }
        else if (test_type == "all") {
            std::cout << "🎯 运行所有测试..." << std::endl;
            demo.test_api_completeness();
            demo.test_single_image();
            demo.test_batch_processing();
            demo.test_stress_concurrent();
            
            // 如果有默认视频文件，也运行视频测试
            std::string default_video = "/home/ubuntu/Desktop/DJI_20250501091406_0001.mp4";
            std::ifstream video_file(default_video);
            if (video_file.good()) {
                demo.test_video_processing(default_video);
            } else {
                std::cout << "⚠️ 未找到默认视频文件，跳过视频测试" << std::endl;
            }
        }
        else {
            std::cerr << "❌ 未知的测试类型: " << test_type << std::endl;
            print_usage();
            return 1;
        }
        
        std::cout << "\n🎉 所有测试完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
