#include "pipeline_manager.h"
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

int main() {
  // 创建流水线管理器 - 配置线程数量
  // 语义分割：1个线程，Mask后处理：10个线程，目标检测：2个线程
  PipelineManager pipeline(8, 20, 8);

  // 启动流水线
  pipeline.start();

  // 记录处理开始时间

  // 监控处理进度
  int processed_count = 0;
  int total_images = 0; // 将在读取视频信息后更新
  std::atomic<bool> result_thread_running(true);
  auto start_time = std::chrono::high_resolution_clock::now();

  // 创建结果处理线程
  std::thread result_thread([&pipeline, &processed_count, &total_images,
                             &result_thread_running, start_time]() {
    while (result_thread_running.load() || processed_count < total_images) {
      // 检查是否有完成的结果
      ImageDataPtr result;
      bool has_result = false;

      // 尝试获取一个结果
      if (pipeline.get_final_result(result)) {

        for (auto box : result->detection_results) {
          // 在图像上绘制检测框
          cv::rectangle(*result->imageMat, cv::Point(box.left, box.top),
                        cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0),
                        2);
          int track_id = box.track_id;
          // Draw track ID above the bounding box
          cv::putText(*result->imageMat, "ID: " + std::to_string(track_id),
                      cv::Point(box.left, box.top - 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        // 保存处理后的帧 - 使用原始帧序号命名
        std::string output_filename =
            "outs/output_frame_" + std::to_string(result->frame_idx) + ".jpg";
        cv::imwrite(output_filename, *result->imageMat);

        has_result = true;
        processed_count++;
        std::cout << "✅ 处理第 " << result->frame_idx
                  << " 帧，耗时: " << duration.count() << "ms" << std::endl;
      }

      if (!has_result) {
        // 没有结果时短暂休息，避免忙等待
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
      }
    }
  });
  // 打开视频文件
  cv::VideoCapture cap(
      "/home/ubuntu/Desktop/DJI_20250501091754_0003_V.MP4"); // 替换为你的视频文件路径
  if (!cap.isOpened()) {
    std::cerr << "Error: 无法打开视频文件" << std::endl;
    return -1;
  }

  // 获取视频信息
  double fps = cap.get(cv::CAP_PROP_FPS);
  int delay = static_cast<int>(1000.0 / fps); // 根据视频帧率计算延迟
  int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  total_images = frame_count; // 更新总帧数

  std::cout << "视频信息:" << std::endl;
  std::cout << "FPS: " << fps << std::endl;
  std::cout << "总帧数: " << frame_count << std::endl;
  auto total_start_time = std::chrono::high_resolution_clock::now();

  // 逐帧读取并处理
  cv::Mat frame;
  while (cap.read(frame)) {
    if (frame.empty()) {
      std::cerr << "Error: 空帧" << std::endl;
      continue;
    }

    // 创建新的Mat对象（深拷贝）并添加到流水线
    static uint64_t frame_idx = 0; // 静态变量记录帧序号
    cv::Mat *frame_copy = new cv::Mat(frame.clone());

    // 创建并初始化图像数据
    ImageDataPtr img_data = std::make_shared<ImageData>(frame_copy);
    img_data->frame_idx = frame_idx++; // 设置并递增帧序号
    pipeline.add_image(img_data);

    // 按原始视频帧率控制处理速度
    // std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    // 计算总处理时间
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_start_time);
    // 打印处理状态
    pipeline.print_status();
    std::cout << "\n========== 视频处理统计 ==========" << std::endl;
    std::cout << "总处理帧数: " << processed_count << " 帧" << std::endl;
    std::cout << "总处理时间: " << total_duration.count() / 1000.0 << " 秒"
              << std::endl;
    std::cout << "平均每帧耗时: "
              << total_duration.count() / (double)processed_count << " 毫秒"
              << std::endl;
    std::cout << "实际处理帧率: "
              << (processed_count * 1000.0) / total_duration.count() << " FPS"
              << std::endl;
    std::cout << "================================" << std::endl;
  }

  // 主线程等待所有图像处理完成
  while (processed_count < total_images) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // 停止结果处理线程
  result_thread_running.store(false);
  // 等待结果处理线程完成
  if (result_thread.joinable()) {
    result_thread.join();
  }

  // 关闭视频和停止流水线
  cap.release();
  pipeline.stop();

  return 0;
}
