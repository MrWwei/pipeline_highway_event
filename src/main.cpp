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
  // 语义分割：8个线程，Mask后处理：20个线程，目标检测：8个线程，目标跟踪：1个线程，目标框筛选：4个线程
  PipelineManager pipeline(8, 8, 8, 1, 4);

  // 启动流水线
  pipeline.start();

  // 记录处理开始时间

  // 监控处理进度
  int processed_count = 0;
  int total_images = 300; // 限制处理300帧
  std::atomic<bool> result_thread_running(true);
  auto start_time = std::chrono::high_resolution_clock::now();

  // 创建结果处理线程
  std::thread result_thread([&pipeline, &processed_count, &total_images,
                             &result_thread_running, start_time]() {
    std::cout << "结果处理线程已启动" << std::endl;
    while (result_thread_running.load()) {
      if (processed_count >= total_images && pipeline.get_result_queue_size() == 0) {
        std::cout << "结果处理线程检测到退出条件" << std::endl;
        break;  // 如果所有帧都处理完且结果队列为空，则退出
      }
      
      std::cout << "\r🔄 处理进度: " << processed_count << "/" << total_images
                << " 帧" << std::flush;
      // 检查是否有完成的结果
      ImageDataPtr result;
      bool has_result = false;

            // 尝试获取一个结果
      if (pipeline.get_final_result(result)) {
        // for(auto &box : result->track_results) {
        //   // 在图像上绘制检测框
          
        //   int track_id = box.track_id;
        //   cv::Scalar color;
        //   // Draw track ID above the bounding box
        //   if(box.status == ObjectStatus::OCCUPY_EMERGENCY_LANE) {
        //     color = cv::Scalar(0, 0, 255); // 红色
        //   } else {
        //     color = cv::Scalar(0, 255, 0); // 绿色
        //   }
          
        //   cv::rectangle(*result->imageMat, cv::Point(box.left, box.top),
        //                 cv::Point(box.right, box.bottom), color, 2);}

        // 显示筛选出的目标框信息
        // if (result->has_filtered_box) {

        //   cv::rectangle(*result->imageMat,
        //                 cv::Point(result->filtered_box.left,
        //                           result->filtered_box.top),
        //                 cv::Point(result->filtered_box.right,
        //                           result->filtered_box.bottom),
        //                 cv::Scalar(0, 0, 255), 2);
        // }
          
        //   // 计算筛选区域
        //   int region_top = result->height * 2 / 7;
        //   int region_bottom = result->height * 6 / 7;
        //   bool in_target_region = (box_center_y >= region_top && box_center_y <= region_bottom);
          
        //   std::cout << "🎯 筛选结果 - 帧 " << result->frame_idx << ":" << std::endl;
        //   std::cout << "   目标框: [" 
        //             << result->filtered_box.left << ", " << result->filtered_box.top 
        //             << ", " << result->filtered_box.right << ", " << result->filtered_box.bottom 
        //             << "]" << std::endl;
        //   std::cout << "   尺寸: " << box_width << "x" << box_height << "px" 
        //             << " (宽度: " << box_width << "px)" << std::endl;
        //   std::cout << "   位置: " << (in_target_region ? "目标区域内" : "全图范围内") 
        //             << " (中心Y: " << box_center_y << ")" << std::endl;
        //   std::cout << "   置信度: " << std::fixed << std::setprecision(3) 
        //             << result->filtered_box.confidence << std::endl;
        // } else {
        //   std::cout << "⚠️ 帧 " << result->frame_idx << " - 未找到符合条件的目标框" << std::endl;
        // }
        
        // for (auto box : result->track_results) {
        //   // 在图像上绘制检测框
        //   cv::rectangle(*result->imageMat, cv::Point(box.left, box.top),
        //                 cv::Point(box.right, box.bottom), cv::Scalar(0, 255,
        //                 0), 2);
        //   int track_id = box.track_id;
        //   // Draw track ID above the bounding box
        //   cv::putText(*result->imageMat, "ID: " + std::to_string(track_id),
        //               cv::Point(box.left, box.top - 10),
        //               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),
        //               2);
        // }

        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     end_time - start_time);

        // // 保存处理后的帧 - 使用原始帧序号命名
        // std::string output_filename =
        //     "outs/output_frame_" + std::to_string(result->frame_idx) +
        //     ".jpg";
        // cv::imwrite(output_filename, *result->imageMat);

        has_result = true;
        processed_count++;
        // std::cout << "✅ 处理第 " << result->frame_idx
        //           << " 帧，耗时: " << duration.count() << "ms" << std::endl;
      } else {
        // get_final_result 返回 false，说明队列已关闭，准备退出
        if (!result_thread_running.load()) {
          std::cout << "\n结果处理线程收到停止信号，准备退出" << std::endl;
          break;
        }
      }

      if (!has_result) {
        // 没有结果时短暂休息，避免忙等待
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
      }
    }
    std::cout << "结果处理线程即将退出" << std::endl;
  });
  // 打开视频文件
  cv::VideoCapture cap(
      "/home/ubuntu/Desktop/DJI_20250501091406_0001.mp4"); // 替换为你的视频文件路径
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
  
  // 显示流水线配置信息
  std::cout << "\n🔧 流水线配置:" << std::endl;
  std::cout << "   语义分割: 8 线程" << std::endl;
  std::cout << "   Mask后处理: 8 线程" << std::endl;
  std::cout << "   目标检测: 8 线程" << std::endl;
  std::cout << "   目标跟踪: 1 线程" << std::endl;
  std::cout << "   目标框筛选: 4 线程" << std::endl;
  std::cout << "   处理帧数限制: 300 帧" << std::endl;
  std::cout << "   流水线阶段: 语义分割 → Mask后处理 → 目标检测 → 目标跟踪 → 目标框筛选 → 最终结果" << std::endl;
  std::cout << "   筛选条件: 图像 2/7~6/7 区域内宽度最小的目标框（无则全图搜索）" << std::endl;
  
  auto total_start_time = std::chrono::high_resolution_clock::now();

  // 定义状态打印函数
  auto print_pipeline_status = [&pipeline]() {
    std::cout << "\n📊 Pipeline Status:" << std::endl;
    std::cout << "   语义分割队列: " << pipeline.get_seg_queue_size() << " 帧" << std::endl;
    std::cout << "   Mask后处理队列: " << pipeline.get_mask_queue_size() << " 帧" << std::endl;
    std::cout << "   目标检测队列: " << pipeline.get_det_queue_size() << " 帧" << std::endl;
    std::cout << "   目标跟踪队列: " << pipeline.get_track_queue_size() << " 帧" << std::endl;
    std::cout << "   目标框筛选队列: " << pipeline.get_filter_queue_size() << " 帧" << std::endl;
    std::cout << "   结果队列: " << pipeline.get_result_queue_size() << " 帧" << std::endl;
  };

  // 逐帧读取并处理
  cv::Mat frame;
  int input_frame_count = 0;
  auto last_status_time = std::chrono::steady_clock::now();
  
  while (cap.read(frame)) {
    if (frame.empty()) {
      std::cerr << "Error: 空帧" << std::endl;
      continue;
    }

    // 优化：减少内存分配开销
    static uint64_t frame_idx = 0; // 静态变量记录帧序号
    
    // 直接在堆上分配，避免临时对象
    cv::Mat *frame_ptr = new cv::Mat(frame.rows, frame.cols, frame.type());
    frame.copyTo(*frame_ptr); // 必要的拷贝，但优化了分配过程

    // 创建并初始化图像数据
    ImageDataPtr img_data = std::make_shared<ImageData>(frame_ptr);
    img_data->frame_idx = frame_idx++; // 设置并递增帧序号
    // 去除主线程输入打印
    pipeline.add_image(img_data);

    input_frame_count++;
    if(input_frame_count > 300)break;

    // 每隔1秒或每10帧打印一次状态（减少清屏频率）
    auto current_time = std::chrono::steady_clock::now();
    if (current_time - last_status_time > std::chrono::seconds(5)) {
      print_pipeline_status();
      std::cout << "总进度 - 已输入: " << input_frame_count << " 帧, 已处理: " << processed_count << " 帧" << std::endl;
      last_status_time = current_time;
    }

    // 按原始视频帧率控制处理速度
    // std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  }
  
  std::cout << "📥 完成输入 " << input_frame_count << " 帧，等待处理完成..." << std::endl;

  // 主线程等待所有图像处理完成
  while (processed_count < input_frame_count) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 定期显示等待状态
    static auto last_wait_print = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - last_wait_print > std::chrono::seconds(2)) {
      print_pipeline_status();
      std::cout << "⏳ 等待处理完成: " << processed_count << "/" << input_frame_count << " 帧" << std::endl;
      last_wait_print = now;
    }
  }
  
  std::cout << "✅ 所有帧处理完成！" << std::endl;

  // 等待所有结果被处理
  while (pipeline.get_result_queue_size() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "\r⏳ 等待最后 " << pipeline.get_result_queue_size() << " 帧结果处理完成..." << std::flush;
  }
  std::cout << std::endl;

  // 停止结果处理线程
  std::cout << "正在停止结果处理线程..." << std::endl;
  result_thread_running.store(false);
  std::cout << "已设置停止标志，等待线程退出..." << std::endl;
  
  // 添加超时机制
  auto join_start = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);
  
  if (result_thread.joinable()) {
    // 用一个循环来检查线程是否还在运行
    while (std::chrono::steady_clock::now() - join_start < timeout) {
      if (!result_thread.joinable()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      std::cout << "." << std::flush;
    }
    
    if (result_thread.joinable()) {
      std::cout << "\n强制结束结果处理线程..." << std::endl;
      result_thread.detach();  // 如果超时，就分离线程
    } else {
      std::cout << "\n结果处理线程已正常退出" << std::endl;
    }
  }

  std::cout << "正在关闭流水线..." << std::endl;
  // 关闭视频和停止流水线
  cap.release();
  pipeline.stop();
  std::cout << "流水线已停止" << std::endl;

  return 0;
}
