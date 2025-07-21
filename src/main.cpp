#include "pipeline_manager.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main() {
  std::cout << "=== 高速公路事件检测流水线系统 ===" << std::endl;
  std::cout << "🚗 整合语义分割和目标检测的非阻塞流水线处理" << std::endl;
  std::cout << std::endl;

  // 创建流水线管理器 - 配置线程数量
  // 语义分割：3个线程，目标检测：2个线程
  PipelineManager pipeline(5, 2);

  // 显示线程配置信息
  pipeline.print_thread_info();

  // 启动流水线
  pipeline.start();

  // 模拟图像输入
  std::vector<std::string> test_images = {
      "highway_scene_001.jpg", "highway_scene_002.jpg", "highway_scene_003.jpg",
      "highway_scene_004.jpg", "highway_scene_005.jpg"};

  std::cout << "📸 开始添加测试图像到流水线..." << std::endl;

  // 以不同间隔添加图像，模拟实时输入
  for (int i = 0; i < 10000; ++i)
    for (const auto &image_path : test_images) {
      pipeline.add_image(image_path);
      std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 间隔20ms
      pipeline.print_status();
    }

  std::cout << "\n⏰ 等待流水线处理完成..." << std::endl;

  // 监控处理进度
  int processed_count = 0;
  int total_images = test_images.size() * 10; // 修正总数量计算

  while (processed_count < total_images) {
    // 检查是否有完成的结果
    ImageDataPtr result;
    while (pipeline.get_final_result(result)) {
      processed_count++;
      std::cout << "\n🎊 完整处理结果 [" << processed_count << "/"
                << total_images << "]:" << std::endl;
      std::cout << "   图像: " << result->image_path << std::endl;
      std::cout << "   分辨率: " << result->width << "x" << result->height
                << std::endl;
      std::cout << "   语义分割: "
                << (result->segmentation_complete ? "✅ 完成" : "❌ 未完成")
                << std::endl;
      std::cout << "   目标检测: "
                << (result->detection_complete ? "✅ 完成" : "❌ 未完成")
                << std::endl;
      std::cout << "   检测到的目标数量: " << result->detection_results.size()
                << std::endl;

      for (size_t i = 0; i < result->detection_results.size(); ++i) {
        const auto &bbox = result->detection_results[i];
        std::cout << "     " << (i + 1) << ". " << bbox.class_name
                  << " (置信度: " << std::fixed << std::setprecision(2)
                  << bbox.confidence << ", 位置: [" << bbox.x << "," << bbox.y
                  << "," << bbox.width << "x" << bbox.height << "])"
                  << std::endl;
      }
    }

    // 定期打印状态
    static int status_counter = 0;
    if (++status_counter % 50 == 0) { // 每500ms打印一次状态
      pipeline.print_status();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::cout << "\n🎉 所有图像处理完成！" << std::endl;

  // 最终状态报告
  pipeline.print_status();

  // 等待一段时间以观察系统状态
  std::cout << "\n⏱️  等待3秒后停止流水线..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(3));

  // 停止流水线
  pipeline.stop();

  std::cout << "\n✨ 流水线系统演示完成!" << std::endl;
  std::cout << "\n📝 系统特性总结:" << std::endl;
  std::cout << "   ✓ 非阻塞并行处理" << std::endl;
  std::cout << "   ✓ 线程安全的队列管理" << std::endl;
  std::cout << "   ✓ 语义分割和目标检测同步执行" << std::endl;
  std::cout << "   ✓ 智能结果协调和合并" << std::endl;
  std::cout << "   ✓ 实时状态监控" << std::endl;

  return 0;
}
