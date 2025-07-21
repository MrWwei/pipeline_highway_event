#include "object_detection.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

ObjectDetection::ObjectDetection(int num_threads)
    : ImageProcessor(num_threads, "目标检测") {
  // 基类已经完成了初始化工作
}

void ObjectDetection::process_image(ImageDataPtr image, int thread_id) {
  // 调用具体的目标检测算法
  perform_object_detection(image, thread_id);
}

void ObjectDetection::on_processing_start(ImageDataPtr image, int thread_id) {
  // 目标检测开始前的准备工作
  std::cout << "🔄 线程 " << thread_id
            << " 开始目标检测处理: " << image->image_path << std::endl;
}

void ObjectDetection::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  // 目标检测完成后的清理工作
  std::cout << "✅ 线程 " << thread_id
            << " 目标检测处理完成: " << image->image_path << std::endl;
}

void ObjectDetection::perform_object_detection(ImageDataPtr image,
                                               int thread_id) {
  // 检查是否有语义分割结果可以利用
  if (image->segmentation_complete) {
    std::cout << "🎯 线程 " << thread_id
              << " 利用语义分割结果进行目标检测: " << image->image_path
              << std::endl;
  } else {
    std::cout << "⚠️  线程 " << thread_id
              << " 语义分割未完成，使用原始图像进行目标检测: "
              << image->image_path << std::endl;
  }

  // 模拟目标检测算法的耗时处理
  // 这里使用睡眠来模拟复杂的计算过程
  std::this_thread::sleep_for(
      std::chrono::milliseconds(40)); // 模拟40毫秒的处理时间

  // 生成模拟的目标检测结果
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> num_objects_dis(1, 5); // 1-5个目标
  std::uniform_int_distribution<> x_dis(0, image->width - 100);
  std::uniform_int_distribution<> y_dis(0, image->height - 100);
  std::uniform_int_distribution<> size_dis(50, 150);
  std::uniform_real_distribution<> conf_dis(0.5, 0.99);

  std::vector<std::string> class_names = {"car", "truck", "person", "bicycle",
                                          "motorcycle"};
  std::uniform_int_distribution<> class_dis(0, class_names.size() - 1);

  int num_objects = num_objects_dis(gen);
  image->detection_results.clear();

  for (int i = 0; i < num_objects; ++i) {
    ImageData::BoundingBox bbox;
    bbox.x = x_dis(gen);
    bbox.y = y_dis(gen);
    bbox.width = size_dis(gen);
    bbox.height = size_dis(gen);

    // 如果有语义分割结果，可以提高检测置信度
    if (image->segmentation_complete) {
      bbox.confidence = conf_dis(gen) * 1.1; // 提高10%置信度
      if (bbox.confidence > 0.99)
        bbox.confidence = 0.99;
    } else {
      bbox.confidence = conf_dis(gen);
    }

    bbox.class_name = class_names[class_dis(gen)];
    image->detection_results.push_back(bbox);
  }

  image->detection_complete = true;

  std::cout << "🎯 线程 " << thread_id << " 目标检测结果: 检测到 "
            << image->detection_results.size() << " 个目标" << std::endl;
  for (const auto &bbox : image->detection_results) {
    std::cout << "   - " << bbox.class_name << " (置信度: " << bbox.confidence
              << ")" << std::endl;
  }
}
