#pragma once

#include <memory>
#include <string>
#include <vector>

/**
 * 图像数据结构，用于在流水线各阶段之间传递数据
 */
struct ImageData {
  std::string image_path;
  std::vector<uint8_t> raw_data;
  int width;
  int height;
  int channels;

  // 语义分割结果
  std::vector<int> segmentation_mask;
  bool segmentation_complete;

  // 目标检测结果
  struct BoundingBox {
    int x, y, width, height;
    float confidence;
    std::string class_name;
  };
  std::vector<BoundingBox> detection_results;
  bool detection_complete;

  // 构造函数
  ImageData(const std::string &path, int w = 640, int h = 480, int c = 3);

  // 检查是否完全处理完成
  bool is_fully_processed() const;
};

using ImageDataPtr = std::shared_ptr<ImageData>;
