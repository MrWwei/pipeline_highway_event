#include "image_data.h"
#include <iostream>
#include <random>

ImageData::ImageData(const std::string &path, int w, int h, int c)
    : image_path(path), width(w), height(h), channels(c),
      segmentation_complete(false), detection_complete(false) {

  // 模拟加载图像数据
  raw_data.resize(width * height * channels);

  // 用随机数据填充（模拟真实图像数据）
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (auto &pixel : raw_data) {
    pixel = static_cast<uint8_t>(dis(gen));
  }

  std::cout << "✓ 图像数据加载完成: " << image_path << " (" << width << "x"
            << height << "x" << channels << ")" << std::endl;
}

bool ImageData::is_fully_processed() const {
  return segmentation_complete && detection_complete;
}
