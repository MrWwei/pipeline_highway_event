#include "image_data.h"
#include <iostream>
#include <random>

// 析构函数
ImageData::~ImageData() {
  // cv::Mat objects will be automatically destroyed by their destructors
}

// 检查是否完全处理完成
bool ImageData::is_fully_processed() const {
  // 简化版本：基于数据本身判断是否处理完成
  return !track_results.empty() || !detection_results.empty();
}