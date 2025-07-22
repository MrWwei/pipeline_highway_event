#include "image_data.h"
#include <iostream>
#include <random>

// 析构函数
ImageData::~ImageData() {
  // 确保所有promise都被设置，防止有等待的future
  try {
    if (segmentation_promise) {
      segmentation_promise->set_value();
    }
    if (mask_postprocess_promise) {
      mask_postprocess_promise->set_value();
    }
    if (detection_promise) {
      detection_promise->set_value();
    }
  } catch (const std::future_error &) {
    // 忽略promise已经被设置的错误
  }

  if (imageMat) {
    delete imageMat;
    imageMat = nullptr;
  }
  if (segInResizeMat) {
    delete segInResizeMat;
    segInResizeMat = nullptr;
  }
}

// 检查是否完全处理完成
bool ImageData::is_fully_processed() const {
  return segmentation_future.wait_for(std::chrono::seconds(0)) ==
             std::future_status::ready &&
         mask_postprocess_future.wait_for(std::chrono::seconds(0)) ==
             std::future_status::ready &&
         detection_future.wait_for(std::chrono::seconds(0)) ==
             std::future_status::ready;
}