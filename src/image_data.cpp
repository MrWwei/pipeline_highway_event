#include "image_data.h"
#include <iostream>
#include <random>

// 析构函数
ImageData::~ImageData() {
  // 只有在future还没有ready的情况下才设置promise，避免重复设置
  try {
    if (segmentation_promise && 
        segmentation_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      segmentation_promise->set_value();
    }
    if (mask_postprocess_promise && 
        mask_postprocess_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      mask_postprocess_promise->set_value();
    }
    if (detection_promise && 
        detection_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      detection_promise->set_value();
    }
    if (box_filter_promise && 
        box_filter_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      box_filter_promise->set_value();
    }
    if (tracking_promise && 
        tracking_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      tracking_promise->set_value();
    }
  } catch (const std::future_error &) {
    // 忽略promise已经被设置的错误
  }

  // cv::Mat objects will be automatically destroyed by their destructors
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