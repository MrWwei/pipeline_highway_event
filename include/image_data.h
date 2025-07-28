#pragma once

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "event_type.h"

/**
 * 图像数据结构，用于在流水线各阶段之间传递数据
 */
struct ImageData {
  cv::Mat imageMat;
  cv::Mat segInResizeMat;
  int width;
  int height;
  int channels;
  uint64_t frame_idx; // 添加帧序号，用于保证处理顺序

  // 语义分割结果
  int mask_height;
  int mask_width;
  std::vector<uint8_t> label_map;
  std::shared_ptr<std::promise<void>> segmentation_promise;
  std::shared_future<void> segmentation_future;
  cv::Mat mask; // 用于存储Mask后处理的结果 resize后的mask 1024x1024

  // Mask后处理结果
  std::shared_ptr<std::promise<void>> mask_postprocess_promise;
  std::shared_future<void> mask_postprocess_future;

  // 裁剪后的ROI
  cv::Rect roi;

  // 目标检测结果
  struct BoundingBox {
    int left, top, right, bottom;
    float confidence;
    int class_id;
    int track_id;
    ObjectStatus status; // 目标状态
  };
  std::vector<BoundingBox> detection_results;
  std::vector<BoundingBox> track_results;
  std::shared_ptr<std::promise<void>> detection_promise;
  std::shared_future<void> detection_future;

  // 目标框筛选结果
  BoundingBox filtered_box;  // 筛选出的宽度最小的目标框
  bool has_filtered_box;     // 是否有筛选结果
  std::shared_ptr<std::promise<void>> event_determine_promise;
  std::shared_future<void> event_determine_future;

  // 目标跟踪阶段的同步原语
  std::shared_ptr<std::promise<void>> tracking_promise;
  std::shared_future<void> tracking_future;

  // 默认构造函数
  ImageData()
      : width(0), height(0),
        channels(0), frame_idx(0), mask_height(0), mask_width(0), 
        has_filtered_box(false) {
    segmentation_promise = std::make_shared<std::promise<void>>();
    segmentation_future = segmentation_promise->get_future();
    mask_postprocess_promise = std::make_shared<std::promise<void>>();
    mask_postprocess_future = mask_postprocess_promise->get_future();
    detection_promise = std::make_shared<std::promise<void>>();
    detection_future = detection_promise->get_future();
    event_determine_promise = std::make_shared<std::promise<void>>();
    event_determine_future = event_determine_promise->get_future();
    tracking_promise = std::make_shared<std::promise<void>>();
    tracking_future = tracking_promise->get_future();
  }

  // 带图像的构造函数
  ImageData(const cv::Mat& img) : ImageData() {
    imageMat = img.clone();
    width = img.cols;
    height = img.rows;
    channels = img.channels();
    
    // 内存优化：预分配常用缓冲区
    label_map.reserve(1024 * 1024); // 预留分割结果空间
    detection_results.reserve(100);  // 预留检测结果空间
    track_results.reserve(100);      // 预留跟踪结果空间
  }

  // 移动构造函数
  ImageData(cv::Mat&& img) : ImageData() {
    imageMat = std::move(img);
    width = imageMat.cols;
    height = imageMat.rows;
    channels = imageMat.channels();
    
    // 内存优化：预分配常用缓冲区
    label_map.reserve(1024 * 1024); // 预留分割结果空间
    detection_results.reserve(100);  // 预留检测结果空间
    track_results.reserve(100);      // 预留跟踪结果空间
  }

  // 析构函数
  ~ImageData();

  // 检查是否完全处理完成
  bool is_fully_processed() const;
};

using ImageDataPtr = std::shared_ptr<ImageData>;
