#include "mask_postprocess.h"
#include "process_mask.h"
#include "event_utils.h"
#include <chrono>
#include <future>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>

MaskPostProcess::MaskPostProcess(int num_threads)
    : ImageProcessor(num_threads, "Mask后处理") {
  // 基类已经完成了初始化工作
}

void MaskPostProcess::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "Error: Invalid image pointer in Mask post-process"
              << std::endl;
    return;
  }

  try {
    // 等待语义分割完成（去除输出）
    // std::cout << "⏳ [线程 " << thread_id << "] 等待语义分割完成..." << std::endl;
    image->segmentation_future.get(); // 阻塞等待语义分割完成
    // std::cout << "✅ [线程 " << thread_id << "] 语义分割已完成" << std::endl;

    // 检查结果是否有效
    if (image->label_map.empty()) {
      std::cerr << "❌ [线程 " << thread_id << "] 语义分割结果无效" << std::endl;
      try {
        if (image->mask_postprocess_promise && 
            image->mask_postprocess_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
          image->mask_postprocess_promise->set_exception(
              std::make_exception_ptr(std::runtime_error("语义分割结果无效")));
        }
      } catch (const std::future_error& e) {
        // std::cout << "⚠️ Promise异常已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
      }
      return;
    }
  } catch (const std::exception &e) {
    std::cerr << "❌ [线程 " << thread_id
              << "] 等待语义分割时发生错误: " << e.what() << std::endl;
    try {
      if (image->mask_postprocess_promise && 
          image->mask_postprocess_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        image->mask_postprocess_promise->set_exception(std::current_exception());
      }
    } catch (const std::future_error& e) {
      // std::cout << "⚠️ Promise异常已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
    }
    return;
  }
  // 语义分割已完成，执行Mask后处理
  perform_mask_postprocess(image, thread_id);
}

void MaskPostProcess::on_processing_start(ImageDataPtr image, int thread_id) {
  // std::cout << "🔍 Mask后处理准备开始 (线程 " << thread_id << ", 帧 " << image->frame_idx << ")" << std::endl;
}

void MaskPostProcess::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  // std::cout << "🔍 Mask后处理完成 (线程 " << thread_id << ", 帧 " << image->frame_idx << ")" << std::endl;
}

void MaskPostProcess::perform_mask_postprocess(ImageDataPtr image,
                                               int thread_id) {
  auto start_time = std::chrono::high_resolution_clock::now();
  cv::Mat mask(image->mask_height, image->mask_width, CV_8UC1);

  // 将label_map数据复制到mask中
  for (int j = 0; j < image->label_map.size(); ++j) {
    mask.data[j] = image->label_map[j];
  }

  // 去除小的白色区域
  // cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

  image->mask = remove_small_white_regions_cuda(mask);
  cv::threshold(image->mask, image->mask, 0, 255, cv::THRESH_BINARY);
  // 获取roi

  // cv::imwrite("mask_output_seg_pre.jpg", mask);
  // cv::imwrite("mask_output_seg_post.jpg", image->mask);
  // exit(0);
  DetectRegion detect_region = crop_detect_region_optimized(
      image->mask, image->mask.rows, image->mask.cols);
  // cv::rectangle(
  //     image->segInResizeMat, cv::Point(detect_region.x1, detect_region.y1),
  //     cv::Point(detect_region.x2, detect_region.y2), cv::Scalar(0, 0, 255),
  //     2);
  //将resize的roi映射回原图大小
  detect_region.x1 = static_cast<int>(detect_region.x1 * image->width /
                                      static_cast<double>(image->mask_width));
  detect_region.x2 = static_cast<int>(detect_region.x2 * image->width /
                                      static_cast<double>(image->mask_width));
  detect_region.y1 = static_cast<int>(detect_region.y1 * image->height /
                                      static_cast<double>(image->mask_height));
  detect_region.y2 = static_cast<int>(detect_region.y2 * image->height /
                                      static_cast<double>(image->mask_height));
  // cv::rectangle(image->imageMat, cv::Point(detect_region.x1,
  // detect_region.y1),
  //               cv::Point(detect_region.x2, detect_region.y2),
  //               cv::Scalar(0, 0, 255), 2);
  image->roi = cv::Rect(detect_region.x1, detect_region.y1,
                        detect_region.x2 - detect_region.x1,
                        detect_region.y2 - detect_region.y1);

  // cv::Mat cropped_image = (image->imageMat)(image->roi);
  // if (!cropped_image.isContinuous()) {
  //   cropped_image = cropped_image.clone();
  // }
  // cv::imwrite("crop_out.jpg", cropped_image);
  // cv::imwrite("src_out.jpg", image->imageMat);
  // exit(0);
  // // 裁剪检测区域
  // cv::Rect roi(detect_region.x1, detect_region.y1,
  //              detect_region.x2 - detect_region.x1,
  //              detect_region.y2 - detect_region.y1);
  // image->roi = roi;
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // 去除Mask后处理完成输出
  // std::cout << "✅ Mask后处理完成 (线程 " << thread_id << ")，耗时: " << duration.count() << "ms" << std::endl;

  // 通知mask后处理完成 - 先检查是否已经设置
  try {
    if (image->mask_postprocess_promise && 
        image->mask_postprocess_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      image->mask_postprocess_promise->set_value();
    }
  } catch (const std::future_error& e) {
    // std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
  }
}
