#include "event_determine.h"
#include "pipeline_manager.h"
#include <chrono>
#include <future>
#include <iostream>
#include <limits>


EventDetermine::EventDetermine(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "事件判定") {
  // 使用配置参数，如果没有提供则使用默认值
  if (config) {
    top_fraction_ = config->event_determine_top_fraction;
    bottom_fraction_ = config->event_determine_bottom_fraction;
    times_car_width_ = config->times_car_width; // 车宽倍数
    enable_lane_show_ = config->enable_lane_show;
    lane_show_image_path_ = config->lane_show_image_path;
  } else {
    // 默认配置
    top_fraction_ = 4.0f / 7.0f;
    bottom_fraction_ = 8.0f / 9.0f;
    times_car_width_ = 3.0f; // 默认车宽倍数
  }
  // std::cout << "🔍 事件判定模块初始化完成" << std::endl;
}

EventDetermine::~EventDetermine() {}

void EventDetermine::set_lane_show_enabled(bool enabled, const std::string& save_path) {
  std::lock_guard<std::mutex> lock(lane_show_mutex_);
  enable_lane_show_ = enabled;
  if (!save_path.empty()) {
    lane_show_image_path_ = save_path;
  }
}

void EventDetermine::set_lane_show_interval(int interval) {
  std::lock_guard<std::mutex> lock(lane_show_mutex_);
  if (interval > 0) {
    lane_show_interval_ = interval;
    std::cout << "🎯 车道线绘制间隔已设置为: " << interval << " 帧" << std::endl;
  }
}

void EventDetermine::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "Error: Invalid image data in EventDetermine::process_image" << std::endl;
    return;
  }
  
  perform_event_determination(image, thread_id);
}

void EventDetermine::on_processing_start(ImageDataPtr image, int thread_id) {
  // std::cout << "📦 事件判定准备开始 (线程 " << thread_id << ")" << std::endl;
}
void EventDetermine::change_params(const PipelineConfig &config) {
  top_fraction_ = config.event_determine_top_fraction;
  bottom_fraction_ = config.event_determine_bottom_fraction;
  times_car_width_ = config.times_car_width; // 车宽倍数
  enable_lane_show_ = config.enable_lane_show;
  lane_show_image_path_ = config.lane_show_image_path;
}
void EventDetermine::on_processing_complete(ImageDataPtr image, int thread_id) {
  // std::cout << "📦 事件判定处理完成 (线程 " << thread_id << ")" << std::endl;
}

void EventDetermine::perform_event_determination(ImageDataPtr image, int thread_id) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (image->detection_results.empty()) {
    // 去除无目标框的输出
    // std::cout << "⚠️ 图像 " << image->frame_idx << " 没有检测到目标框" << std::endl;
    image->has_filtered_box = false;
    // 设置promise完成 - 先检查是否已经设置
    try {
      if (image->event_determine_promise && 
          image->event_determine_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        image->event_determine_promise->set_value();
      }
    } catch (const std::future_error& e) {
      // std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // 去除筛选完成输出
    // std::cout << "✅ 事件判定完成 (无目标)，耗时: " << duration.count() << "ms" << std::endl;
    return;
  }
  
  // 使用配置的区域比例
  int image_height = image->height;
  int region_top = image_height * top_fraction_;
  int region_bottom = image_height * bottom_fraction_;
  
  // std::cout << "🎯 筛选区域: [" << region_top << ", " << region_bottom 
  //           << "] (图像高度: " << image_height << ")" << std::endl;
  
  // 首先在指定区域内寻找宽度最小的目标框
  ImageData::BoundingBox* min_width_box = find_min_width_box_in_region(
      image->detection_results, region_top, region_bottom);
  
  if (min_width_box == nullptr) {
    // 指定区域内没有目标框，在全图范围内寻找
    // std::cout << "⚠️ 指定区域内没有目标框，扩展到全图搜索" << std::endl;
    min_width_box = find_min_width_box_in_region(
        image->detection_results, 0, image_height);
  }
  
  if (min_width_box != nullptr) {
    // 找到了宽度最小的目标框，将其保存为筛选结果
    image->filtered_box = *min_width_box;
    image->has_filtered_box = true;
    
    int box_width = calculate_box_width(*min_width_box);
    // 转换到mask的坐标系
    box_width = box_width * image->mask_width / image->width;

    // 根据mask获得车道线
    EmergencyLaneResult eRes = get_Emergency_Lane(image->mask, box_width, min_width_box->bottom, times_car_width_);
    // 将eRes结果转换到原图
    for(auto& point : eRes.left_quarter_points) {
      point.x = static_cast<int>(point.x * image->width / static_cast<double>(image->mask_width));
      point.y = static_cast<int>(point.y * image->height / static_cast<double>(image->mask_height));
    }
    for(auto& point : eRes.right_quarter_points) {
      point.x = static_cast<int>(point.x * image->width / static_cast<double>(image->mask_width));
      point.y = static_cast<int>(point.y * image->height / static_cast<double>(image->mask_height));
    }
    for(auto& point : eRes.left_lane_region) {
      point.x = static_cast<int>(point.x * image->width / static_cast<double>(image->mask_width));
      point.y = static_cast<int>(point.y * image->height / static_cast<double>(image->mask_height));
    }
    for(auto& point : eRes.right_lane_region) {
      point.x = static_cast<int>(point.x * image->width / static_cast<double>(image->mask_width));
      point.y = static_cast<int>(point.y * image->height / static_cast<double>(image->mask_height));
    }
    for(auto& point : eRes.middle_lane_region) {
      point.x = static_cast<int>(point.x * image->width / static_cast<double>(image->mask_width));
      point.y = static_cast<int>(point.y * image->height / static_cast<double>(image->mask_height));
    } 
    // 判断车辆是否在应急车道内
    for(auto &track_box:image->track_results) {
      track_box.status = determineObjectStatus(track_box, eRes);
    }
    
    // 线程安全地访问车道线显示相关变量
    std::lock_guard<std::mutex> lock(lane_show_mutex_);
    
    // 增加帧计数器
    frame_counter_++;
    
    // 检查是否需要绘制车道线（手动启用或每200帧自动绘制一次）
    bool should_draw_lane = enable_lane_show_ || (frame_counter_ % lane_show_interval_ == 0);
    
    if(should_draw_lane && !lane_show_image_path_.empty()) {
      // 绘制车道线结果
      cv::Mat show_mat = image->imageMat.clone();
      drawEmergencyLaneQuarterPoints(show_mat, eRes);
      // 保存车道线结果图像
      std::string filename = lane_show_image_path_ + "/" + std::to_string(image->frame_idx) + ".jpg";
      cv::imwrite(filename, show_mat);
      
      // 如果是自动绘制（每200帧），输出提示信息
      if (!enable_lane_show_) {
        std::cout << "🎨 自动绘制车道线结果 (第" << frame_counter_ << "帧): " << filename << std::endl;
      }
    }    // drawEmergencyLaneQuarterPoints(image->imageMat, eRes);
    // cv::imwrite("mask_" + std::to_string(image->frame_idx) + ".jpg", image->imageMat);
    // 绘制到原图



    // std::cout << "✅ 找到宽度最小的目标框: [" 
    //           << min_width_box->left << ", " << min_width_box->top 
    //           << ", " << min_width_box->right << ", " << min_width_box->bottom 
    //           << "] 宽度: " << box_width << "px" << std::endl;
  } else {
    // 全图范围内也没有目标框
    image->has_filtered_box = false;
    // std::cout << "⚠️ 全图范围内都没有找到目标框" << std::endl;
  }
  
  // 设置promise完成 - 先检查是否已经设置
  try {
    if (image->event_determine_promise && 
        image->event_determine_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      image->event_determine_promise->set_value();
    }
  } catch (const std::future_error& e) {
    // std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  // std::cout << "✅ 事件判定完成，耗时: " << duration.count() << "ms" << std::endl;
}

int EventDetermine::calculate_box_width(const ImageData::BoundingBox& box) const {
  return box.right - box.left;
}

bool EventDetermine::is_box_in_region(const ImageData::BoundingBox& box, 
                                 int region_top, int region_bottom) const {
  // 检查目标框的中心点是否在指定区域内
  int box_center_y = (box.top + box.bottom) / 2;
  return box_center_y >= region_top && box_center_y <= region_bottom;
}

ImageData::BoundingBox* EventDetermine::find_min_width_box_in_region(
    const std::vector<ImageData::BoundingBox>& boxes,
    int region_top, int region_bottom) const {
  
  ImageData::BoundingBox* min_width_box = nullptr;
  int min_width = std::numeric_limits<int>::max();
  
  // 遍历所有目标框，找到指定区域内宽度最小的
  for (auto& box : boxes) {
    if (is_box_in_region(box, region_top, region_bottom)) {
      int width = calculate_box_width(box);
      if (width < min_width) {
        min_width = width;
        // 注意：这里需要进行const_cast，因为我们需要返回非const指针
        min_width_box = const_cast<ImageData::BoundingBox*>(&box);
      }
    }
  }
  
  return min_width_box;
}

void
  EventDetermine::drawEmergencyLaneQuarterPoints(cv::Mat &image,
                                 const EmergencyLaneResult &emergency_lane) {
    if (!emergency_lane.is_valid) {
      return;
    }

    // 绘制左车道四分之一点
    if (!emergency_lane.left_quarter_points.empty()) {
      for (const auto &point : emergency_lane.left_quarter_points) {
        cv::circle(image, cv::Point(point.x, point.y), 3, cv::Scalar(0, 255, 0),
                   -1); // 绿色圆点
      }
    }

    // 绘制右车道四分之一点
    if (!emergency_lane.right_quarter_points.empty()) {
      for (const auto &point : emergency_lane.right_quarter_points) {
        cv::circle(image, cv::Point(point.x, point.y), 3, cv::Scalar(0, 0, 255),
                   -1); // 红色圆点
      }
    }

    // 可选：绘制应急车道区域边界
    if (!emergency_lane.left_lane_region.empty()) {
      std::vector<cv::Point> left_contour;
      for (const auto &pt : emergency_lane.left_lane_region) {
        left_contour.emplace_back(pt.x, pt.y);
      }
      cv::polylines(image, left_contour, true, cv::Scalar(255, 255, 0),
                    2); // 青色线条
    }

    if (!emergency_lane.right_lane_region.empty()) {
      std::vector<cv::Point> right_contour;
      for (const auto &pt : emergency_lane.right_lane_region) {
        right_contour.emplace_back(pt.x, pt.y);
      }
      cv::polylines(image, right_contour, true, cv::Scalar(255, 0, 255),
                    2); // 紫色线条
    }
  }

  ObjectStatus
  EventDetermine::determineObjectStatus(const ImageData::BoundingBox &box,
                        const EmergencyLaneResult &emergency_lane) {
    if (!emergency_lane.is_valid) {
      return ObjectStatus::NORMAL;
    }
    // 检查目标框的中心点是否在应急车道区域内
    PointT center((box.left + box.right) / 2, (box.top + box.bottom) / 2);
    // 判断点是否在应急车道区域内
    auto is_in_region = [](const std::vector<PointT> &region, const PointT &pt) {
      if (region.size() < 3)
        return false;

      std::vector<cv::Point> contour;
      for (const auto &p : region) {
        contour.emplace_back(p.x, p.y);
      }
      return cv::pointPolygonTest(contour, cv::Point2f(pt.x, pt.y), false) >= 0;
    };

    if (is_in_region(emergency_lane.left_lane_region, center) ||
        is_in_region(emergency_lane.right_lane_region, center)) {
      return ObjectStatus::OCCUPY_EMERGENCY_LANE;
    }

    return ObjectStatus::NORMAL;
  }
