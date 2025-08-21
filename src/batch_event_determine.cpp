#include "batch_event_determine.h"
#include "logger_manager.h"
#include <iostream>
#include <algorithm>
// #include <execution>
#include <opencv2/imgproc.hpp>
#include <cmath>

BatchEventDetermine::BatchEventDetermine(int num_threads, const PipelineConfig* config)
    : num_threads_(num_threads), running_(false), stop_requested_(false) { // 默认水平向右为正常方向
    
    // 初始化配置
    if (config) {
        config_ = *config;
        top_fraction_ = config->event_determine_top_fraction;
        bottom_fraction_ = config->event_determine_bottom_fraction;
        times_car_width_ = config->times_car_width;
        lane_show_image_path_ = config->lane_show_image_path;
    }
    
    // 创建输入输出连接器
    input_connector_ = std::make_unique<BatchConnector>(10);
    output_connector_ = std::make_unique<BatchConnector>(10);
    
}

BatchEventDetermine::~BatchEventDetermine() {
    stop();
}

void BatchEventDetermine::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    stop_requested_.store(false);
    
    // 启动连接器
    input_connector_->start();
    output_connector_->start();
    
    // 启动工作线程
    worker_threads_.clear();
    worker_threads_.reserve(num_threads_);
    
    for (int i = 0; i < num_threads_; ++i) {
        worker_threads_.emplace_back(&BatchEventDetermine::worker_thread_func, this);
    }
    
    std::cout << "✅ 批次事件判定已启动，使用 " << num_threads_ << " 个线程" << std::endl;
}

void BatchEventDetermine::stop() {
    if (!running_.load()) {
        return;
    }
    
    stop_requested_.store(true);
    running_.store(false);
    
    // 停止连接器
    input_connector_->stop();
    output_connector_->stop();
    
    // 等待所有工作线程结束
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    LOG_INFO("🛑 批次事件判定已停止");
}

bool BatchEventDetermine::add_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    return input_connector_->send_batch(batch);
}

bool BatchEventDetermine::get_processed_batch(BatchPtr& batch) {
    return output_connector_->receive_batch(batch);
}

bool BatchEventDetermine::process_batch(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // std::cout << "⚠️ 开始处理批次 " << batch->batch_id 
    //           << " 事件判定，包含 " << batch->actual_size << " 个图像" << std::endl;
    
    try {
        // 确保图像按帧序号排序
        std::sort(batch->images.begin(), batch->images.begin() + batch->actual_size,
                  [](const ImageDataPtr& a, const ImageDataPtr& b) {
                      return a->frame_idx < b->frame_idx;
                  });
        
        // 使用批次处理锁确保事件数据一致性
        std::lock_guard<std::mutex> batch_lock(batch_processing_mutex_);
        
        for(auto & image : batch->images) {
            if (!image) {
                continue;
            }
            // 执行事件判定
            perform_event_determination(image);
            
        }
        
        // 标记批次完成
        batch->event_completed.store(true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 更新统计信息
        processed_batch_count_.fetch_add(1);
        total_processing_time_ms_.fetch_add(duration.count());
        total_images_processed_.fetch_add(batch->actual_size);
        
        // 统计检测到的事件数量
        uint64_t events_in_batch = 0;
        for (size_t i = 0; i < batch->actual_size; ++i) {
            // 这里需要根据实际的事件存储结构来统计
            // 暂时简化处理
        }
        total_events_detected_.fetch_add(events_in_batch);
        
        // std::cout << "✅ 批次 " << batch->batch_id << " 事件判定完成，耗时: " 
        //           << duration.count() << "ms，检测到 " << events_in_batch << " 个事件" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次 " << batch->batch_id << " 事件判定异常: " << e.what() << std::endl;
        return false;
    }
}

void BatchEventDetermine::worker_thread_func() {
    while (running_.load()) {
        BatchPtr batch;
        
        // 从输入连接器获取批次
        if (input_connector_->receive_batch(batch)) {
            if (batch) {
                // 处理批次
                bool success = process_batch(batch);
                
                if (success) {
                    // 发送到输出连接器
                    output_connector_->send_batch(batch);
                } else {
                    std::cerr << "❌ 批次 " << batch->batch_id << " 事件判定失败，丢弃" << std::endl;
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
}

// BatchStage接口实现
std::string BatchEventDetermine::get_stage_name() const {
    return "批次事件判定";
}

size_t BatchEventDetermine::get_processed_count() const {
    return processed_batch_count_.load();
}

double BatchEventDetermine::get_average_processing_time() const {
    auto count = processed_batch_count_.load();
    if (count == 0) return 0.0;
    return (double)total_processing_time_ms_.load() / count;
}

size_t BatchEventDetermine::get_queue_size() const {
    return input_connector_->get_queue_size();
}

void BatchEventDetermine::perform_event_determination(ImageDataPtr image) {
  
  if (image->detection_results.empty()) {
    image->has_filtered_box = false;
    return;
  }
  
  // 使用配置的区域比例
  int image_height = image->height;
  int region_top = image_height * top_fraction_;
  int region_bottom = image_height * bottom_fraction_;
  
  // 首先在指定区域内寻找宽度最小的目标框
  ImageData::BoundingBox* min_width_box = find_min_width_box_in_region(
      image->detection_results, region_top, region_bottom);
  
  if (min_width_box == nullptr) {
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
   
    // lane_show_image_path_ = "lane_results";
    if(image->frame_idx % 200 == 0 && !lane_show_image_path_.empty()) {
      // 绘制车道线结果
      cv::Mat show_mat = image->imageMat.clone();
      drawEmergencyLaneQuarterPoints(show_mat, eRes);
      // 保存车道线结果图像
      std::string filename = lane_show_image_path_ + "/" + std::to_string(image->frame_idx) + ".jpg";
      cv::imwrite(filename, show_mat);
      
     
    }    

  } else {
    // 全图范围内也没有目标框
    image->has_filtered_box = false;
    // LOG_INFO("⚠️ 全图范围内都没有找到目标框");
  }
  
}

int BatchEventDetermine::calculate_box_width(const ImageData::BoundingBox& box) const {
  return box.right - box.left;
}

bool BatchEventDetermine::is_box_in_region(const ImageData::BoundingBox& box, 
                                 int region_top, int region_bottom) const {
  // 检查目标框的中心点是否在指定区域内
  int box_center_y = (box.top + box.bottom) / 2;
  return box_center_y >= region_top && box_center_y <= region_bottom;
}

ImageData::BoundingBox* BatchEventDetermine::find_min_width_box_in_region(
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
  BatchEventDetermine::drawEmergencyLaneQuarterPoints(cv::Mat &image,
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
  BatchEventDetermine::determineObjectStatus(const ImageData::BoundingBox &box,
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
