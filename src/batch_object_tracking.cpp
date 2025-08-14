#include "batch_object_tracking.h"
#include <iostream>
#include <algorithm>
// #include <execution>
#include <opencv2/imgproc.hpp>
#include <future>


BatchObjectTracking::BatchObjectTracking(int num_threads, const PipelineConfig* config)
    : num_threads_(num_threads), running_(false), stop_requested_(false),
      tracking_confidence_threshold_(0.3f), max_disappeared_frames_(10), 
      iou_threshold_(0.3f), next_track_id_(1) {
    
    // 初始化配置
    if (config) {
        config_ = *config;
        // tracking_confidence_threshold_ = config->tracking_confidence_threshold;
        // max_disappeared_frames_ = config->max_disappeared_frames;
        // iou_threshold_ = config->tracking_iou_threshold;
    }
    track_instances_.reserve(num_threads_);
    auto track_instance = xtkj::createTracker(30, 30, 0.5, 0.6, 0.8);
    track_instances_.push_back(std::unique_ptr<xtkj::ITracker>(track_instance));
    vehicle_parking_instance_ = createVehicleParkingDetect();
    // 创建输入输出连接器
    input_connector_ = std::make_unique<BatchConnector>(10);
    output_connector_ = std::make_unique<BatchConnector>(10);
    
    // 初始化跟踪模型
    if (!initialize_tracking_models()) {
        std::cerr << "❌ 批次目标跟踪模型初始化失败" << std::endl;
    }
}

BatchObjectTracking::~BatchObjectTracking() {
    stop();
    cleanup_tracking_models();
}

void BatchObjectTracking::start() {
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
        worker_threads_.emplace_back(&BatchObjectTracking::worker_thread_func, this);
    }
    
    std::cout << "✅ 批次目标跟踪已启动，使用 " << num_threads_ << " 个线程" << std::endl;
}

void BatchObjectTracking::stop() {
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
    
    std::cout << "🛑 批次目标跟踪已停止" << std::endl;
}

bool BatchObjectTracking::add_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    return input_connector_->send_batch(batch);
}

bool BatchObjectTracking::get_processed_batch(BatchPtr& batch) {
    return output_connector_->receive_batch(batch);
}

bool BatchObjectTracking::process_batch(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // std::cout << "🏃 开始处理批次 " << batch->batch_id 
    //           << " 目标跟踪，包含 " << batch->actual_size << " 个图像" << std::endl;
    
    try {
        // 串行处理以保证跟踪的时序性
        // 批次内的图像需要按帧序号顺序处理
        std::sort(batch->images.begin(), batch->images.begin() + batch->actual_size,
                  [](const ImageDataPtr& a, const ImageDataPtr& b) {
                      return a->frame_idx < b->frame_idx;
                  });
        
        // 使用批次处理锁确保轨迹数据一致性
        std::lock_guard<std::mutex> batch_lock(batch_processing_mutex_);
        
        // 逐帧处理跟踪（保持时序）
        for (size_t i = 0; i < batch->actual_size; ++i) {
            int thread_id = i % num_threads_;
            process_image_tracking(batch->images[i], thread_id);
        }
        
        // 批次内轨迹一致性处理
        process_batch_trajectory_consistency(batch);
        
        // 标记批次完成
        batch->tracking_completed.store(true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 更新统计信息
        processed_batch_count_.fetch_add(1);
        total_processing_time_ms_.fetch_add(duration.count());
        total_images_processed_.fetch_add(batch->actual_size);
        
        std::cout << "✅ 批次 " << batch->batch_id << " 目标跟踪完成，耗时: " 
                  << duration.count() << "ms，平均每张: " 
                  << (double)duration.count() / batch->actual_size << "ms" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次 " << batch->batch_id << " 目标跟踪异常: " << e.what() << std::endl;
        return false;
    }
}

void BatchObjectTracking::worker_thread_func() {
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
                    std::cerr << "❌ 批次 " << batch->batch_id << " 目标跟踪失败，丢弃" << std::endl;
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
}

void BatchObjectTracking::process_image_tracking(ImageDataPtr image, int thread_id) {
    if (!image) {
        return;
    }
    
    try {
        // 执行目标跟踪
        perform_object_tracking(image, thread_id);
        
        // 标记跟踪完成
        image->track_completed = true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 图像 " << image->frame_idx << " 目标跟踪失败: " << e.what() << std::endl;
        image->track_completed = true; // 即使失败也标记为完成，避免死锁
    }
}

void BatchObjectTracking::perform_object_tracking(ImageDataPtr image, int thread_id) {
    if (!image || image->imageMat.empty()) {
        return;
    }
    
    // 检查线程ID是否有效
    if (thread_id < 0 || thread_id >= track_instances_.size()) {
        std::cerr << "❌ 无效的跟踪线程ID: " << thread_id << std::endl;
        return;
    }
    
    // 清空之前的跟踪结果
    image->track_results.clear();
    
    try {
        // 如果没有检测结果，直接返回
        if (image->detection_results.empty()) {
            return;
        }
        detect_result_group_t *out = new detect_result_group_t();
        for(auto detect_box:image->detection_results) {
            detect_result_t result;
            result.cls_id = detect_box.class_id;
            result.box.left = detect_box.left;
            result.box.top = detect_box.top;
            result.box.right = detect_box.right;
            result.box.bottom = detect_box.bottom;
            result.prop = detect_box.confidence;
            result.track_id = detect_box.track_id; // 保留跟踪ID
            out->results[out->count++] = result;
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        track_instances_[0]->track(out, image->roi.width,
                                            image->roi.height);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "🎯 目标跟踪耗时: " << duration.count() << " ms" << std::endl;
        image->track_results.clear();
        std::vector<TrackBox> track_boxes;
        for (int i = 0; i < out->count; ++i) {
            detect_result_t &result = out->results[i];
            // 这里的box是resize后的坐标，需要转换回原图像坐标系
            TrackBox box = TrackBox(result.track_id, 
                                        cv::Rect((result.box.left + image->roi.x) * image->parkingResizeMat.cols / image->width, 
                                        (result.box.top + image->roi.y) * image->parkingResizeMat.rows / image->height,
                                        (result.box.right - result.box.left) * image->parkingResizeMat.cols / image->width,
                                        (result.box.bottom - result.box.top) * image->parkingResizeMat.rows / image->height),
                                        result.cls_id, 
                                        result.prop, 
                                        false, 0.0);
            track_boxes.push_back(box);
            // cv::rectangle(image->parkingResizeMat, box.box, cv::Scalar(0, 255, 0), 2);

            // cv::rectangle(image->imageMat, 
            //               cv::Rect((result.box.left + image->roi.x), 
            //                        (result.box.top + image->roi.y),
            //                        (result.box.right - result.box.left),
            //                        (result.box.bottom - result.box.top)), 
            //               cv::Scalar(0, 255, 0), 2);
            // ImageData::BoundingBox track_box;
            // track_box.track_id = result.track_id;
            // track_box.left = result.box.left;
            // track_box.top = result.box.top;
            // track_box.right = result.box.right;
            // track_box.bottom = result.box.bottom;
            // track_box.confidence = track_box.confidence;
            // track_box.class_id = track_box.class_id;
            // track_box.is_still = track_box.is_still;
            // image->track_results.push_back(track_box);
        }
        // cv::imwrite("parkingResizeMat.png", image->parkingResizeMat);
        // cv::imwrite("imageMat.png", image->imageMat);
        // exit(0);
        start_time = std::chrono::high_resolution_clock::now();
        vehicle_parking_instance_->detect(image->parkingResizeMat, track_boxes);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "🚗 车辆违停检测耗时: " << duration.count() << " ms," << "图片大小：" << image->parkingResizeMat.rows << " " << image->parkingResizeMat.cols << std::endl;
        for(const auto &track_box : track_boxes) {
          ImageData::BoundingBox box;
          box.track_id = track_box.track_id;
          box.left = track_box.box.x * image->width / image->parkingResizeMat.cols;
          box.top = track_box.box.y * image->height / image->parkingResizeMat.rows;
          box.right = (track_box.box.x + track_box.box.width) * image->width / image->parkingResizeMat.cols;
          box.bottom = (track_box.box.y + track_box.box.height) * image->height / image->parkingResizeMat.rows;
          box.confidence = track_box.confidence;
          box.class_id = track_box.cls_id;
          box.is_still = track_box.is_still;
          image->track_results.push_back(box);
        //   cv::rectangle(image->imageMat, 
        //                 cv::Rect(box.left, box.top, 
        //                          box.right - box.left, 
        //                          box.bottom - box.top), 
        //                 cv::Scalar(0, 255, 0), 2);
        }
        // cv::imwrite("tracking_result_" + std::to_string(image->frame_idx) + ".png", image->imageMat);
        // exit(0);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 目标跟踪执行失败: " << e.what() << std::endl;
    }
}

void BatchObjectTracking::process_batch_trajectory_consistency(BatchPtr batch) {
    // 批次内轨迹一致性处理
    // 确保同一目标在批次内的轨迹ID一致
    
    // std::map<int, std::vector<size_t>> track_id_to_frames;
    
    // // 收集所有轨迹ID及其出现的帧
    // for (size_t i = 0; i < batch->actual_size; ++i) {
    //     for (const auto& track_result : batch->images[i]->track_results) {
    //         track_id_to_frames[track_result.track_id].push_back(i);
    //     }
    // }
    
    // // 对于在批次内出现多次的轨迹，进行一致性检查
    // for (const auto& [track_id, frame_indices] : track_id_to_frames) {
    //     if (frame_indices.size() > 1) {
    //         // 检查轨迹的连续性和合理性
    //         for (size_t i = 1; i < frame_indices.size(); ++i) {
    //             size_t prev_frame_idx = frame_indices[i-1];
    //             size_t curr_frame_idx = frame_indices[i];
                
    //             // 获取前后帧的同一轨迹
    //             auto& prev_tracks = batch->images[prev_frame_idx]->track_results;
    //             auto& curr_tracks = batch->images[curr_frame_idx]->track_results;
                
    //             auto prev_it = std::find_if(prev_tracks.begin(), prev_tracks.end(),
    //                 [track_id](const ImageData::BoundingBox& box) { 
    //                     return box.track_id == track_id; 
    //                 });
                    
    //             auto curr_it = std::find_if(curr_tracks.begin(), curr_tracks.end(),
    //                 [track_id](const ImageData::BoundingBox& box) { 
    //                     return box.track_id == track_id; 
    //                 });
                
    //             if (prev_it != prev_tracks.end() && curr_it != curr_tracks.end()) {
    //                 // 计算位移距离
    //                 float dx = (curr_it->left + curr_it->right) / 2.0f - (prev_it->left + prev_it->right) / 2.0f;
    //                 float dy = (curr_it->top + curr_it->bottom) / 2.0f - (prev_it->top + prev_it->bottom) / 2.0f;
    //                 float distance = std::sqrt(dx * dx + dy * dy);
                    
    //                 // 根据位移判断是否静止
    //                 if (distance < 10.0f) { // 10像素以内认为静止
    //                     curr_it->is_still = true;
    //                     curr_it->status = ObjectStatus::Stationary;
    //                 } else {
    //                     curr_it->is_still = false;
    //                     curr_it->status = ObjectStatus::Moving;
    //                 }
    //             }
    //         }
    //     }
    // }
}

void BatchObjectTracking::update_trajectory_database(const std::vector<ImageData::BoundingBox>& track_results, uint64_t frame_idx) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    // 更新现有轨迹
    for (const auto& track_result : track_results) {
        int track_id = track_result.track_id;
        
        if (trajectory_database_.find(track_id) != trajectory_database_.end()) {
            // 更新现有轨迹
            auto& trajectory = trajectory_database_[track_id];
            trajectory.last_bbox = cv::Rect(track_result.left, track_result.top, 
                                          track_result.right - track_result.left,
                                          track_result.bottom - track_result.top);
            trajectory.last_frame_idx = frame_idx;
            trajectory.disappeared_count = 0;
            trajectory.is_active = true;
        } else {
            // 创建新轨迹
            TrajectoryInfo new_trajectory;
            new_trajectory.track_id = track_id;
            new_trajectory.last_bbox = cv::Rect(track_result.left, track_result.top,
                                              track_result.right - track_result.left,
                                              track_result.bottom - track_result.top);
            new_trajectory.last_frame_idx = frame_idx;
            new_trajectory.disappeared_count = 0;
            new_trajectory.is_active = true;
            
            trajectory_database_[track_id] = new_trajectory;
        }
    }
    
    // 更新消失的轨迹
    for (auto& [track_id, trajectory] : trajectory_database_) {
        if (trajectory.last_frame_idx < frame_idx) {
            trajectory.disappeared_count++;
            if (trajectory.disappeared_count > max_disappeared_frames_) {
                trajectory.is_active = false;
            }
        }
    }
}

int BatchObjectTracking::assign_track_id(const ImageData::BoundingBox& detection, uint64_t frame_idx) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    cv::Rect detection_rect(detection.left, detection.top,
                           detection.right - detection.left,
                           detection.bottom - detection.top);
    
    // 尝试匹配现有轨迹
    float best_iou = 0.0f;
    int best_track_id = -1;
    
    for (const auto& [track_id, trajectory] : trajectory_database_) {
        if (!trajectory.is_active) continue;
        
        // 计算IoU
        cv::Rect intersection = detection_rect & trajectory.last_bbox;
        cv::Rect union_rect = detection_rect | trajectory.last_bbox;
        
        if (union_rect.area() > 0) {
            float iou = (float)intersection.area() / union_rect.area();
            if (iou > best_iou && iou > iou_threshold_) {
                best_iou = iou;
                best_track_id = track_id;
            }
        }
    }
    
    if (best_track_id != -1) {
        return best_track_id;
    } else {
        // 创建新的轨迹ID
        return next_track_id_++;
    }
}

bool BatchObjectTracking::initialize_tracking_models() {
    // track_instances_.clear();
    // track_instances_.reserve(num_threads_);
    
    // for (int i = 0; i < num_threads_; ++i) {
    //     auto track_instance = xtkj::CreateTrack();
        
    //     xtkj::TrackInitParam track_params;
    //     track_params.model_path = config_.track_model_path.empty() ? "track_model" : config_.track_model_path;
    //     track_params.confidence_threshold = tracking_confidence_threshold_;
    //     track_params.max_disappeared_frames = max_disappeared_frames_;
    //     track_params.iou_threshold = iou_threshold_;
        
    //     if (track_instance->Init(track_params) != 0) {
    //         std::cerr << "❌ 目标跟踪模型初始化失败，线程 " << i << std::endl;
    //         return false;
    //     } else {
    //         std::cout << "✅ 目标跟踪模型初始化成功，线程 " << i << std::endl;
    //     }
        
    //     track_instances_.push_back(std::move(track_instance));
    // }
    
    return true;
}

void BatchObjectTracking::cleanup_tracking_models() {
    for (auto& instance : track_instances_) {
        if (instance) {
            // instance->Release();
        }
    }
    track_instances_.clear();
}

// BatchStage接口实现
std::string BatchObjectTracking::get_stage_name() const {
    return "批次目标跟踪";
}

size_t BatchObjectTracking::get_processed_count() const {
    return processed_batch_count_.load();
}

double BatchObjectTracking::get_average_processing_time() const {
    auto count = processed_batch_count_.load();
    if (count == 0) return 0.0;
    return (double)total_processing_time_ms_.load() / count;
}

size_t BatchObjectTracking::get_queue_size() const {
    return input_connector_->get_queue_size();
}
