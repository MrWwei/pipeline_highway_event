#include "highway_event.h"
#include "image_data.h"
#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <sstream>

/**
 * HighwayEventDetector的具体实现类
 */
class HighwayEventDetectorImpl : public HighwayEventDetector {
public:
    HighwayEventDetectorImpl();
    ~HighwayEventDetectorImpl() override;
    
    // 实现纯虚函数
    bool initialize(const HighwayEventConfig& config) override;
    bool change_params(const HighwayEventConfig& config) override;
    bool start() override;
    int64_t add_frame(const cv::Mat& image) override;
    int64_t add_frame(cv::Mat&& image) override;
    ProcessResult get_result(uint64_t frame_id) override;
    ProcessResult get_result_with_timeout(uint64_t frame_id, int timeout_ms) override;
    void stop() override;
    bool is_initialized() const override;
    bool is_running() const override;
    const HighwayEventConfig& get_config() const override;
    std::string get_pipeline_status() const override;

private:
    // 成员变量
    std::unique_ptr<PipelineManager> pipeline_manager_;
    HighwayEventConfig config_;
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};
    std::atomic<uint64_t> next_frame_id_{0};
    
    // 结果管理
    mutable std::mutex result_mutex_;
    mutable std::condition_variable result_cv_;
    mutable std::condition_variable result_space_cv_; // 新增：用于等待结果缓存有空间
    std::unordered_map<uint64_t, ImageDataPtr> completed_results_;
    static constexpr size_t MAX_COMPLETED_RESULTS = 100; // 最大结果缓存数量
    
    // 内部结果处理线程
    std::thread result_thread_;
    std::atomic<bool> result_thread_running_{false};
    
    // 内部方法
    void result_processing_thread();
    
    // 转换函数：从ImageData转换为ProcessResult
    ProcessResult convert_to_process_result(ImageDataPtr image_data);
};

// 实现类的方法定义
HighwayEventDetectorImpl::HighwayEventDetectorImpl() {
    // 构造函数中不做实际初始化，等待用户调用initialize
}

HighwayEventDetectorImpl::~HighwayEventDetectorImpl() {
    stop();
}

void HighwayEventDetectorImpl::result_processing_thread() {
    while (result_thread_running_.load()) {
        ImageDataPtr result;
        
        // 从流水线获取完成的结果
        if (pipeline_manager_->get_final_result(result)) {
            {
                std::unique_lock<std::mutex> lock(result_mutex_);
                
                // 等待直到有空间存储新结果（阻塞机制）
                result_space_cv_.wait(lock, [this]() {
                    return completed_results_.size() < MAX_COMPLETED_RESULTS || !result_thread_running_.load();
                });
                
                // 如果线程已停止，退出
                if (!result_thread_running_.load()) {
                    break;
                }
                
                // 存储结果
                completed_results_[result->frame_idx] = result;
                
                if (config_.enable_debug_log) {
                    std::cout << "✅ 结果处理完成，帧ID: " << result->frame_idx 
                              << "，当前缓存数量: " << completed_results_.size() 
                              << "/" << MAX_COMPLETED_RESULTS << std::endl;
                }
            }
            result_cv_.notify_all(); // 通知等待结果的线程
        } else {
            // 没有结果，短暂休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

ProcessResult HighwayEventDetectorImpl::convert_to_process_result(ImageDataPtr image_data) {
    ProcessResult result;
    result.status = ResultStatus::SUCCESS;
    result.frame_id = image_data->frame_idx;
    result.roi = image_data->roi;
    // result.srcImage = image_data->imageMat->clone(); // 保留源图像
    // result.mask = image_data->mask.clone();
    cv::Mat image_src = image_data->imageMat;
    // if(!image_data->mask.empty()) {
    //     cv::Mat mask = image_data->mask.clone();
    //     cv::imwrite("mask_outs/output_" + std::to_string(result.frame_id) + ".jpg", mask);
    // }
    
    // 转换检测结果
    result.detections.reserve(image_data->track_results.size());
    for (const auto& box : image_data->track_results) {
        DetectionBox det_box;
        det_box.left = box.left;
        det_box.top = box.top;
        det_box.right = box.right;
        det_box.bottom = box.bottom;

        det_box.confidence = box.confidence;
        det_box.class_id = box.class_id;
        det_box.track_id = box.track_id;
        det_box.status = box.status;
        result.detections.push_back(det_box);
        // cv::Scalar color = box.is_still ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        // cv::rectangle(image_src, 
        //             cv::Point(box.left, box.top), 
        //             cv::Point(box.right, box.bottom), 
        //             color);
        // cv::putText(image_src, 
        //           std::to_string(box.track_id), 
        //           cv::Point(box.left, box.top - 5), 
        //           cv::FONT_HERSHEY_SIMPLEX, 
        //           0.5,
        //           color, 1);
    }
    // cv::imwrite("track_outs/output_" + std::to_string(result.frame_id) + ".jpg", image_src);
    // for (const auto& box : image_data->detection_results) {
    //     DetectionBox det_box;
    //     det_box.left = box.left;
    //     det_box.top = box.top;
    //     det_box.right = box.right;
    //     det_box.bottom = box.bottom;

    //     det_box.confidence = box.confidence;
    //     det_box.class_id = box.class_id;
    //     det_box.track_id = box.track_id;
    //     det_box.status = box.status;
    //     cv::rectangle(image_src, 
    //                 cv::Point(box.left, box.top), 
    //                 cv::Point(box.right, box.bottom), 
    //                 cv::Scalar(0, 255, 0), 2);
    //     cv::putText(image_src, 
    //               std::to_string(box.track_id), 
    //               cv::Point(box.left, box.top - 5), 
    //               cv::FONT_HERSHEY_SIMPLEX, 
    //               0.5,
    //               cv::Scalar(0, 255, 0), 1);
    // }
    // cv::imwrite("detect_outs/output_" + std::to_string(result.frame_id) + ".jpg", image_src);
    
    // 转换筛选结果
    result.has_filtered_box = image_data->has_filtered_box;
    if (result.has_filtered_box) {
        const auto& box = image_data->filtered_box;
        result.filtered_box.left = box.left;
        result.filtered_box.top = box.top;
        result.filtered_box.right = box.right;
        result.filtered_box.bottom = box.bottom;
        result.filtered_box.confidence = box.confidence;
        result.filtered_box.class_id = box.class_id;
        result.filtered_box.track_id = box.track_id;
        result.filtered_box.status = box.status;
    }
    
    return result;
}

// HighwayEventDetectorImpl公共接口实现
bool HighwayEventDetectorImpl::initialize(const HighwayEventConfig& config) {
    if (is_initialized_.load()) {
        std::cerr << "❌ HighwayEventDetector 已经初始化过了" << std::endl;
        return false;
    }
    
    try {
        config_ = config;
        
        // 创建流水线配置
        PipelineConfig pipeline_config;
        pipeline_config.semantic_threads = config.semantic_threads;
        pipeline_config.mask_postprocess_threads = config.mask_threads;
        pipeline_config.detection_threads = config.detection_threads;
        pipeline_config.tracking_threads = config.tracking_threads;
        pipeline_config.event_determine_threads = config.filter_threads;
        
        // 添加模块开关配置
        pipeline_config.enable_segmentation = config.enable_segmentation;
        pipeline_config.enable_mask_postprocess = config.enable_mask_postprocess;
        pipeline_config.enable_detection = config.enable_detection;
        pipeline_config.enable_tracking = config.enable_tracking;
        pipeline_config.enable_event_determine = config.enable_event_determine;
        pipeline_config.enable_pedestrian_detect = config.enable_pedestrian_detect;
        
        pipeline_config.seg_model_path = config.seg_model_path;
        pipeline_config.car_det_model_path = config.car_det_model_path;
        pipeline_config.pedestrian_det_model_path = config.pedestrian_det_model_path;
        pipeline_config.enable_seg_show = config.enable_seg_show;
        pipeline_config.seg_show_image_path = config.seg_show_image_path;
        pipeline_config.det_algor_name = config.det_algor_name;
        pipeline_config.det_img_size = config.det_img_size;
        pipeline_config.det_conf_thresh = config.det_conf_thresh;
        pipeline_config.det_iou_thresh = config.det_iou_thresh;
        pipeline_config.det_max_batch_size = config.det_max_batch_size;
        pipeline_config.det_min_opt = config.det_min_opt;
        pipeline_config.det_mid_opt = config.det_mid_opt;
        pipeline_config.det_max_opt = config.det_max_opt;
        pipeline_config.det_is_ultralytics = config.det_is_ultralytics;
        pipeline_config.det_gpu_id = config.det_gpu_id;
        pipeline_config.event_determine_top_fraction = config.box_filter_top_fraction;
        pipeline_config.event_determine_bottom_fraction = config.box_filter_bottom_fraction;
        pipeline_config.final_result_queue_capacity = config.result_queue_capacity;
        pipeline_config.times_car_width = config.times_car_width; // 车宽倍数
        pipeline_config.enable_lane_show = config.enable_lane_show;
        pipeline_config.lane_show_image_path = config.lane_show_image_path;

        
        // 创建流水线管理器（但不启动）
        pipeline_manager_ = std::make_unique<PipelineManager>(pipeline_config);
        
        is_initialized_.store(true);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ HighwayEventDetector 初始化失败: " << e.what() << std::endl;
        is_initialized_.store(false);
        return false;
    }
}

bool HighwayEventDetectorImpl::change_params(const HighwayEventConfig& config) {
    if (!is_initialized_.load()) {
        std::cerr << "❌ HighwayEventDetector 尚未初始化，请先调用 initialize()" << std::endl;
        return false;
    }
    
    // 更新配置
    config_ = config;
    PipelineConfig pipeline_config;
    pipeline_config.enable_seg_show = config.enable_seg_show;
    pipeline_config.enable_lane_show = config.enable_lane_show;
    pipeline_config.seg_show_image_path = config.seg_show_image_path;
    pipeline_config.lane_show_image_path = config.lane_show_image_path;
    pipeline_config.times_car_width = config.times_car_width; // 车宽倍数
    pipeline_config.event_determine_top_fraction = config.box_filter_top_fraction;
    pipeline_config.event_determine_bottom_fraction = config.box_filter_bottom_fraction;
    pipeline_manager_->change_params(pipeline_config);
    
    // 这里可以添加更多的参数更新逻辑
    // 例如，更新流水线管理器的配置等
    
    return true;
}

bool HighwayEventDetectorImpl::start() {
    if (!is_initialized_.load()) {
        std::cerr << "❌ HighwayEventDetector 尚未初始化，请先调用 initialize()" << std::endl;
        return false;
    }
    
    if (is_running_.load()) {
        std::cerr << "❌ HighwayEventDetector 已经在运行中" << std::endl;
        return false;
    }
    
    
    try {
        // 启动流水线
        pipeline_manager_->start();
        
        // 启动内部结果处理线程
        result_thread_running_.store(true);
        result_thread_ = std::thread(&HighwayEventDetectorImpl::result_processing_thread, this);
        
        is_running_.store(true);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ HighwayEventDetector 启动失败: " << e.what() << std::endl;
        is_running_.store(false);
        return false;
    }
}

int64_t HighwayEventDetectorImpl::add_frame(const cv::Mat& image) {
    if (!is_running_.load()) {
        std::cerr << "❌ 流水线未初始化或未运行，请先调用 initialize()" << std::endl;
        return -1;
    }
    
    if (image.empty()) {
        std::cerr << "❌ 输入图像为空" << std::endl;
        return -1;
    }
    
    try {
        // 分配帧ID
        uint64_t frame_id = next_frame_id_.fetch_add(1);
        // 创建图像数据（拷贝） - 使用异常安全的方式
        ImageDataPtr img_data = std::make_shared<ImageData>(image);
        img_data->frame_idx = frame_id;
        img_data->roi = cv::Rect(0, 0, image.cols, image.rows); // 默认ROI为整个图像
        
        // 添加到流水线
        pipeline_manager_->add_image(img_data);
        
        return static_cast<int64_t>(frame_id);
    } catch (const std::exception& e) {
        // 异常安全：报告错误
        std::cerr << "❌ 添加帧失败: " << e.what() << std::endl;
        return -1;
    }
}

int64_t HighwayEventDetectorImpl::add_frame(cv::Mat&& image) {
    if (!is_running_.load()) {
        std::cerr << "❌ 流水线未初始化或未运行，请先调用 initialize()" << std::endl;
        return -1;
    }
    
    if (image.empty()) {
        std::cerr << "❌ 输入图像为空" << std::endl;
        return -1;
    }
    
    try {
        // 分配帧ID
        uint64_t frame_id = next_frame_id_.fetch_add(1);
        
        // 创建图像数据（移动） - 使用异常安全的方式
        ImageDataPtr img_data = std::make_shared<ImageData>(std::move(image));
        img_data->frame_idx = frame_id;
        img_data->roi = cv::Rect(0, 0, img_data->width, img_data->height); // 设置默认ROI为整个图像
        
        // 添加到流水线
        pipeline_manager_->add_image(img_data);
        
        return static_cast<int64_t>(frame_id);
    } catch (const std::exception& e) {
        // 异常安全：报告错误
        std::cerr << "❌ 添加帧失败: " << e.what() << std::endl;
        return -1;
    }
}

ProcessResult HighwayEventDetectorImpl::get_result(uint64_t frame_id) {
    return get_result_with_timeout(frame_id, config_.get_timeout_ms);
}

ProcessResult HighwayEventDetectorImpl::get_result_with_timeout(uint64_t frame_id, int timeout_ms) {
    ProcessResult result;
    result.frame_id = frame_id;
    
    if (!is_running_.load()) {
        result.status = ResultStatus::ERROR;
        return result;
    }
    std::unique_lock<std::mutex> lock(result_mutex_);
    
    // 先检查结果是否已经存在
    auto it = completed_results_.find(frame_id);
    if (it != completed_results_.end()) {
        result = convert_to_process_result(it->second);
        completed_results_.erase(it);
        
        // 通知结果处理线程有空间了
        result_space_cv_.notify_one();

        return result;
    }
    
    // 等待结果完成
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    
    bool found = result_cv_.wait_until(lock, deadline, [&]() {
        return completed_results_.find(frame_id) != completed_results_.end();
    });
    
    if (!found) {
        if (config_.enable_debug_log) {
            std::cout << "⏰ 帧 " << frame_id << " 等待超时，当前缓存数量: " << completed_results_.size() << std::endl;
        }
        result.status = ResultStatus::TIMEOUT;
        return result;
    }
    
    it = completed_results_.find(frame_id);
    if (it != completed_results_.end()) {
        if (config_.enable_debug_log) {
            std::cout << "✅ 帧 " << frame_id << " 等待成功，开始转换结果" << std::endl;
        }
        result = convert_to_process_result(it->second);
        // 获取后删除结果，避免内存积累
        completed_results_.erase(it);
        
        // 通知结果处理线程有空间了
        result_space_cv_.notify_one();
    } else {
        if (config_.enable_debug_log) {
            std::cout << "❌ 帧 " << frame_id << " 等待结束后未找到结果" << std::endl;
        }
        result.status = ResultStatus::NOT_FOUND;
    }
    
    return result;
}

void HighwayEventDetectorImpl::stop() {
    if (is_running_.load()) {
        is_running_.store(false);
        
        // 停止结果处理线程
        if (result_thread_running_.load()) {
            result_thread_running_.store(false);
            
            // 唤醒可能阻塞的结果处理线程
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_space_cv_.notify_all();
                result_cv_.notify_all();
            }
            
            if (result_thread_.joinable()) {
                result_thread_.join();
            }
        }
        
        // 停止流水线
        if (pipeline_manager_) {
            pipeline_manager_->stop();
        }
        
        // 清理结果
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            completed_results_.clear();
        }
    }
}

bool HighwayEventDetectorImpl::is_initialized() const {
    return is_initialized_.load();
}

bool HighwayEventDetectorImpl::is_running() const {
    return is_running_.load();
}

const HighwayEventConfig& HighwayEventDetectorImpl::get_config() const {
    return config_;
}

std::string HighwayEventDetectorImpl::get_pipeline_status() const {
    if (!pipeline_manager_) {
        return "流水线未初始化";
    }
    
    // 使用 PipelineManager::print_status() 来实时监控队列
    pipeline_manager_->print_status();
    
    // 返回简化的状态信息
    std::ostringstream oss;
    oss << "下一帧ID: " << next_frame_id_.load();
    
    std::lock_guard<std::mutex> lock(result_mutex_);
    oss << ", 结果缓存: " << completed_results_.size() << "/" << MAX_COMPLETED_RESULTS << " 帧";
    
    return oss.str();
}

// 工厂函数实现
std::unique_ptr<HighwayEventDetector> create_highway_event_detector() {
    return std::make_unique<HighwayEventDetectorImpl>();
}

// 销毁函数实现（用于C接口）
void destroy_highway_event_detector(HighwayEventDetector* detector) {
    delete detector;
}
