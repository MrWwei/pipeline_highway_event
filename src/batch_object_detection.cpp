#include "batch_object_detection.h"
#include <iostream>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <future>

BatchObjectDetection::BatchObjectDetection(int num_threads, const PipelineConfig* config)
    : num_threads_(num_threads), running_(false), stop_requested_(false),
      cuda_available_(false), confidence_threshold_(0.5f), nms_threshold_(0.4f),
      enable_car_detection_(true), enable_person_detection_(false) {
    
    // 初始化配置
    if (config) {
        config_ = *config;
        // confidence_threshold_ = config->detection_confidence_threshold;
        // nms_threshold_ = config->detection_nms_threshold;
        // enable_car_detection_ = config->enable_car_detection;
        // enable_person_detection_ = config->enable_person_detection;
    }
    
    // 创建输入输出连接器
    input_connector_ = std::make_unique<BatchConnector>(10);
    output_connector_ = std::make_unique<BatchConnector>(10);
    
    // 初始化CUDA
    try {
        cv::cuda::getCudaEnabledDeviceCount();
        gpu_src_cache_.create(1024, 1024, CV_8UC3);
        gpu_dst_cache_.create(1024, 1024, CV_8UC3);
        cuda_available_ = true;
        std::cout << "✅ CUDA已启用，批次目标检测将使用GPU加速" << std::endl;
    } catch (const cv::Exception& e) {
        cuda_available_ = false;
        std::cout << "⚠️ 未检测到CUDA设备，批次目标检测将使用CPU" << std::endl;
    }
    
    // 初始化检测模型
    if (!initialize_detection_models()) {
        std::cerr << "❌ 批次目标检测模型初始化失败" << std::endl;
    }
}

BatchObjectDetection::~BatchObjectDetection() {
    stop();
    cleanup_detection_models();
}

void BatchObjectDetection::start() {
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
        worker_threads_.emplace_back(&BatchObjectDetection::worker_thread_func, this);
    }
    
    std::cout << "✅ 批次目标检测已启动，使用 " << num_threads_ << " 个线程" << std::endl;
}

void BatchObjectDetection::stop() {
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
    
    std::cout << "🛑 批次目标检测已停止" << std::endl;
}

bool BatchObjectDetection::add_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    return input_connector_->send_batch(batch);
}

bool BatchObjectDetection::get_processed_batch(BatchPtr& batch) {
    return output_connector_->receive_batch(batch);
}

bool BatchObjectDetection::process_batch(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "🎯 开始处理批次 " << batch->batch_id 
              << " 目标检测，包含 " << batch->actual_size << " 个图像" << std::endl;
    
    try {
        // 将图像分配给不同线程处理
        std::vector<cv::Mat> crop_images;
        crop_images.reserve(batch->actual_size);
        for(int i = 0;i<batch->actual_size; ++i) {
            if (batch->images[i]) {
                auto& image = batch->images[i];
                if (image->imageMat.empty()) {
                    std::cerr << "❌ 图像 " << image->frame_idx << " 为空，跳过处理" << std::endl;
                    continue;
                }
                // 进行裁剪
                cv::Mat crop_image = image->imageMat(image->roi);
                crop_images.push_back(crop_image);
            }
        }
        std::vector<detect_result_group_t> car_outs(crop_images.size());
        std::vector<detect_result_group_t*> car_out_ptrs;
        car_out_ptrs.reserve(crop_images.size());
        for (auto& out : car_outs) {
            car_out_ptrs.push_back(&out);
        }
        car_detect_instances_[0]->forward(crop_images, car_out_ptrs.data());
        for(size_t i = 0; i < batch->images.size(); ++i) {
            auto& image = batch->images[i];
            cv::Mat crop_image = crop_images[i];
            if (image && car_out_ptrs[i]) {
                for (int j = 0; j < car_out_ptrs[i]->count; ++j) {
                    auto& result = car_out_ptrs[i]->results[j];
                    ImageData::BoundingBox box;
                    // box.left = result.box.left + image->roi.x;
                    // box.top = result.box.top + image->roi.y;
                    // box.right = result.box.right + image->roi.x;
                    // box.bottom = result.box.bottom + image->roi.y;
                    
                    box.left = result.box.left;
                    box.top = result.box.top;
                    box.right = result.box.right;
                    box.bottom = result.box.bottom;
                    box.confidence = result.prop;
                    box.class_id = result.cls_id;
                    box.track_id = result.track_id;
                    // std::cout << "检测到目标: "
                    //           << "类ID: " << result.cls_id
                    //           << ", 置信度: " << result.prop
                    //           << ", 位置: (" << box.left << ", " << box.top 
                    //           << ") - (" << box.right << ", " << box.bottom << ")" 
                    //           << std::endl;
                    // box.is_still = result.is_still;
                    // box.status = static_cast<ObjectStatus>(result.status);
                    image->detection_results.push_back(box);
                    // cv::rectangle(image->imageMat,
                    //             cv::Point(box.left, box.top), 
                    //             cv::Point(box.right, box.bottom), 
                    //             cv::Scalar(255, 0, 0), 2);
                }
                // 标记检测完成
                image->detection_completed = true;
            }
            // cv::imwrite("batch_outs/batch_" + std::to_string(batch->batch_id) + "_img_" + std::to_string(i) + ".jpg", image->imageMat);
        }
        // 标记批次完成
        batch->detection_completed.store(true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 更新统计信息
        processed_batch_count_.fetch_add(1);
        total_processing_time_ms_.fetch_add(duration.count());
        total_images_processed_.fetch_add(batch->actual_size);
        
        std::cout << "✅ 批次 " << batch->batch_id << " 目标检测完成，耗时: " 
                  << duration.count() << "ms，平均每张: " 
                  << (double)duration.count() / batch->actual_size << "ms" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次 " << batch->batch_id << " 目标检测异常: " << e.what() << std::endl;
        return false;
    }
}

void BatchObjectDetection::worker_thread_func() {
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
                    std::cerr << "❌ 批次 " << batch->batch_id << " 目标检测失败，丢弃" << std::endl;
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
}

void BatchObjectDetection::process_image_detection(ImageDataPtr image, int thread_id) {
    if (!image) {
        return;
    }
    
    try {
        // 执行目标检测
        perform_object_detection(image, thread_id);
        
        // 标记检测完成
        image->detection_completed = true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 图像 " << image->frame_idx << " 目标检测失败: " << e.what() << std::endl;
        image->detection_completed = true; // 即使失败也标记为完成，避免死锁
    }
}

void BatchObjectDetection::perform_object_detection(ImageDataPtr image, int thread_id) {
    if (!image || image->imageMat.empty()) {
        return;
    }
    
    
}

bool BatchObjectDetection::initialize_detection_models() {
    car_detect_instances_.clear();
    personal_detect_instances_.clear();
    
    // 初始化车辆检测模型
    if (enable_car_detection_) {
        car_detect_instances_.resize(1);
        
        for (int i = 0; i < 1; ++i) {
            auto car_detect = xtkj::createDetect();
            
            AlgorConfig car_params;
            car_params.model_path = config_.car_det_model_path.empty() ? "car_detect.trt" : config_.car_det_model_path;
  
            car_detect->init(car_params);
            // if (car_detect->init(car_params)) {
            //     std::cerr << "❌ 车辆检测模型初始化失败，线程 " << i << std::endl;
            //     return false;
            // } else {
            //     std::cout << "✅ 车辆检测模型初始化成功，线程 " << i << std::endl;
            // }
            car_detect_instances_[i] = std::unique_ptr<xtkj::IDetect>(car_detect);
        }
    }
    
    // 初始化行人检测模型（如果启用）
    // if (enable_person_detection_) {
    //     personal_detect_instances_.reserve(num_threads_);
        
    //     for (int i = 0; i < num_threads_; ++i) {
    //         auto person_detect = xtkj::CreateDetect();
            
    //         xtkj::DetectInitParam person_params;
    //         person_params.model_path = config_.person_model_path.empty() ? "person_detect.trt" : config_.person_model_path;
    //         person_params.confidence_threshold = confidence_threshold_;
    //         person_params.nms_threshold = nms_threshold_;
            
    //         if (person_detect->Init(person_params) != 0) {
    //             std::cerr << "❌ 行人检测模型初始化失败，线程 " << i << std::endl;
    //             return false;
    //         } else {
    //             std::cout << "✅ 行人检测模型初始化成功，线程 " << i << std::endl;
    //         }
            
    //         personal_detect_instances_.push_back(std::move(person_detect));
    //     }
    // }
    
    return true;
}

void BatchObjectDetection::cleanup_detection_models() {
}

// BatchStage接口实现
std::string BatchObjectDetection::get_stage_name() const {
    return "批次目标检测";
}

size_t BatchObjectDetection::get_processed_count() const {
    return processed_batch_count_.load();
}

double BatchObjectDetection::get_average_processing_time() const {
    auto count = processed_batch_count_.load();
    if (count == 0) return 0.0;
    return (double)total_processing_time_ms_.load() / count;
}

size_t BatchObjectDetection::get_queue_size() const {
    return input_connector_->get_queue_size();
}
