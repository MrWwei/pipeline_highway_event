#include "batch_semantic_segmentation.h"
#include "logger_manager.h"
#include <iostream>
#include <algorithm>
// #include <execution>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <future>

BatchSemanticSegmentation::BatchSemanticSegmentation(int num_threads, const PipelineConfig* config)
    : num_threads_(num_threads), running_(false), stop_requested_(false),
      cuda_available_(false), enable_seg_show_(false), seg_show_interval_(10) {
    LOG_INFO("🏗️ 初始化批次语义分割阶段...");
    
    // 创建线程池
    thread_pool_ = std::make_unique<ThreadPool>(8);
    
    // 初始化配置
    if (config) {
        config_ = *config;
        enable_seg_show_ = config->enable_seg_show;
        seg_show_image_path_ = config->seg_show_image_path;
    }
    
    // 创建输入输出连接器
    input_connector_ = std::make_unique<BatchConnector>(10);  // 最多10个批次排队
    output_connector_ = std::make_unique<BatchConnector>(10);
    
    // 初始化CUDA
    try {
        cv::cuda::getCudaEnabledDeviceCount();
        gpu_src_cache_.create(1024, 1024, CV_8UC3);
        gpu_dst_cache_.create(1024, 1024, CV_8UC3);
        cuda_available_ = true;
        LOG_INFO("✅ CUDA已启用，批次语义分割将使用GPU加速");
    } catch (const cv::Exception& e) {
        cuda_available_ = false;
        LOG_INFO("⚠️ 未检测到CUDA设备，批次语义分割将使用CPU");
    }
    
    // 初始化语义分割模型
    if (!initialize_seg_models()) {
        LOG_ERROR("❌ 批次语义分割模型初始化失败");
    }
}

BatchSemanticSegmentation::~BatchSemanticSegmentation() {
    stop();
    cleanup_seg_models();
}

void BatchSemanticSegmentation::start() {
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
        worker_threads_.emplace_back(&BatchSemanticSegmentation::worker_thread_func, this);
    }
    
    std::cout << "✅ 批次语义分割已启动，使用 " << num_threads_ << " 个线程" << std::endl;
}

void BatchSemanticSegmentation::stop() {
    if (!running_.load()) {
        return;
    }
    
    stop_requested_.store(true);
    running_.store(false);
    
    // 停止线程池
    if (thread_pool_) {
        thread_pool_->stop();
    }
    
    // 停止连接器
    input_connector_->stop();
    output_connector_->stop();
    
    // 通知所有等待的线程
    processing_queue_cv_.notify_all();
    
    // 等待所有工作线程结束
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    LOG_INFO("🛑 批次语义分割已停止");
}

bool BatchSemanticSegmentation::add_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    return input_connector_->send_batch(batch);
}

bool BatchSemanticSegmentation::get_processed_batch(BatchPtr& batch) {
    return output_connector_->receive_batch(batch);
}

bool BatchSemanticSegmentation::process_batch(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    batch->start_processing();
    
    // std::cout << "🎨 开始处理批次 " << batch->batch_id 
    //           << "，包含 " << batch->actual_size << " 个图像" << std::endl;
    
    try {
        // 第一步：预处理所有图像
        preprocess_batch(batch);
        
        // 第二步：批量推理
        if (!inference_batch(batch)) {
            std::cerr << "❌ 批次 " << batch->batch_id << " 推理失败" << std::endl;
            return false;
        }
        
        // 第三步：后处理
        postprocess_batch(batch);
        
        // 标记批次完成
        batch->segmentation_completed.store(true);
        batch->complete_processing();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 更新统计信息
        processed_batch_count_.fetch_add(1);
        total_processing_time_ms_.fetch_add(duration.count());
        total_images_processed_.fetch_add(batch->actual_size);
        
        // std::cout << "✅ 批次 " << batch->batch_id << " 语义分割完成，耗时: " 
        //           << duration.count() << "ms，平均每张: " 
        //           << (double)duration.count() / batch->actual_size << "ms" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次 " << batch->batch_id << " 处理异常: " << e.what() << std::endl;
        return false;
    }
}

void BatchSemanticSegmentation::worker_thread_func() {
    while (running_.load()) {
        BatchPtr batch;
        LOG_INFO("🔄 等待输入批次...");
        // 从输入连接器获取批次
        if (input_connector_->receive_batch(batch)) {
            if (batch) {
                // 处理批次
                bool success = process_batch(batch);
                
                if (success) {
                    // 发送到输出连接器
                    std::cout << "📦 批次 " << batch->batch_id << " 处理完成，发送到输出连接器" << std::endl;
                    output_connector_->send_batch(batch);
                } else {
                    std::cerr << "❌ 批次 " << batch->batch_id << " 处理失败，丢弃" << std::endl;
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
}

void BatchSemanticSegmentation::preprocess_batch(BatchPtr batch) {
    // std::cout << "🔄 批次 " << batch->batch_id << " 开始预处理..." << std::endl;
    
    // 使用线程池并发预处理所有图像
    bool success = preprocess_batch_with_threadpool(batch);
    
    if (success) {
        // std::cout << "✅ 批次 " << batch->batch_id << " 预处理完成" << std::endl;
    } else {
        std::cerr << "❌ 批次 " << batch->batch_id << " 预处理失败" << std::endl;
    }
}

bool BatchSemanticSegmentation::preprocess_batch_with_threadpool(BatchPtr batch) {
    if (!batch || batch->is_empty() || !thread_pool_ || !thread_pool_->is_running()) {
        return false;
    }
    
    std::vector<std::future<bool>> futures;
    futures.reserve(batch->actual_size);
    
    // 为批次中的每个图像提交预处理任务到线程池
    for (size_t i = 0; i < batch->actual_size; ++i) {
        if (batch->images[i]) {
            try {
                auto future = thread_pool_->enqueue([this, image = batch->images[i], thread_id = i % num_threads_]() -> bool {
                    try {
                        // auto start_time = std::chrono::high_resolution_clock::now();
                        this->preprocess_image(image, thread_id);
                        // auto end_time = std::chrono::high_resolution_clock::now();
                        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                        // std::cout << "语义分割预处理图像 " << image->frame_idx 
                        //           << " 耗时: " << duration.count() << " ms" << std::endl;
                        return true;
                    } catch (const std::exception& e) {
                        std::cerr << "❌ 图像 " << image->frame_idx << " 预处理异常: " << e.what() << std::endl;
                        return false;
                    }
                });
                futures.push_back(std::move(future));
            } catch (const std::exception& e) {
                std::cerr << "❌ 无法提交图像 " << batch->images[i]->frame_idx << " 预处理任务到线程池: " << e.what() << std::endl;
                return false;
            }
        } else {
            std::cerr << "⚠️ 批次 " << batch->batch_id << " 图像 " << i << " 为空" << std::endl;
        }
    }
    
    // 等待所有预处理任务完成
    bool all_success = true;
    for (auto& future : futures) {
        try {
            if (!future.get()) {
                all_success = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ 获取预处理任务结果异常: " << e.what() << std::endl;
            all_success = false;
        }
    }
    
    return all_success;
}

bool BatchSemanticSegmentation::inference_batch(BatchPtr batch) {
    
    if (seg_instances_.empty()) {
        LOG_ERROR("❌ 语义分割模型实例未初始化");
        return false;
    }
    
    std::cout << "🧠 批次 " << batch->batch_id << " 开始推理..." << std::endl;

    // 准备批量输入数据
    std::vector<cv::Mat> image_mats;
    image_mats.reserve(batch->actual_size);
    std::cout << "批次实际图像数量: " << batch->actual_size << std::endl;
    // exit(0);
    
    for (size_t i = 0; i < batch->actual_size; ++i) {
        if (!batch->images[i]->segInResizeMat.empty()) {
            // cv::imwrite("resize_outs/output_" + std::to_string(batch->images[i]->frame_idx) + ".jpg", batch->images[i]->segInResizeMat);
            image_mats.push_back(batch->images[i]->segInResizeMat);
        } else {
            std::cerr << "⚠️ 图像 " << i << " 预处理结果为空" << std::endl;
            return false;
        }
    }
    
    // 使用第一个模型实例进行批量推理
    std::vector<SegmentationResult> seg_results;
    auto seg_start = std::chrono::high_resolution_clock::now();
    
    bool inference_success = seg_instances_[0]->Predict(image_mats, seg_results);
    
    auto seg_end = std::chrono::high_resolution_clock::now();
    auto seg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seg_end - seg_start);
    std::cout << "🧠 批次 " << batch->batch_id 
              << " 语义分割推理完成，耗时: " << seg_duration.count() << " ms" 
              << ", 实际图像数量: " << batch->actual_size << std::endl;
    if (!inference_success) {
        LOG_ERROR("❌ 批次推理失败");
        return false;
    }
    
    if (seg_results.size() != batch->actual_size) {
        std::cerr << "❌ 推理结果数量不匹配，期望: " << batch->actual_size 
                  << "，实际: " << seg_results.size() << std::endl;
        return false;
    }
    
    // 将推理结果分配给对应的图像
    for (size_t i = 0; i < batch->actual_size; ++i) {
        if (!seg_results[i].label_map.empty()) {
            batch->images[i]->label_map = std::move(seg_results[i].label_map);
            // cv::Mat mask(1024, 1024, CV_8UC1, batch->images[i]->label_map.data());
            // cv::imwrite("mask_outs/output_" + std::to_string(batch->images[i]->frame_idx) + ".jpg", mask*255);
            batch->images[i]->mask_height = 1024;
            batch->images[i]->mask_width = 1024;
            if(batch->images[i]->frame_idx % 200 == 0) {
                cv::Mat label_map(1024, 1024, CV_8UC1, (void*) batch->images[i]->label_map.data());
                // cv::imwrite(seg_show_image_path_+"/mask_" + std::to_string(batch->images[i]->frame_idx) + ".jpg", label_map*255);
                // 创建彩色mask：浅绿色 (BGR格式: 绿色为主)
                cv::Mat colored_mask = cv::Mat::zeros(1024, 1024, CV_8UC3);
                // 设置浅绿色 (B=100, G=255, R=100)
                colored_mask.setTo(cv::Scalar(0, 0, 255), label_map > 0);
                cv::Mat blended_result;
                cv::addWeighted(batch->images[i]->segInResizeMat, 0.4, colored_mask, 0.6, 0, blended_result);
                cv::imwrite(seg_show_image_path_+"/output_" + std::to_string(batch->images[i]->frame_idx) + ".jpg", blended_result);
            }
        } else {
            std::cerr << "⚠️ 图像 " << i << " 分割结果为空" << std::endl;
            batch->images[i]->mask_height = 1024;
            batch->images[i]->mask_width = 1024;
        }
        batch->images[i]->segmentation_completed = true;
    }

    return true;
}

void BatchSemanticSegmentation::postprocess_batch(BatchPtr batch) {
    
}

void BatchSemanticSegmentation::preprocess_image(ImageDataPtr image, int thread_id) {
    if (!image || image->imageMat.empty()) {
        return;
    }
    
    try {
        // 计算停车检测所需的缩放比例（长边到1920）
        int max_dim = std::max(image->imageMat.rows, image->imageMat.cols);
        double parking_scale = 640.0 / max_dim;
        cv::Size parking_size(static_cast<int>(image->imageMat.cols * parking_scale),
                             static_cast<int>(image->imageMat.rows * parking_scale));
        
        if (false) {
            // 使用CUDA加速预处理，复用预分配的GPU缓存
            std::lock_guard<std::mutex> lock(gpu_mutex_);
            
            // 确保缓存大小足够
            if (gpu_src_cache_.rows < image->imageMat.rows || gpu_src_cache_.cols < image->imageMat.cols) {
                gpu_src_cache_.create(std::max(gpu_src_cache_.rows, image->imageMat.rows), 
                                    std::max(gpu_src_cache_.cols, image->imageMat.cols), 
                                    CV_8UC3);
            }
            
            // 上传到GPU缓存的ROI区域
            cv::cuda::GpuMat gpu_src_roi = gpu_src_cache_(cv::Rect(0, 0, image->imageMat.cols, image->imageMat.rows));
            gpu_src_roi.upload(image->imageMat);
            
            // 语义分割预处理：确保目标缓存大小为1024x1024
            if (gpu_dst_cache_.rows != 1024 || gpu_dst_cache_.cols != 1024) {
                gpu_dst_cache_.create(1024, 1024, CV_8UC3);
            }
            
            // 执行语义分割resize操作
            cv::cuda::resize(gpu_src_roi, gpu_dst_cache_, cv::Size(1024, 1024));
            
            // 下载语义分割结果
            gpu_dst_cache_.download(image->segInResizeMat);
            
            // 停车检测预处理：创建临时GPU Mat用于停车检测缩放
            cv::cuda::GpuMat gpu_parking_resized;
            cv::cuda::resize(gpu_src_roi, gpu_parking_resized, parking_size);
            
            // 下载停车检测结果
            gpu_parking_resized.download(image->parkingResizeMat);
            
        } else {
            // CPU预处理
            cv::resize(image->imageMat, image->segInResizeMat, cv::Size(1024, 1024));
            cv::resize(image->imageMat, image->parkingResizeMat, parking_size);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "❌ 图像预处理失败: " << e.what() << std::endl;
        // 创建空的预处理结果避免后续处理失败
        image->segInResizeMat = cv::Mat::zeros(1024, 1024, CV_8UC3);
    }
}

void BatchSemanticSegmentation::save_segmentation_result(ImageDataPtr image) {
    if (!enable_seg_show_ || seg_show_image_path_.empty()) {
        return;
    }
    
    try {
        // 这里可以添加分割结果可视化和保存逻辑
        std::cout << "💾 保存分割结果，帧序号: " << image->frame_idx << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ 保存分割结果失败: " << e.what() << std::endl;
    }
}

bool BatchSemanticSegmentation::initialize_seg_models() {
    seg_instances_.clear();
    seg_instances_.reserve(num_threads_);
    
    PPSegInitParameters init_params;
    init_params.model_path = config_.seg_model_path.empty() ? "seg_model" : config_.seg_model_path;
    
    // 为每个线程创建独立的模型实例
    for (int i = 0; i < num_threads_; ++i) {
        auto seg_instance = CreatePureTRTPPSeg();
        int init_result = seg_instance->Init(init_params);
        
        if (init_result < 0) {
            std::cerr << "❌ 语义分割模型初始化失败，线程 " << i << std::endl;
            return false;
        } else {
            std::cout << "✅ 语义分割模型初始化成功，线程 " << i << std::endl;
        }
        
        seg_instances_.push_back(std::move(seg_instance));
    }
    
    return true;
}

void BatchSemanticSegmentation::cleanup_seg_models() {
    for (auto& instance : seg_instances_) {
        if (instance) {
            // instance->Release();
        }
    }
    seg_instances_.clear();
}

// BatchStage接口实现
std::string BatchSemanticSegmentation::get_stage_name() const {
    return "批次语义分割";
}

size_t BatchSemanticSegmentation::get_processed_count() const {
    return processed_batch_count_.load();
}

double BatchSemanticSegmentation::get_average_processing_time() const {
    auto count = processed_batch_count_.load();
    if (count == 0) return 0.0;
    return (double)total_processing_time_ms_.load() / count;
}

size_t BatchSemanticSegmentation::get_queue_size() const {
    return input_connector_->get_queue_size();
}

void BatchSemanticSegmentation::change_params(const PipelineConfig& config) {
    config_ = config;
    enable_seg_show_ = config.enable_seg_show;
    seg_show_image_path_ = config.seg_show_image_path;
}
