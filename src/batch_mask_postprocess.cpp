#include "batch_mask_postprocess.h"
#include <iostream>
#include <algorithm>
#include "process_mask.h"
#include "event_utils.h"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <future>

BatchMaskPostProcess::BatchMaskPostProcess(int num_threads)
    : num_threads_(num_threads), running_(false), stop_requested_(false),
      min_area_threshold_(1000), morphology_kernel_size_(5), roi_expansion_ratio_(0.1) {
    
    // 创建线程池
    thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
    
    // 创建输入输出连接器
    input_connector_ = std::make_unique<BatchConnector>(10);
    output_connector_ = std::make_unique<BatchConnector>(10);
}

BatchMaskPostProcess::~BatchMaskPostProcess() {
    stop();
}

void BatchMaskPostProcess::start() {
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
        worker_threads_.emplace_back(&BatchMaskPostProcess::worker_thread_func, this);
    }
    
    std::cout << "✅ 批次Mask后处理已启动，使用 " << num_threads_ << " 个工作线程和线程池" << std::endl;
}

void BatchMaskPostProcess::stop() {
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
    
    // 等待所有工作线程结束
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "🛑 批次Mask后处理已停止" << std::endl;
}

bool BatchMaskPostProcess::add_batch(BatchPtr batch) {
    if (!running_.load() || !batch) {
        return false;
    }
    return input_connector_->send_batch(batch);
}

bool BatchMaskPostProcess::get_processed_batch(BatchPtr& batch) {
    return output_connector_->receive_batch(batch);
}

bool BatchMaskPostProcess::process_batch(BatchPtr batch) {
    if (!batch || batch->is_empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // std::cout << "🔧 开始处理批次 " << batch->batch_id 
    //           << " Mask后处理，包含 " << batch->actual_size << " 个图像" << std::endl;
    
    try {
        // 使用线程池并发处理批次中的所有图像
        bool success = process_batch_with_threadpool(batch);
        
        if (success) {
            // 标记批次完成
            batch->mask_postprocess_completed.store(true);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // 更新统计信息
            processed_batch_count_.fetch_add(1);
            total_processing_time_ms_.fetch_add(duration.count());
            total_images_processed_.fetch_add(batch->actual_size);
            
            // std::cout << "✅ 批次 " << batch->batch_id << " Mask后处理完成，耗时: " 
            //           << duration.count() << "ms，平均每张: " 
            //           << (double)duration.count() / batch->actual_size << "ms" << std::endl;
            
            return true;
        } else {
            std::cerr << "❌ 批次 " << batch->batch_id << " Mask后处理失败" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 批次 " << batch->batch_id << " Mask后处理异常: " << e.what() << std::endl;
        return false;
    }
}

bool BatchMaskPostProcess::process_batch_with_threadpool(BatchPtr batch) {
    if (!batch || batch->is_empty() || !thread_pool_ || !thread_pool_->is_running()) {
        return false;
    }
    
    std::vector<std::future<bool>> futures;
    futures.reserve(batch->actual_size);
    
    // 为批次中的每个图像提交任务到线程池
    for (int i = 0; i < batch->actual_size; ++i) {
        if (batch->images[i]) {
            try {
                auto future = thread_pool_->enqueue([this, image = batch->images[i]]() -> bool {
                    try {
                        this->process_image_mask(image);
                        return true;
                    } catch (const std::exception& e) {
                        std::cerr << "❌ 图像 " << image->frame_idx << " Mask后处理异常: " << e.what() << std::endl;
                        return false;
                    }
                });
                futures.push_back(std::move(future));
            } catch (const std::exception& e) {
                std::cerr << "❌ 无法提交图像 " << batch->images[i]->frame_idx << " 到线程池: " << e.what() << std::endl;
                return false;
            }
        }
    }
    
    // 等待所有任务完成
    bool all_success = true;
    for (auto& future : futures) {
        try {
            if (!future.get()) {
                all_success = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ 获取线程池任务结果异常: " << e.what() << std::endl;
            all_success = false;
        }
    }
    
    return all_success;
}

void BatchMaskPostProcess::worker_thread_func() {
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
                    std::cerr << "❌ 批次 " << batch->batch_id << " Mask后处理失败，丢弃" << std::endl;
                }
            }
        }
        
        if (stop_requested_.load()) {
            break;
        }
    }
}

void BatchMaskPostProcess::process_image_mask(ImageDataPtr image) {
    if (!image || image->label_map.empty()) {
        std::cerr << "⚠️ 图像或label_map为空，跳过Mask后处理" << std::endl;
        image->roi = cv::Rect(0, 0, image->width, image->height);
        image->mask_postprocess_completed = true;
        return;
    }
    
    try {
        // 将label_map转换为Mat格式
        cv::Mat mask(image->mask_height, image->mask_width, CV_8UC1, image->label_map.data());
        image->mask = remove_small_white_regions_cuda(mask);
        cv::threshold(image->mask, image->mask, 0, 255, cv::THRESH_BINARY);
        // cv::imwrite("mask_outs/processed_" + std::to_string(image->frame_idx) + ".jpg", image->mask);
        DetectRegion detect_region = crop_detect_region_optimized(
        image->mask, image->mask.rows, image->mask.cols);
        //将resize的roi映射回原图大小
        detect_region.x1 = static_cast<int>(detect_region.x1 * image->width /
                                            static_cast<double>(image->mask_width));
        detect_region.x2 = static_cast<int>(detect_region.x2 * image->width /
                                            static_cast<double>(image->mask_width));
        detect_region.y1 = static_cast<int>(detect_region.y1 * image->height /
                                            static_cast<double>(image->mask_height));
        detect_region.y2 = static_cast<int>(detect_region.y2 * image->height /
                                            static_cast<double>(image->mask_height));
        image->roi = cv::Rect(detect_region.x1, detect_region.y1,
                                detect_region.x2 - detect_region.x1,
                                detect_region.y2 - detect_region.y1);
        
        
        image->mask_postprocess_completed = true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "❌ Mask后处理失败: " << e.what() << std::endl;
        // 设置默认ROI
        image->roi = cv::Rect(0, 0, image->width, image->height);
        image->mask_postprocess_completed = true;
    }
}


// BatchStage接口实现
std::string BatchMaskPostProcess::get_stage_name() const {
    return "批次Mask后处理";
}

size_t BatchMaskPostProcess::get_processed_count() const {
    return processed_batch_count_.load();
}

double BatchMaskPostProcess::get_average_processing_time() const {
    auto count = processed_batch_count_.load();
    if (count == 0) return 0.0;
    return (double)total_processing_time_ms_.load() / count;
}

size_t BatchMaskPostProcess::get_queue_size() const {
    size_t input_queue_size = input_connector_->get_queue_size();
    size_t threadpool_queue_size = (thread_pool_ && thread_pool_->is_running()) ? thread_pool_->get_queue_size() : 0;
    return input_queue_size + threadpool_queue_size;
}
