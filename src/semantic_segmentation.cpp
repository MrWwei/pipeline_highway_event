#include "semantic_segmentation.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "thread_safe_queue.h"
const int batch_size = 16;
//析构函数
SemanticSegmentation::~SemanticSegmentation() {
  delete road_seg_instance_;
}

SemanticSegmentation::SemanticSegmentation(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "语义分割", BATCH_SIZE, 100) { // 输入队列设为32，输出队列保持100

  // 初始化模型
  SegInitParams init_params;
  
  // 使用配置参数，如果没有提供则使用默认值
  if (config) {
    init_params.model_path = config->seg_model_path;
    init_params.enable_show = config->enable_seg_show;
    init_params.seg_show_image_path = config->seg_show_image_path;
  } else {
    // 默认配置
    init_params.model_path = "seg_model";
    init_params.enable_show = false;
    init_params.seg_show_image_path = "./segmentation_results/";
  }

  road_seg_instance_ = createRoadSeg();
  int init_result = road_seg_instance_->init_seg(init_params);
  
  // 初始化CUDA状态
  try {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
      // 预分配GPU内存以提高性能
      gpu_src_cache_.create(1080, 1920, CV_8UC3); // 假设最大输入尺寸
      gpu_dst_cache_.create(1024, 1024, CV_8UC3); // 目标尺寸
      cuda_available_ = true;
      std::cout << "✅ CUDA已启用，语义分割将使用GPU加速" << std::endl;
    } else {
      cuda_available_ = false;
      std::cout << "⚠️ 未检测到CUDA设备，语义分割将使用CPU" << std::endl;
    }
  } catch (const cv::Exception& e) {
    cuda_available_ = false;
    std::cerr << "⚠️ CUDA初始化失败: " << e.what() << "，将使用CPU" << std::endl;
  }
}

void SemanticSegmentation::set_seg_show_interval(int interval) {
  if (interval > 0) {
    seg_show_interval_ = interval;
    std::cout << "🎯 分割结果保存间隔已设置为: " << interval << " 帧" << std::endl;
  }
}

void SemanticSegmentation::process_image(ImageDataPtr image, int thread_id) {
  if (!image || image->imageMat.empty()) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }

  // 预处理：调用 on_processing_start
  on_processing_start(image, thread_id);

  // 执行单个分割
  std::vector<cv::Mat *> image_ptrs{&image->segInResizeMat};
  SegInputParams input_params(image_ptrs);
  
  SegResult seg_result;
  // std::cout << "单个处理帧序号: " << image->frame_idx << std::endl;
  if (road_seg_instance_->seg_road(input_params, seg_result) != 0) {
    std::cerr << "❌ 语义分割执行失败，帧序号: " << image->frame_idx << std::endl;
    return;
  }

  // 检查并设置结果
  if (!seg_result.results.empty() &&
      !seg_result.results[0].label_map.empty()) {
    // 优化：使用移动语义避免拷贝大量数据
    image->label_map = std::move(seg_result.results[0].label_map);
    image->mask_height = image->segInResizeMat.rows;
    image->mask_width = image->segInResizeMat.cols;

    // 标记分割完成
    image->segmentation_completed = true;
  }

  // 后处理：调用 on_processing_complete
  on_processing_complete(image, thread_id);
}

void SemanticSegmentation::on_processing_start(ImageDataPtr image,
                                               int thread_id) {
  // 使用CUDA进行图像resize处理（带缓存优化）
  if (cuda_available_) {
    try {
      std::lock_guard<std::mutex> lock(gpu_mutex_); // 保护GPU操作
      
      // 检查是否需要调整缓存大小
      if (gpu_src_cache_.rows < image->imageMat.rows || 
          gpu_src_cache_.cols < image->imageMat.cols) {
        gpu_src_cache_.create(image->imageMat.rows, image->imageMat.cols, CV_8UC3);
      }
      
      // 上传到GPU (只上传实际需要的区域)
      cv::cuda::GpuMat gpu_src_roi = gpu_src_cache_(cv::Rect(0, 0, image->imageMat.cols, image->imageMat.rows));
      gpu_src_roi.upload(image->imageMat);
      
      // 在GPU上进行resize操作
      cv::cuda::resize(gpu_src_roi, gpu_dst_cache_, cv::Size(1024, 1024));
      
      // 下载回CPU
      gpu_dst_cache_.download(image->segInResizeMat);
      
    } catch (const cv::Exception& e) {
      // 如果CUDA操作失败，标记CUDA不可用并回退到CPU实现
      std::cerr << "⚠️ CUDA resize失败，禁用CUDA并回退到CPU: " << e.what() << std::endl;
      cuda_available_ = false;
      cv::resize(image->imageMat, image->segInResizeMat, cv::Size(1024, 1024));
    }
  } else {
    // 使用CPU实现
    cv::resize(image->imageMat, image->segInResizeMat, cv::Size(1024, 1024));
  }
  return;
}

void SemanticSegmentation::on_processing_complete(ImageDataPtr image,
                                                  int thread_id) {
  // 可以在这里添加语义分割特有的后处理逻辑
  // 例如：结果验证、统计信息更新等
}

  void SemanticSegmentation::change_params(const PipelineConfig &config)  {
    if (config.enable_seg_show) {
      enable_seg_show_ = config.enable_seg_show;
      seg_show_image_path_ = config.seg_show_image_path;
      SegInitParams update_params;
      update_params.enable_show = enable_seg_show_;
      update_params.seg_show_image_path = seg_show_image_path_;
      road_seg_instance_->change_params(update_params);
    }
  }

// 重写工作线程函数以支持批量处理
void SemanticSegmentation::worker_thread_func(int thread_id) {
  std::cout << "🔄 " << processor_name_ << "批量工作线程 " << thread_id << " 启动"
            << std::endl;

  while (running_.load()) {
    std::vector<ImageDataPtr> batch_images;
    batch_images.reserve(BATCH_SIZE);
    
    // 第一步：阻塞等待第一个图像
    ImageDataPtr first_image;
    input_queue_.wait_and_pop(first_image);
    
    // 检查停止信号
    if (!first_image) {
      if (!running_.load()) {
        break;
      }
      continue;
    }
    
    batch_images.push_back(first_image);
    
    // 第二步：非阻塞方式收集剩余图像，带超时机制
    ImageDataPtr image;
    auto collection_start = std::chrono::high_resolution_clock::now();
    const auto timeout_ms = std::chrono::milliseconds(10); // 10ms超时
    
    while (batch_images.size() < BATCH_SIZE && running_.load()) {
      if (input_queue_.try_pop(image)) {
        if (image) {
          batch_images.push_back(image);
        }
      } else {
        // 检查是否超时
        auto now = std::chrono::high_resolution_clock::now();
        if (now - collection_start > timeout_ms) {
          // std::cout << "⏱️ 批量收集超时，当前批次: " << batch_images.size() << std::endl;
          break;
        }
        // 短暂休眠，避免占用过多CPU
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    
    // 第三步：智能选择处理方式
    const size_t min_batch_for_optimization = 4; // 最小优化批次大小
    
    if (batch_images.size() >= min_batch_for_optimization) {
      // 批量处理（优化批次）
      std::cout << "🔄 语义分割线程 " << thread_id << " 开始批量处理 " 
                << batch_images.size() << " 张图像" 
                << (batch_images.size() >= BATCH_SIZE ? "（满批次）" : "（优化批次）") << std::endl;
      
      process_images_batch(batch_images, thread_id);
      
      std::cout << "✅ 语义分割线程 " << thread_id << " 批量处理完成，输出 " 
                << batch_images.size() << " 张图像" << std::endl;
    } else {
      // 单张处理（小批次）
      std::cout << "🔄 语义分割线程 " << thread_id << " 开始单张处理 " 
                << batch_images.size() << " 张图像（小批次）" << std::endl;
      
      auto start_time = std::chrono::high_resolution_clock::now();
      
      for (auto& single_image : batch_images) {
        process_image(single_image, thread_id);
      }
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      
      std::cout << "✅ 语义分割线程 " << thread_id << " 单张处理完成，用时: " 
                << duration.count() << "ms，处理 " << batch_images.size() << " 张图像" << std::endl;
    }
    
    // 第四步：将所有处理结果添加到输出队列
    for (auto& processed_image : batch_images) {
      output_queue_.push(processed_image);
    }
  }
  
  std::cout << "🔄 " << processor_name_ << "批量工作线程 " << thread_id << " 退出"
            << std::endl;
}

// 批量处理方法
void SemanticSegmentation::process_images_batch(std::vector<ImageDataPtr>& images, int thread_id) {
  if (images.empty()) {
    return;
  }
  
  // 准备批量输入数据
  std::vector<cv::Mat*> image_ptrs;
  image_ptrs.reserve(images.size());
  
  auto preprocess_start = std::chrono::high_resolution_clock::now();
  
  // 批量预处理：使用CUDA流的真正并行批量处理
  if (cuda_available_ && images.size() > 1) {
    try {
      std::lock_guard<std::mutex> lock(gpu_mutex_);
      
      // 创建CUDA流用于异步操作
      cv::cuda::Stream streams[4]; // 使用4个流进行并行处理
      const int num_streams = std::min(4, static_cast<int>(images.size()));
      
      // 分配批量GPU内存
      std::vector<cv::cuda::GpuMat> gpu_src_batch(images.size());
      std::vector<cv::cuda::GpuMat> gpu_dst_batch(images.size());
      
      // 异步上传所有图像到GPU
      for (size_t i = 0; i < images.size(); ++i) {
        int stream_idx = i % num_streams;
        gpu_src_batch[i].create(images[i]->imageMat.rows, images[i]->imageMat.cols, CV_8UC3);
        gpu_dst_batch[i].create(1024, 1024, CV_8UC3);
        
        // 异步上传
        gpu_src_batch[i].upload(images[i]->imageMat, streams[stream_idx]);
      }
      
      // 等待所有上传完成
      for (int i = 0; i < num_streams; ++i) {
        streams[i].waitForCompletion();
      }
      
      // 并行resize操作
      for (size_t i = 0; i < images.size(); ++i) {
        int stream_idx = i % num_streams;
        cv::cuda::resize(gpu_src_batch[i], gpu_dst_batch[i], cv::Size(1024, 1024), 0, 0, cv::INTER_LINEAR, streams[stream_idx]);
      }
      
      // 等待所有resize完成
      for (int i = 0; i < num_streams; ++i) {
        streams[i].waitForCompletion();
      }
      
      // 异步下载回CPU
      for (size_t i = 0; i < images.size(); ++i) {
        int stream_idx = i % num_streams;
        gpu_dst_batch[i].download(images[i]->segInResizeMat, streams[stream_idx]);
      }
      
      // 等待所有下载完成
      for (int i = 0; i < num_streams; ++i) {
        streams[i].waitForCompletion();
      }
      
      // 准备指针数组
      for (auto& image : images) {
        image_ptrs.push_back(&image->segInResizeMat);
      }
      
      std::cout << "🚀 并行CUDA流处理: " << images.size() << " 张图像，使用 " << num_streams << " 个流" << std::endl;
      
    } catch (const cv::Exception& e) {
      std::cerr << "⚠️ 并行CUDA流处理失败，回退到单张处理: " << e.what() << std::endl;
      cuda_available_ = false;
      // 回退到逐张处理
      image_ptrs.clear(); // 清空之前可能的部分结果
      for (auto& image : images) {
        on_processing_start(image, thread_id);
        image_ptrs.push_back(&image->segInResizeMat);
      }
    }
  } else {
    // 单张预处理或CUDA不可用
    for (auto& image : images) {
      on_processing_start(image, thread_id);
      image_ptrs.push_back(&image->segInResizeMat);
    }
  }
  
  auto preprocess_end = std::chrono::high_resolution_clock::now();
  auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
  
  // 批量语义分割处理
  SegInputParams input_params(image_ptrs);
  SegResult seg_result;
  auto seg_start = std::chrono::high_resolution_clock::now();
  if (road_seg_instance_->seg_road(input_params, seg_result) != 0) {
    std::cerr << "❌ 批量语义分割执行失败" << std::endl;
    return;
  }
  auto seg_end = std::chrono::high_resolution_clock::now();
  auto seg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seg_end - seg_start);
  
  std::cout << "⚡ 批量语义分割性能统计 - 预处理: " << preprocess_duration.count() 
            << "ms, 推理: " << seg_duration.count() 
            << "ms, 总计: " << (preprocess_duration + seg_duration).count() 
            << "ms, 处理 " << images.size() << " 张图像" << std::endl;
  
  // 处理批量结果
  if (seg_result.results.size() != images.size()) {
    std::cerr << "❌ 批量分割结果数量不匹配，期望: " << images.size() 
              << "，实际: " << seg_result.results.size() << std::endl;
    return;
  }
  
  // 批量后处理：快速结果分配
  auto postprocess_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < images.size(); ++i) {
    if (!seg_result.results[i].label_map.empty()) {
      images[i]->label_map = std::move(seg_result.results[i].label_map);
      images[i]->mask_height = 1024; // 固定值，避免重复访问
      images[i]->mask_width = 1024;  // 固定值，避免重复访问
      images[i]->segmentation_completed = true;
    }
    
    // 跳过后处理调用以提高性能（如果不需要的话）
    // on_processing_complete(images[i], thread_id);
  }
  auto postprocess_end = std::chrono::high_resolution_clock::now();
  auto postprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - postprocess_start);
  
  std::cout << "📊 后处理用时: " << postprocess_duration.count() << "ms" << std::endl;
}
