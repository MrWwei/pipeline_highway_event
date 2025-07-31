#include "semantic_segmentation.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

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
  // Resize the image for segmentation processing
  cv::resize(image->imageMat, image->segInResizeMat, cv::Size(1024, 1024));
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
    
    // 第二步：非阻塞方式收集剩余图像，直到达到批处理大小或队列为空
    ImageDataPtr image;
    while (batch_images.size() < BATCH_SIZE && running_.load()) {
      if (input_queue_.try_pop(image)) {
        if (image) {
          batch_images.push_back(image);
        }
      } else {
        // 队列为空，检查当前批次大小决定处理方式
        break;
      }
    }
    
    // 第三步：根据批次大小决定处理方式
    if (batch_images.size() >= BATCH_SIZE) {
      // 批量处理（满批次）
      std::cout << "🔄 语义分割线程 " << thread_id << " 开始批量处理 " 
                << batch_images.size() << " 张图像（满批次）" << std::endl;
      
      process_images_batch(batch_images, thread_id);
      
      std::cout << "✅ 语义分割线程 " << thread_id << " 批量处理完成，输出 " 
                << batch_images.size() << " 张图像" << std::endl;
    } else {
      // 单张处理（不足批次大小）
      std::cout << "🔄 语义分割线程 " << thread_id << " 开始单张处理 " 
                << batch_images.size() << " 张图像（不足批次）" << std::endl;
      
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
  
  // 预处理：resize所有图像
  for (auto& image : images) {
    on_processing_start(image, thread_id);
    image_ptrs.push_back(&image->segInResizeMat);
  }
  
  // 批量语义分割处理
  SegInputParams input_params(image_ptrs);
  SegResult seg_result;
  if (road_seg_instance_->seg_road(input_params, seg_result) != 0) {
    std::cerr << "❌ 批量语义分割执行失败" << std::endl;
    return;
  }
  
  // 处理批量结果
  if (seg_result.results.size() != images.size()) {
    std::cerr << "❌ 批量分割结果数量不匹配，期望: " << images.size() 
              << "，实际: " << seg_result.results.size() << std::endl;
    return;
  }
  
  // 将结果分配给对应的图像
  for (size_t i = 0; i < images.size(); ++i) {
    if (!seg_result.results[i].label_map.empty()) {
      images[i]->label_map = std::move(seg_result.results[i].label_map);
      images[i]->mask_height = images[i]->segInResizeMat.rows;
      images[i]->mask_width = images[i]->segInResizeMat.cols;
      images[i]->segmentation_completed = true;
    }
    
    // 后处理
    on_processing_complete(images[i], thread_id);
  }
}
