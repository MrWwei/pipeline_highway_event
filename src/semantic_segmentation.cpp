#include "semantic_segmentation.h"
#include "pipeline_manager.h"
#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <thread>

#include "thread_safe_queue.h"
const int batch_size = 16;
//析构函数
SemanticSegmentation::~SemanticSegmentation() {
  stop_worker_ = true;
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  delete road_seg_instance_;
}

SemanticSegmentation::SemanticSegmentation(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "语义分割"), stop_worker_(false) {
  // 初始化处理队列
  segmentation_queue_ =
      std::make_unique<ThreadSafeQueue<ImageDataPtr>>(100); // 设置队列容量为100

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
  std::lock_guard<std::mutex> lock(seg_show_mutex_);
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
  segmentation_queue_->push(image);
}

void SemanticSegmentation::on_processing_start(ImageDataPtr image,
                                               int thread_id) {
  // Resize the image for segmentation processing
  auto start_time = std::chrono::high_resolution_clock::now();
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

// 队列处理线程
void SemanticSegmentation::segmentation_worker() {
  std::cout << "🔄 语义分割专用工作线程启动" << std::endl;
  
  while (!stop_worker_) {
    try {
      // 在循环开始时再次检查
      if (stop_worker_) {
        break;
      }
      
      // 检查队列大小决定使用批处理还是单个处理
      if (segmentation_queue_->size() >= batch_size && !stop_worker_) {
        // 批量处理
        std::vector<ImageDataPtr> batch_images;

        // 批量取出数据
        for (int i = 0; i < batch_size && !stop_worker_; ++i) {
          ImageDataPtr img;
          segmentation_queue_->wait_and_pop(img);
          
          // 检查是否是停止信号（空数据）
          if (!img) {
            if (stop_worker_) {
              break;  // 收到停止信号，退出批处理循环
            }
            continue;  // 忽略空数据，继续处理
          }
          
          if (img->segInResizeMat.empty()) {
            std::cerr << "⚠️ 批处理中发现无效的图像数据，跳过" << std::endl;
            continue;
          }
          batch_images.push_back(img);
        }
        
        // 如果收到停止信号或没有有效图像，退出
        if (stop_worker_ || batch_images.empty()) {
          break;
        }

        // 构建批量输入
        std::vector<cv::Mat *> image_ptrs;
        for (const auto &img : batch_images) {
          image_ptrs.push_back(&img->segInResizeMat);
        }

        // 执行批量分割
        SegInputParams input_params(image_ptrs);
        SegResult seg_result;
        
        if (road_seg_instance_->seg_road(input_params, seg_result) != 0) {
          throw std::runtime_error("批量语义分割执行失败");
        }

        // 处理每个图像的结果
        for (size_t idx = 0; idx < batch_images.size(); ++idx) {
          auto &image = batch_images[idx];
          try {
            if (seg_result.results.size() > idx &&
                !seg_result.results[idx].label_map.empty()) {
              // 优化：使用移动语义避免拷贝大量数据
              image->label_map = std::move(seg_result.results[idx].label_map);
              image->mask_height = image->segInResizeMat.rows;
              image->mask_width = image->segInResizeMat.cols;

              // 线程安全地检查是否需要保存分割结果
              {
                std::lock_guard<std::mutex> lock(seg_show_mutex_);
                seg_frame_counter_++;
                
                // 检查是否需要保存分割结果（手动启用或每200帧自动保存一次）
                // bool should_save_seg = enable_seg_show_ || (seg_frame_counter_ % seg_show_interval_ == 0);
                bool should_save_seg = enable_seg_show_;
                
                if (should_save_seg && !seg_show_image_path_.empty() && !image->label_map.empty()) {
                  // 将label_map vector转换为cv::Mat进行可视化
                  cv::Mat seg_mask(image->mask_height, image->mask_width, CV_8UC1, image->label_map.data());
                  cv::Mat seg_visualization;
                  cv::applyColorMap(seg_mask, seg_visualization, cv::COLORMAP_JET);
                  
                  // 保存分割结果图像
                  std::string filename = seg_show_image_path_ + "/seg_" + std::to_string(image->frame_idx) + ".jpg";
                  cv::imwrite(filename, seg_visualization);
                  
                  // 如果是自动保存（每200帧），输出提示信息
                  if (!enable_seg_show_) {
                    std::cout << "🎨 自动保存分割结果 (第" << seg_frame_counter_ << "帧): " << filename << std::endl;
                  }
                }
              }

              // 通知完成 - 先检查是否已经设置
              try {
                if (image->segmentation_promise && 
                    image->segmentation_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                  image->segmentation_promise->set_value();
                }
              } catch (const std::future_error& e) {
                std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
              }
            } else {
              throw std::runtime_error("无效的批处理结果");
            }
          } catch (const std::exception &e) {
            std::cerr << "处理批量结果 " << idx << " 失败: " << e.what()
                      << std::endl;
            try {
              if (image->segmentation_promise && 
                  image->segmentation_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                image->segmentation_promise->set_exception(
                    std::current_exception());
              }
            } catch (const std::future_error& e) {
              std::cout << "⚠️ Promise异常已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
            }
          }
        }
      } else {
        // 单个处理
        ImageDataPtr image;
        segmentation_queue_->wait_and_pop(image);

        // 检查是否是停止信号（空数据）
        if (!image) {
          if (stop_worker_) {
            break;  // 收到停止信号，退出循环
          }
          continue;  // 忽略空数据，继续处理
        }

        if (image->segInResizeMat.empty()) {
          throw std::runtime_error("无效的图像数据");
        }

        try {
          // 执行单个分割
          std::vector<cv::Mat *> image_ptrs{&image->segInResizeMat};
          SegInputParams input_params(image_ptrs);
          SegResult seg_result;
          // std::cout << "单个处理帧序号: " << image->frame_idx << std::endl;
          if (road_seg_instance_->seg_road(input_params, seg_result) != 0) {
            throw std::runtime_error("语义分割执行失败");
          }

          // 检查并设置结果
          if (!seg_result.results.empty() &&
              !seg_result.results[0].label_map.empty()) {
            // 优化：使用移动语义避免拷贝大量数据
            image->label_map = std::move(seg_result.results[0].label_map);
            image->mask_height = image->segInResizeMat.rows;
            image->mask_width = image->segInResizeMat.cols;

            // 线程安全地检查是否需要保存分割结果
            {
              std::lock_guard<std::mutex> lock(seg_show_mutex_);
              seg_frame_counter_++;
              
              // 检查是否需要保存分割结果（手动启用或每200帧自动保存一次）
              bool should_save_seg = enable_seg_show_ || (seg_frame_counter_ % seg_show_interval_ == 0);
              
              if (should_save_seg && !seg_show_image_path_.empty() && !image->label_map.empty()) {
                // 将label_map vector转换为cv::Mat进行可视化
                cv::Mat seg_mask(image->mask_height, image->mask_width, CV_8UC1, image->label_map.data());
                cv::Mat seg_visualization;
                cv::applyColorMap(seg_mask, seg_visualization, cv::COLORMAP_JET);
                
                // 保存分割结果图像
                std::string filename = seg_show_image_path_ + "/seg_" + std::to_string(image->frame_idx) + ".jpg";
                cv::imwrite(filename, seg_visualization);
                
                // 如果是自动保存（每200帧），输出提示信息
                if (!enable_seg_show_) {
                  std::cout << "🎨 自动保存分割结果 (第" << seg_frame_counter_ << "帧): " << filename << std::endl;
                }
              }
            }

            // 通知完成 - 先检查是否已经设置
            try {
              if (image->segmentation_promise && 
                  image->segmentation_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                image->segmentation_promise->set_value();
              }
            } catch (const std::future_error& e) {
              std::cout << "⚠️ Promise已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
            }
          } else {
            throw std::runtime_error("语义分割结果无效");
          }
        } catch (const std::exception &e) {
          std::cerr << "单个处理失败: " << e.what() << std::endl;
          try {
            if (image->segmentation_promise && 
                image->segmentation_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
              image->segmentation_promise->set_exception(std::current_exception());
            }
          } catch (const std::future_error& e) {
            std::cout << "⚠️ Promise异常已被设置，帧 " << image->frame_idx << ": " << e.what() << std::endl;
          }
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "语义分割工作线程异常: " << e.what() << std::endl;
      // 检查是否应该停止
      if (stop_worker_) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 避免死循环
    }
  }
  
  std::cout << "🔄 语义分割工作线程正在退出..." << std::endl;
}
