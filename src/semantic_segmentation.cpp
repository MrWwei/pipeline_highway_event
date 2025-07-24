#include "semantic_segmentation.h"
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

SemanticSegmentation::SemanticSegmentation(int num_threads)
    : ImageProcessor(num_threads, "语义分割"), stop_worker_(false) {
  // 初始化处理队列
  segmentation_queue_ =
      std::make_unique<ThreadSafeQueue<ImageDataPtr>>(100); // 设置队列容量为100

  // 初始化模型
  SegInitParams init_params;
  init_params.model_path = "seg_model";
  init_params.enable_show = false; // 启用可视化
  init_params.seg_show_image_path = "./segmentation_results/";

  road_seg_instance_ = createRoadSeg();
  int init_result = road_seg_instance_->init_seg(init_params);
}

void SemanticSegmentation::process_image(ImageDataPtr image, int thread_id) {
  if (!image || !image->segInResizeMat) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }
  segmentation_queue_->push(image);
}

void SemanticSegmentation::on_processing_start(ImageDataPtr image,
                                               int thread_id) {
  // 为 segInResizeMat 分配内存
  if (!image->segInResizeMat) {
    image->segInResizeMat = new cv::Mat();
  }
  auto start_time = std::chrono::high_resolution_clock::now();
  cv::resize(*image->imageMat, *image->segInResizeMat, cv::Size(1024, 1024));
  return;
}

void SemanticSegmentation::on_processing_complete(ImageDataPtr image,
                                                  int thread_id) {
  // 可以在这里添加语义分割特有的后处理逻辑
  // 例如：结果验证、统计信息更新等
}

// 只负责入队
void SemanticSegmentation::perform_semantic_segmentation(ImageDataPtr image,
                                                         int thread_id) {

  // cv::Mat &segInMat = *image->segInResizeMat;
  // std::vector<cv::Mat *> image_ptrs;
  // image_ptrs.push_back(&segInMat);
  // SegInputParams input_params(image_ptrs);
  // SegResult seg_result;
  // road_seg_instance_->seg_road(input_params, seg_result);

  // // 检查结果是否有效
  // if (!seg_result.results.empty() &&
  // !seg_result.results[0].label_map.empty()) {
  //   image->label_map.resize(seg_result.results[0].label_map.size());
  //   std::copy(seg_result.results[0].label_map.begin(),
  //             seg_result.results[0].label_map.end(),
  //             image->label_map.begin());
  //   image->mask_height = segInMat.rows;
  //   image->mask_width = segInMat.cols;
  // } else {
  //   image->label_map.resize(image->mask_height * image->mask_width, 0);
  // }
  // image->segmentation_complete = true;
  return;
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
          
          if (!img->segInResizeMat) {
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
          image_ptrs.push_back(img->segInResizeMat);
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
              image->mask_height = image->segInResizeMat->rows;
              image->mask_width = image->segInResizeMat->cols;

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

        if (!image->segInResizeMat) {
          throw std::runtime_error("无效的图像数据");
        }

        try {
          // 执行单个分割
          std::vector<cv::Mat *> image_ptrs{image->segInResizeMat};
          SegInputParams input_params(image_ptrs);
          SegResult seg_result;
          std::cout << "单个处理帧序号: " << image->frame_idx << std::endl;
          if (road_seg_instance_->seg_road(input_params, seg_result) != 0) {
            throw std::runtime_error("语义分割执行失败");
          }

          // 检查并设置结果
          if (!seg_result.results.empty() &&
              !seg_result.results[0].label_map.empty()) {
            // 优化：使用移动语义避免拷贝大量数据
            image->label_map = std::move(seg_result.results[0].label_map);
            image->mask_height = image->segInResizeMat->rows;
            image->mask_width = image->segInResizeMat->cols;

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
