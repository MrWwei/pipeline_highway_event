#include "object_detection.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

const int det_batch_size = 8;
ObjectDetection::ObjectDetection(int num_threads)
    : ImageProcessor(num_threads, "目标检测"), stop_worker_(false) {

  // 初始化处理队列
  detection_queue_ =
      std::make_unique<ThreadSafeQueue<ImageDataPtr>>(100); // 设置队列容量为100

  // 基类已经完成了初始化工作
  car_track_instance_ = xtkj::createTracker(30, 30, 0.5, 0.6, 0.8);
  AlgorConfig config;
  config.algorName_ = "object_detect";
  config.model_path = "car_detect.onnx";
  config.img_size = 640;
  config.conf_thresh = 0.25f;
  config.iou_thresh = 0.2f;
  config.max_batch_size = det_batch_size;
  config.min_opt = 1;
  config.mid_opt = 16;
  config.max_opt = 32;
  config.is_ultralytics = 1;
  config.gpu_id = 0;

  // 初始化检测器
  car_detect_instance_ = xtkj::createDetect();
  car_detect_instance_->init(config);

  // 启动工作线程
  worker_thread_ = std::thread(&ObjectDetection::detection_worker, this);

  std::cout << "🔍 目标检测模块初始化完成" << std::endl;
}

void ObjectDetection::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }
  detection_queue_->push(image);
  // std::this_thread::sleep_for(std::chrono::milliseconds(10000)); // 间隔2ms
  // image->detection_promise->set_value();
}

void ObjectDetection::on_processing_start(ImageDataPtr image, int thread_id) {
  std::cout << "🎯 目标检测准备开始 (线程 " << thread_id << ")" << std::endl;
}

void ObjectDetection::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  std::cout << "🎯 目标检测处理完成 (线程 " << thread_id << ")" << std::endl;
}

void ObjectDetection::perform_object_detection(ImageDataPtr image,
                                               int thread_id) {
  // 保留单图接口，实际不再直接调用
}

void ObjectDetection::detection_worker() {
  while (!stop_worker_) {
    try {
      // 先将所有可用帧放入排序队列
      while (!detection_queue_->empty()) {
        ImageDataPtr img;
        detection_queue_->wait_and_pop(img);
        frame_order_queue_.push(img);
      }

      // 当积累了足够的帧或等待超时时处理
      if (frame_order_queue_.size() >= det_batch_size) {
        // 按序批量处理
        std::vector<ImageDataPtr> batch_images;

        // 按序批量取出数据
        for (int i = 0; i < det_batch_size && !frame_order_queue_.empty();
             ++i) {
          ImageDataPtr img = frame_order_queue_.top();
          frame_order_queue_.pop();
          if (!img) {
            throw std::runtime_error("批处理中存在无效的图像数据");
          }
          batch_images.push_back(img);
        }

        // 等待所有图像的mask后处理完成
        for (auto &img : batch_images) {
          try {
            img->mask_postprocess_future.get(); // 等待mask后处理完成
          } catch (const std::exception &e) {
            throw std::runtime_error("Mask后处理失败: " +
                                     std::string(e.what()));
          }
        }

        // 构建批量输入
        std::vector<cv::Mat> mats;
        for (auto &img : batch_images) {
          cv::Mat cropped_image = (*img->imageMat)(img->roi);

          mats.push_back(cropped_image);
        }

        // 执行批量目标检测
        detect_result_group_t **outs =
            new detect_result_group_t *[batch_images.size()];
        for (size_t i = 0; i < batch_images.size(); ++i) {
          outs[i] = new detect_result_group_t();
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        try {
          car_detect_instance_->forward(mats, outs);

          // 处理每个图像的结果
          for (size_t i = 0; i < batch_images.size(); ++i) {
            car_track_instance_->track(outs[i], batch_images[i]->roi.width,
                                       batch_images[i]->roi.height);

            batch_images[i]->detection_results.clear();
            for (size_t j = 0; j < outs[i]->count; ++j) {
              batch_images[i]->detection_results.push_back(
                  {outs[i]->results[j].box.left + batch_images[i]->roi.x,
                   outs[i]->results[j].box.top + batch_images[i]->roi.y,
                   outs[i]->results[j].box.right + batch_images[i]->roi.x,
                   outs[i]->results[j].box.bottom + batch_images[i]->roi.y,
                   outs[i]->results[j].prop, outs[i]->results[j].cls_id,
                   outs[i]->results[j].track_id});
            }

            // 设置promise完成
            batch_images[i]->detection_promise->set_value();
          }

          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              end_time - start_time);
          std::cout << "✅ 批量目标检测完成，耗时: " << duration.count() << "ms"
                    << std::endl;

        } catch (const std::exception &e) {
          // 如果检测过程出错，设置所有图像的promise为异常状态
          for (auto &img : batch_images) {
            img->detection_promise->set_exception(std::current_exception());
          }
          throw;
        }

        // 清理内存
        for (size_t i = 0; i < batch_images.size(); ++i) {
          delete outs[i];
        }
        delete[] outs;

      } else {
        // 单个处理
        ImageDataPtr image;
        detection_queue_->wait_and_pop(image);

        if (!image) {
          throw std::runtime_error("无效的图像数据");
        }

        try {
          // 等待mask后处理完成
          image->mask_postprocess_future.get();

          cv::Mat cropped_image = (*image->imageMat)(image->roi);
          std::vector<cv::Mat> mats;
          mats.push_back(cropped_image);

          detect_result_group_t **outs = new detect_result_group_t *[1];
          outs[0] = new detect_result_group_t();

          auto start_time = std::chrono::high_resolution_clock::now();

          car_detect_instance_->forward(mats, outs);
          car_track_instance_->track(outs[0], image->roi.width,
                                     image->roi.height);

          image->detection_results.clear();
          for (size_t j = 0; j < outs[0]->count; ++j) {
            image->detection_results.push_back(
                {outs[0]->results[j].box.left + image->roi.x,
                 outs[0]->results[j].box.top + image->roi.y,
                 outs[0]->results[j].box.right + image->roi.x,
                 outs[0]->results[j].box.bottom + image->roi.y,
                 outs[0]->results[j].prop, outs[0]->results[j].cls_id,
                 outs[0]->results[j].track_id});
          }
          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              end_time - start_time);
          std::cout << "✅ 目标检测完成 (单图)，耗时: " << duration.count()
                    << "ms" << std::endl;

          // 设置promise完成
          image->detection_promise->set_value();

          // 清理内存
          delete outs[0];
          delete[] outs;

        } catch (const std::exception &e) {
          std::cerr << "目标检测失败: " << e.what() << std::endl;
          image->detection_promise->set_exception(std::current_exception());
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "目标检测工作线程异常: " << e.what() << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 避免死循环
    }
  }
}

ObjectDetection::~ObjectDetection() {
  stop_worker_ = true;
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  delete car_detect_instance_;
}
