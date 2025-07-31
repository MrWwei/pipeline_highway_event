#include "object_detection.h"
#include "pipeline_manager.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
const int det_batch_size = 8;
ObjectDetection::ObjectDetection(int num_threads, const PipelineConfig* config)
    : ImageProcessor(num_threads, "目标检测"), config_(*config) { // 使用传入的线程数

  AlgorConfig algor_config;
  
  // 使用配置参数，如果没有提供则使用默认值
  if (config) {
    algor_config.algorName_ = config->det_algor_name;
    algor_config.model_path = config->car_det_model_path;
    algor_config.img_size = config->det_img_size;
    algor_config.conf_thresh = config->det_conf_thresh;
    algor_config.iou_thresh = config->det_iou_thresh;
    algor_config.max_batch_size = config->det_max_batch_size;
    algor_config.min_opt = config->det_min_opt;
    algor_config.mid_opt = config->det_mid_opt;
    algor_config.max_opt = config->det_max_opt;
    algor_config.is_ultralytics = config->det_is_ultralytics;
    algor_config.gpu_id = config->det_gpu_id;
  } else {
    // 默认配置
    algor_config.algorName_ = "object_detect";
    algor_config.model_path = "car_detect.onnx";
    algor_config.img_size = 640;
    algor_config.conf_thresh = 0.25f;
    algor_config.iou_thresh = 0.2f;
    algor_config.max_batch_size = det_batch_size;
    algor_config.min_opt = 1;
    algor_config.mid_opt = 16;
    algor_config.max_opt = 32;
    algor_config.is_ultralytics = 1;
    algor_config.gpu_id = 0;
  }

  // 初始化检测器
  car_detect_instance_ = xtkj::createDetect();
  car_detect_instance_->init(algor_config);
  
  // 初始化行人检测器（如果启用）
  if (config && config->enable_pedestrian_detect) {
    personal_detect_instance_ = xtkj::createDetect();
    // 为行人检测使用单独的配置，只修改模型路径
    AlgorConfig person_config = algor_config; // 复制车辆检测配置
    person_config.model_path = config->pedestrian_det_model_path;
    personal_detect_instance_->init(person_config);
  } else {
    personal_detect_instance_ = nullptr;
  }
  // std::cout << "🔍 目标检测模块初始化完成（正常模式）" << std::endl;
}


void ObjectDetection::process_image(ImageDataPtr image, int thread_id) {
  if (!image) {
    std::cerr << "Error: Invalid image data in process_image" << std::endl;
    return;
  }
  
  try {
    bool seg_enabled = config_.enable_segmentation;
    bool mask_enabled = config_.enable_mask_postprocess;
    
    // 只有在语义分割和mask后处理都启用时才等待mask后处理完成
    if (seg_enabled && mask_enabled) {
      // 等待mask后处理完成
      while (!image->mask_postprocess_completed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } else {
      // 如果语义分割或mask后处理被禁用，直接标记完成
      image->mask_postprocess_completed = true;
    }
    
    // 单帧检测处理
    cv::Mat cropped_image = (image->imageMat)(image->roi);
    int total_detections = 0;
    
    // 车辆检测
    detect_result_group_t car_out;
    std::vector<cv::Mat> mats = {cropped_image};
    detect_result_group_t* car_outs[] = {&car_out};
    
    car_detect_instance_->forward(mats, car_outs);
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // 处理车辆检测结果 (class_id保持原值，通常是0)
    if (car_out.count > 0) {
      for (int i = 0; i < car_out.count; ++i) {
        detect_result_t &result = car_out.results[i];
        image->detection_results.push_back({
            result.box.left+image->roi.x, result.box.top+image->roi.y, 
            result.box.right+image->roi.x, result.box.bottom+image->roi.y,
            result.prop, result.cls_id, result.track_id});
      }
      total_detections += car_out.count;
    }
    
    // 行人检测（如果启用）
    if (personal_detect_instance_) {
      detect_result_group_t person_out;
      detect_result_group_t* person_outs[] = {&person_out};
      
      personal_detect_instance_->forward(mats, person_outs);
      
      // 处理行人检测结果 (class_id设置为1)
      if (person_out.count > 0) {
        for (int i = 0; i < person_out.count; ++i) {
          detect_result_t &result = person_out.results[i];
          image->detection_results.push_back({
              result.box.left+image->roi.x, result.box.top+image->roi.y, 
              result.box.right+image->roi.x, result.box.bottom+image->roi.y,
              result.prop, 1, result.track_id}); // 行人检测类别ID设置为1
        }
        total_detections += person_out.count;
      }
    }
    
    // 标记检测完成
    image->detection_completed = true;
    
    
  } catch (const std::exception& e) {
    std::cerr << "❌ 目标检测处理失败，帧 " << image->frame_idx << ": " << e.what() << std::endl;
    // 标记检测完成避免阻塞
    image->detection_completed = true;
  }
}

void ObjectDetection::on_processing_start(ImageDataPtr image, int thread_id) {
  // std::cout << "🎯 目标检测准备开始 (线程 " << thread_id << ")" << std::endl;
  int max_dim = std::max(image->width, image->height);
  if (max_dim > 1920) {
    // 如果图像尺寸超过1080p，缩小到1080p
    double scale = 1920.0 / max_dim;
    cv::resize(image->imageMat, image->parkingResizeMat, cv::Size(), scale, scale, cv::INTER_LINEAR);
  } else {
    // 否则保持原尺寸
    image->parkingResizeMat = image->imageMat.clone();
  }
}

void ObjectDetection::on_processing_complete(ImageDataPtr image,
                                             int thread_id) {
  // std::cout << "🎯 目标检测处理完成 (线程 " << thread_id << ")" << std::endl;
}

void ObjectDetection::perform_object_detection(ImageDataPtr image,
                                               int thread_id) {
  // 保留单图接口，实际不再直接调用
}

ObjectDetection::~ObjectDetection() {
  // std::cout << "🔄 正在停止目标检测模块..." << std::endl;
  
  if (car_detect_instance_) {
    delete car_detect_instance_;
    car_detect_instance_ = nullptr;
  }
  
  if (personal_detect_instance_) {
    delete personal_detect_instance_;
    personal_detect_instance_ = nullptr;
  }
  
  // std::cout << "✅ 目标检测模块已停止" << std::endl;
}
