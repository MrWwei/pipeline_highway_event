#ifndef PIPELINE_CONFIG_H
#define PIPELINE_CONFIG_H
#include <string>
struct PipelineConfig {
    // 线程配置
    int semantic_threads = 2;              // 语义分割线程数
    int mask_postprocess_threads = 1;      // Mask后处理线程数
    int detection_threads = 5;             // 目标检测线程数
    int tracking_threads = 1;              // 目标跟踪线程数
    int event_determine_threads = 1;       // 事件判定线程数
    
    // 模块开关配置
    bool enable_segmentation = true;       // 启用语义分割模块
    bool enable_mask_postprocess = true;  // 启用Mask后处理模块
    bool enable_detection = true;          // 启用目标检测模块
    bool enable_tracking = true;           // 启用目标跟踪模块
    bool enable_event_determine = true;    // 启用事件判定模块
    
    // 语义分割模型配置
    std::string seg_model_path = "seg_model";               // 语义分割模型路径
    bool enable_seg_show = false;                           // 是否启用分割结果可视化
    std::string seg_show_image_path = "./segmentation_results/"; // 分割结果图像保存路径
    
    // 目标检测算法配置
    std::string det_algor_name = "object_detect";           // 算法名称
    std::string car_det_model_path = "car_detect.onnx";         // 目标检测模型路径
    int det_img_size = 640;                                 // 输入图像尺寸
    float det_conf_thresh = 0.25f;                          // 置信度阈值
    float det_iou_thresh = 0.2f;                            // NMS IoU阈值
    int det_max_batch_size = 16;                            // 最大批处理大小
    int det_min_opt = 1;                                    // 最小优化尺寸
    int det_mid_opt = 16;                                   // 中等优化尺寸
    int det_max_opt = 32;                                   // 最大优化尺寸
    int det_is_ultralytics = 1;                             // 是否使用Ultralytics格式
    int det_gpu_id = 0;                                     // GPU设备ID
    bool enable_pedestrian_detect = false;                  // 是否启用行人检测
    std::string pedestrian_det_model_path = "person_detect.onnx"; // 行人检测模型路径
    
    // 事件判定配置
    float event_determine_top_fraction = 4.0f / 7.0f;           // 筛选区域上边界比例
    float event_determine_bottom_fraction = 8.0f / 9.0f;        // 筛选区域下边界比例
    float times_car_width = 3.0f;                          // 车宽倍数，用于计算车道线位置

    bool enable_lane_show = false; // 启用车道线可视化
    std::string lane_show_image_path = "./lane_results/";   // 车道
    
    // 队列配置
    int final_result_queue_capacity = 500; // 最终结果队列容量
};
#endif // PIPELINE_CONFIG_H