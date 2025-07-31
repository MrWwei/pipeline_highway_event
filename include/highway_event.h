#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <cstdint>
#include "box_event.h"
#include "image_data.h"
#include "pipeline_manager.h"
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <thread>

/**
 * 高速公路事件检测配置参数
 */
struct HighwayEventConfig {
    // === 线程配置 ===
    int semantic_threads = 2;              // 语义分割线程数
    int mask_threads = 1;                  // Mask后处理线程数
    int detection_threads = 2;             // 目标检测线程数
    int tracking_threads = 1;              // 目标跟踪线程数
    int filter_threads = 1;                // 目标框筛选线程数
    
    // === 模型配置 ===
    std::string seg_model_path = "seg_model";               // 语义分割模型路径
    std::string car_det_model_path = "car_detect.onnx";         // 目标检测模型路径
    std::string pedestrian_det_model_path = "Pedestrain_TAG1_yl_S640_V1.2.onnx"; // 行人检测模型路径
    
    // === 检测配置 ===
    std::string det_algor_name = "object_detect";           // 算法名称
    int det_img_size = 640;                                 // 输入图像尺寸
    float det_conf_thresh = 0.25f;                          // 置信度阈值
    float det_iou_thresh = 0.2f;                            // NMS IoU阈值
    int det_max_batch_size = 16;                            // 最大批处理大小
    int det_min_opt = 1;                                    // 最小优化尺寸
    int det_mid_opt = 16;                                   // 中等优化尺寸
    int det_max_opt = 32;                                   // 最大优化尺寸
    int det_is_ultralytics = 1;                             // 是否使用Ultralytics格式
    int det_gpu_id = 0;                                     // GPU设备ID
    
    // === 筛选配置 ===
    float box_filter_top_fraction = 4.0f / 7.0f;           // 筛选区域上边界比例
    float box_filter_bottom_fraction = 8.0f / 9.0f;        // 筛选区域下边界比例
    float times_car_width = 3.0f;                          // 车宽倍数，用于计算车道线位置
    
    // === 队列配置 ===
    int result_queue_capacity = 500;                        // 结果队列容量

    
    // === 模块开关配置 ===
    bool enable_segmentation = true;       // 启用语义分割模块
    bool enable_mask_postprocess = true;                   // 启用Mask后处理模块
    bool enable_detection = true;                           // 启用目标检测模块
    bool enable_tracking = true;                            // 启用目标跟踪模块
    bool enable_event_determine = true;                          // 启用目标框筛选模块

    bool enable_pedestrian_detect = false;                  // 是否启用行人检测
    
    // === 调试配置 ===
    bool enable_debug_log = false;                          // 启用调试日志
    bool enable_seg_show = false;                           // 启用分割结果可视化
    std::string seg_show_image_path = "./segmentation_results/"; // 分割结果图像保存路径
    bool enable_lane_show = false;                           // 启用车道线可视化
    std::string lane_show_image_path = "./lane_results/";   // 车道线结果图像保存路径
    
    // === 超时配置 ===
    int add_timeout_ms = 5000;                              // 添加帧超时时间（毫秒）
    int get_timeout_ms = 30000;                             // 获取结果超时时间（毫秒）
};

/**
 * 目标状态枚举
 */
// enum class ObjectStatus {
//     NORMAL = 0,                 // 正常状态
//     PARKING_LANE = 1,           // 违停
//     PARKING_EMERGENCY_LANE = 2, // 应急车道停车
//     OCCUPY_EMERGENCY_LANE = 3,  // 占用应急车道
//     WALK_HIGHWAY = 4,           // 高速行人
//     HIGHWAY_JAM = 5,            // 高速拥堵
//     TRAFFIC_ACCIDENT = 6        // 交通事故
// };

/**
 * 检测框结果
 */


/**
 * 处理结果状态
 */
enum class ResultStatus {
    SUCCESS = 0,        // 成功
    PENDING = 1,        // 处理中
    TIMEOUT = 2,        // 超时
    NOT_FOUND = 3,      // 帧未找到
    ERROR = 4           // 错误
};

/**
 * 处理结果
 */
struct ProcessResult {
    ResultStatus status;                    // 结果状态
    uint64_t frame_id;                     // 帧ID
    std::vector<DetectionBox> detections;   // 检测结果
    DetectionBox filtered_box;              // 筛选出的最佳目标框
    bool has_filtered_box;                  // 是否有筛选结果
    cv::Mat mask;                          // 语义分割掩码（可选）
    cv::Mat srcImage;                   // 源图像（可选）
    cv::Rect roi;                          // 感兴趣区域
    
    ProcessResult() : status(ResultStatus::PENDING), frame_id(0), 
                     has_filtered_box(false) {}
};

/**
 * 高速公路事件检测器 - 纯虚接口
 * 
 * 使用方法：
 * 1. 使用工厂函数 create_highway_event_detector() 创建实例
 * 2. 调用 initialize() 初始化流水线
 * 3. 调用 start() 启动流水线
 * 4. 调用 add_frame() 向流水线添加图像数据，返回帧序号
 * 5. 调用 get_result() 获取指定帧序号的处理结果
 * 6. 使用完毕后自动析构或显式调用 stop()
 */
class HighwayEventDetector {
public:
    /**
     * 虚析构函数
     */
    virtual ~HighwayEventDetector() = default;
    
    // 禁用拷贝构造和赋值
    HighwayEventDetector(const HighwayEventDetector&) = delete;
    HighwayEventDetector& operator=(const HighwayEventDetector&) = delete;
    
    /**
     * 初始化流水线
     * @param config 配置参数
     * @return 成功返回true，失败返回false
     */
    virtual bool initialize(const HighwayEventConfig& config = HighwayEventConfig()) = 0;

    virtual bool change_params(const HighwayEventConfig& config) = 0;
    
    /**
     * 启动流水线
     * @return 成功返回true，失败返回false
     */
    virtual bool start() = 0;

    
    /**
     * 添加图像数据到流水线
     * @param image 输入图像
     * @return 成功返回帧序号（>=0），失败返回-1
     */
    virtual int64_t add_frame(const cv::Mat& image) = 0;
    
    /**
     * 添加图像数据到流水线（移动语义）
     * @param image 输入图像（移动）
     * @return 成功返回帧序号（>=0），失败返回-1
     */
    virtual int64_t add_frame(cv::Mat&& image) = 0;
    
    /**
     * 获取指定帧序号的处理结果
     * @param frame_id 帧序号
     * @return 处理结果
     */
    virtual ProcessResult get_result(uint64_t frame_id) = 0;
    
    /**
     * 获取指定帧序号的处理结果（带超时）
     * @param frame_id 帧序号
     * @param timeout_ms 超时时间（毫秒）
     * @return 处理结果
     */
    virtual ProcessResult get_result_with_timeout(uint64_t frame_id, int timeout_ms) = 0;
    
    /**
     * 停止流水线
     */
    virtual void stop() = 0;
    
    /**
     * 检查是否已初始化
     * @return 已初始化返回true
     */
    virtual bool is_initialized() const = 0;
    
    /**
     * 检查是否正在运行
     * @return 正在运行返回true
     */
    virtual bool is_running() const = 0;
    
    /**
     * 获取当前配置
     * @return 配置参数的常量引用
     */
    virtual const HighwayEventConfig& get_config() const = 0;
    
    /**
     * 获取流水线状态统计信息
     * @return 状态信息字符串
     */
    virtual std::string get_pipeline_status() const = 0;

protected:
    /**
     * 受保护的构造函数，只允许派生类调用
     */
    HighwayEventDetector() = default;
};

/**
 * 工厂函数：创建高速公路事件检测器实例
 * @return 返回HighwayEventDetector的智能指针
 */
std::unique_ptr<HighwayEventDetector> create_highway_event_detector();

/**
 * 销毁函数：安全销毁检测器实例（用于C接口）
 * @param detector 要销毁的检测器指针
 */
void destroy_highway_event_detector(HighwayEventDetector* detector);
