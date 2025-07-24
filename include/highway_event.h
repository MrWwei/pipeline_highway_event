#pragma once

#include "image_data.h"
#include "pipeline_manager.h"
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>

/**
 * 高速公路事件检测系统配置参数
 */
struct HighwayEventConfig {
    // 线程配置
    int semantic_threads = 8;      // 语义分割线程数
    int mask_threads = 8;          // Mask后处理线程数  
    int detection_threads = 8;     // 目标检测线程数
    int tracking_threads = 1;      // 目标跟踪线程数
    int filter_threads = 4;        // 目标框筛选线程数
    
    // 队列容量配置
    int input_queue_capacity = 100;    // 输入队列容量
    int result_queue_capacity = 500;   // 结果队列容量
    
    // 超时配置
    int add_timeout_ms = 5000;         // 添加数据超时时间(毫秒)
    int get_result_timeout_ms = 10000; // 获取结果超时时间(毫秒)
    
    // 语义分割模型配置
    std::string seg_model_path = "seg_model";               // 语义分割模型路径
    bool seg_enable_show = false;                           // 是否启用分割结果可视化
    std::string seg_show_image_path = "./segmentation_results/"; // 分割结果图像保存路径
    
    // 目标检测算法配置
    std::string det_algor_name = "object_detect";           // 算法名称
    std::string det_model_path = "car_detect.onnx";         // 目标检测模型路径
    int det_img_size = 640;                                 // 输入图像尺寸
    float det_conf_thresh = 0.25f;                          // 置信度阈值
    float det_iou_thresh = 0.2f;                            // NMS IoU阈值
    int det_max_batch_size = 16;                            // 最大批处理大小
    int det_min_opt = 1;                                    // 最小优化尺寸
    int det_mid_opt = 16;                                   // 中等优化尺寸
    int det_max_opt = 32;                                   // 最大优化尺寸
    int det_is_ultralytics = 1;                             // 是否使用Ultralytics格式
    int det_gpu_id = 0;                                     // GPU设备ID
    
    // 目标框筛选配置
    float box_filter_top_fraction = 4.0f / 7.0f;           // 筛选区域上边界比例
    float box_filter_bottom_fraction = 8.0f / 9.0f;        // 筛选区域下边界比例
    
    // 调试配置
    bool enable_debug_log = false;     // 是否启用调试日志
    bool enable_status_print = false;  // 是否启用状态打印
};

/**
 * 处理结果状态
 */
enum class ResultStatus {
    SUCCESS,        // 成功
    TIMEOUT,        // 超时
    NOT_FOUND,      // 未找到指定帧
    PIPELINE_STOPPED, // 流水线已停止
    ERROR           // 错误
};

/**
 * 获取结果的返回值
 */
struct GetResultReturn {
    ResultStatus status;
    ImageDataPtr result;
    
    GetResultReturn(ResultStatus s, ImageDataPtr r = nullptr) 
        : status(s), result(r) {}
};

/**
 * 高速公路事件检测系统接口
 * 
 * 提供简化的API来管理整个图像处理流水线
 */
class HighwayEventDetector {
private:
    std::unique_ptr<PipelineManager> pipeline_manager_;
    HighwayEventConfig config_;
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};
    std::atomic<uint64_t> next_frame_id_{0};
    
    // 结果管理
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::unordered_map<uint64_t, ImageDataPtr> completed_results_;
    
    // 内部结果处理线程
    std::thread result_thread_;
    std::atomic<bool> result_thread_running_{false};
    
    // 内部方法
    void result_processing_thread();
    void cleanup_old_results();

public:
    /**
     * 构造函数
     */
    HighwayEventDetector();
    
    /**
     * 析构函数
     */
    ~HighwayEventDetector();
    
    /**
     * 初始化流水线
     * @param config 配置参数
     * @return true 初始化成功，false 初始化失败
     */
    bool initialize(const HighwayEventConfig& config);
    
    /**
     * 启动流水线
     * @return true 启动成功，false 启动失败
     */
    bool start();
    
    /**
     * 向流水线添加图像数据
     * @param image_mat 输入图像（会被拷贝）
     * @return 返回该帧的序列号，如果失败返回-1
     * @note 如果第一阶段队列满了，此接口会阻塞等待
     */
    int64_t add_frame(const cv::Mat& image_mat);
    
    /**
     * 向流水线添加图像数据（移动语义版本）
     * @param image_mat 输入图像（会被移动）
     * @return 返回该帧的序列号，如果失败返回-1
     * @note 如果第一阶段队列满了，此接口会阻塞等待
     */
    int64_t add_frame(cv::Mat&& image_mat);
    
    /**
     * 向流水线添加图像数据（带超时）
     * @param image_mat 输入图像
     * @param timeout_ms 超时时间（毫秒），0表示不阻塞，-1表示无限等待
     * @return 返回该帧的序列号，如果失败或超时返回-1
     */
    int64_t add_frame_with_timeout(const cv::Mat& image_mat, int timeout_ms);
    
    /**
     * 获取指定帧序号的处理结果
     * @param frame_id 帧序列号
     * @return GetResultReturn 包含状态和结果数据
     * @note 如果结果还未准备好，此接口会阻塞等待
     */
    GetResultReturn get_result(uint64_t frame_id);
    
    /**
     * 获取指定帧序号的处理结果（带超时）
     * @param frame_id 帧序列号
     * @param timeout_ms 超时时间（毫秒），0表示不阻塞，-1表示无限等待
     * @return GetResultReturn 包含状态和结果数据
     */
    GetResultReturn get_result_with_timeout(uint64_t frame_id, int timeout_ms);
    
    /**
     * 尝试获取指定帧序号的处理结果（非阻塞）
     * @param frame_id 帧序列号
     * @return GetResultReturn 包含状态和结果数据
     */
    GetResultReturn try_get_result(uint64_t frame_id);
    
    /**
     * 停止流水线并释放资源
     * @return true 停止成功，false 停止失败
     */
    bool stop();
    
    /**
     * 获取流水线状态信息
     */
    void print_status() const;
    
    /**
     * 获取配置信息
     */
    const HighwayEventConfig& get_config() const { return config_; }
    
    /**
     * 检查是否已初始化
     */
    bool is_initialized() const { return is_initialized_.load(); }
    
    /**
     * 检查是否正在运行
     */
    bool is_running() const { return is_running_.load(); }
    
    /**
     * 获取当前待处理的帧数量
     */
    size_t get_pending_frame_count() const;
    
    /**
     * 获取已完成但未取走的结果数量
     */
    size_t get_completed_result_count() const;
    
    /**
     * 清理指定帧ID之前的所有结果（释放内存）
     * @param before_frame_id 清理此帧ID之前的所有结果
     */
    void cleanup_results_before(uint64_t before_frame_id);
};
