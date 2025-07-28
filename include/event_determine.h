#pragma once

#include "image_processor.h"
#include "thread_safe_queue.h"
#include "event_utils.h"
#include <mutex>

// 前向声明
struct PipelineConfig;

/**
 * 事件判定处理器
 * 负责从检测结果中筛选出特定区域内宽度最小的目标框
 * 并判定车辆是否占用应急车道等事件
 * 区域定义：图片从上开始的七分之二处到七分之六处
 * 如果该区域没有目标框，则在全图范围内寻找
 */
class EventDetermine : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  EventDetermine(int num_threads = 1, const PipelineConfig* config = nullptr);

  // 虚析构函数
  virtual ~EventDetermine();

  // 线程安全地设置车道线显示状态
  void set_lane_show_enabled(bool enabled, const std::string& save_path = "");

  // 设置车道线绘制间隔（帧数）
  void set_lane_show_interval(int interval);

  void change_params(const PipelineConfig& config)override;

protected:
  // 重写基类的纯虚函数：执行事件判定
  virtual void process_image(ImageDataPtr image, int thread_id) override;

  // 重写处理开始前的准备工作
  virtual void on_processing_start(ImageDataPtr image, int thread_id) override;

  // 重写处理完成后的清理工作
  virtual void on_processing_complete(ImageDataPtr image,
                                      int thread_id) override;

private:
  // 配置参数
  float top_fraction_;     // 筛选区域上边界比例
  float bottom_fraction_;  // 筛选区域下边界比例
  float times_car_width_ = 3.0f; // 车宽倍数，用于计算车道线位置
  bool enable_lane_show_ = false; // 是否启用车道线可视化
  std::string lane_show_image_path_; // 车道线结果图像保存路径
  
  // 线程安全锁
  std::mutex lane_show_mutex_;
  
  // 帧计数器和间隔控制
  int frame_counter_ = 0;
  int lane_show_interval_ = 200; // 默认每200帧自动绘制一次车道线

  /**
   * 执行目标框筛选和事件判定
   * @param image 待处理的图像数据
   * @param thread_id 当前线程ID
   */
  void perform_event_determination(ImageDataPtr image, int thread_id);

  /**
   * 计算目标框的宽度
   * @param box 目标框
   * @return 宽度（像素）
   */
  int calculate_box_width(const ImageData::BoundingBox& box) const;

  /**
   * 检查目标框是否在指定区域内
   * @param box 目标框
   * @param region_top 区域上边界
   * @param region_bottom 区域下边界
   * @return 是否在区域内
   */
  bool is_box_in_region(const ImageData::BoundingBox& box, 
                        int region_top, int region_bottom) const;

  /**
   * 在指定区域内寻找宽度最小的目标框
   * @param boxes 目标框列表
   * @param region_top 区域上边界
   * @param region_bottom 区域下边界
   * @return 宽度最小的目标框指针，如果没有则返回nullptr
   */
  ImageData::BoundingBox* find_min_width_box_in_region(
      const std::vector<ImageData::BoundingBox>& boxes,
      int region_top, int region_bottom) const;

  /**
   * 绘制应急车道的四分之一点和区域边界
   * @param image 要绘制的图像
   * @param emergency_lane 应急车道结果
   */
  void drawEmergencyLaneQuarterPoints(cv::Mat& image, 
                                     const EmergencyLaneResult& emergency_lane);

  /**
   * 判断目标的状态（是否占用应急车道）
   * @param box 目标框
   * @param emergency_lane 应急车道结果
   * @return 目标状态
   */
  ObjectStatus determineObjectStatus(const ImageData::BoundingBox& box,
                                   const EmergencyLaneResult& emergency_lane);
};
