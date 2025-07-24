#pragma once

#include "image_processor.h"
#include "thread_safe_queue.h"
#include "seg_utils.h"

// 前向声明
struct PipelineConfig;

/**
 * 目标框筛选处理器
 * 负责从检测结果中筛选出特定区域内宽度最小的目标框
 * 区域定义：图片从上开始的七分之二处到七分之六处
 * 如果该区域没有目标框，则在全图范围内寻找
 */
class BoxFilter : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  BoxFilter(int num_threads = 1, const PipelineConfig* config = nullptr);

  // 虚析构函数
  virtual ~BoxFilter();

protected:
  // 重写基类的纯虚函数：执行目标框筛选
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
  
  // 具体的目标框筛选算法实现
  void perform_box_filtering(ImageDataPtr image, int thread_id);
  
  // 计算目标框的宽度
  int calculate_box_width(const ImageData::BoundingBox& box) const;
  
  // 检查目标框是否在指定区域内
  bool is_box_in_region(const ImageData::BoundingBox& box, 
                        int region_top, int region_bottom) const;
  
  // 从指定区域筛选宽度最小的目标框
  ImageData::BoundingBox* find_min_width_box_in_region(
      const std::vector<ImageData::BoundingBox>& boxes,
      int region_top, int region_bottom) const;
  void
  drawEmergencyLaneQuarterPoints(cv::Mat &image,
                                 const EmergencyLaneResult &emergency_lane);
  ObjectStatus
  determineObjectStatus(const ImageData::BoundingBox &box,
                        const EmergencyLaneResult &emergency_lane);
};
