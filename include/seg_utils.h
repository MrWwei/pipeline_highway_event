#ifndef SEG_UTILS_H
#define SEG_UTILS_H
#include <algorithm>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>

struct PointT {
  int x, y;
  PointT(int x = 0, int y = 0) : x(x), y(y) {}
  bool operator==(const PointT &other) const {
    return x == other.x && y == other.y;
  }
};

struct EmergencyLaneResult {
  std::vector<PointT> left_quarter_points;
  std::vector<PointT> right_quarter_points;
  std::vector<PointT> left_lane_region;
  std::vector<PointT> right_lane_region;
  std::vector<PointT> middle_lane_region;
  bool is_valid;

  EmergencyLaneResult() : is_valid(false) {}
};

struct DetectRegion {
  int y1, y2, x1, x2;
  bool is_valid;

  DetectRegion() : y1(0), y2(0), x1(0), x2(0), is_valid(false) {}
  DetectRegion(int y1, int y2, int x1, int x2)
      : y1(y1), y2(y2), x1(x1), x2(x2), is_valid(true) {}
};

struct LowLevelBoxResult {
  double lowest_y;
  double lowest_box_width;
  bool is_valid;

  LowLevelBoxResult() : lowest_y(0.0), lowest_box_width(0.0), is_valid(false) {}
  LowLevelBoxResult(double y, double width)
      : lowest_y(y), lowest_box_width(width), is_valid(true) {}
};

struct DetectionBox {
  double x1, y1, x2, y2;
  double conf;
  int cls;
  int track_id;

  DetectionBox(double x1 = 0, double y1 = 0, double x2 = 0, double y2 = 0,
               double conf = 0.0, int cls = -1, int track_id = -1)
      : x1(x1), y1(y1), x2(x2), y2(y2), conf(conf), cls(cls),
        track_id(track_id) {}
};

// 函数声明
/**
 * @brief 根据分割图得到双向的应急车道线，并返回中间区域
 * @param mask 分割掩码数据
 * @param height 图像高度
 * @param width 图像宽度
 * @param car_width 车辆宽度
 * @param car_low_y 车辆最低位置y坐标
 * @return 应急车道线相关区域
 */

/**
 * @brief 根据分割图得到双向的应急车道线（cv::Mat版本）
 * @param mask 分割掩码图像
 * @param car_width 车辆宽度
 * @param car_low_y 车辆最低位置y坐标
 * @return 应急车道线相关区域
 */
EmergencyLaneResult get_Emergency_Lane(const cv::Mat &mask, double car_width,
                                       double car_low_y,
                                       float times_car_width = 2.0);


/**
 * @brief 优化后的crop_detect_region函数
 * @param img 二值化后的掩码图像
 * @param height 图像高度
 * @param width 图像宽度
 * @return DetectRegion 包含y1, y2, x1, x2的检测区域
 */
DetectRegion crop_detect_region_optimized(const cv::Mat &img, int height,
                                          int width);

/**
 * @brief 移除小的白色区域，仅保留最大的白色区域
 * @param mask 输入掩码图像
 * @return 处理后的掩码图像
 */
cv::Mat remove_small_white_regions(const cv::Mat &mask);

/**
 * @brief 获取画面下方区域中目标框宽度最小的目标框
 * @param det_result 检测结果列表
 * @param borders 边界区域
 * @param percentage 下方区域比例，默认3/8
 * @return 最小宽度的检测框
 */
DetectionBox get_low_level_box(const std::vector<DetectionBox> &det_result,
                               const DetectRegion &borders,
                               float percentage = 3.0 / 8.0);

/**
 * @brief 获取画面下方区域中目标框宽度最小的目标框（重载版本）
 * @param det_result 原始检测数据
 * @param borders 边界区域
 * @return 最小宽度的检测框
 */
DetectionBox
get_low_level_box(const std::vector<std::vector<double>> &det_result,
                  const DetectRegion &borders);

/**
 * @brief 绘制应急车道四分之一点
 * @param image 输入图像
 * @param emergency_lane 应急车道结果
 * @param left_color 左侧四分之一点颜色，默认为绿色
 * @param right_color 右侧四分之一点颜色，默认为蓝色
 * @param point_size 点的大小，默认为5
 */
void drawEmergencyLaneQuarterPoints(
    cv::Mat &image, const EmergencyLaneResult &emergency_lane,
    const cv::Scalar &left_color = cv::Scalar(0, 0, 255),
    const cv::Scalar &right_color = cv::Scalar(0, 0, 255), int point_size = 5);

#endif // SEG_UTILS_H