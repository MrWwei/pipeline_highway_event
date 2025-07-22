#include "seg_utils.h"

EmergencyLaneResult get_Emergency_Lane(const std::vector<uint8_t> &mask,
                                       int height, int width, double car_width,
                                       double car_low_y,
                                       float times_car_width) {
  /**
   * 根据分割图得到双向的应急车道线，并返回中间区域
   * @param mask: 分割掩码数据
   * @param height: 图像高度
   * @param width: 图像宽度
   * @param car_width: 车辆宽度
   * @param car_low_y: 车辆最低位置y坐标
   * @return: 应急车道线相关区域
   */

  EmergencyLaneResult result;

  // 检查车辆宽度是否有效
  if (car_width <= 0) {
    std::cout << "Invalid car width, cannot calculate emergency lane."
              << std::endl;
    return result;
  }

  // 将 std::vector<uint8_t> 转换为 cv::Mat
  cv::Mat mask_mat(height, width, CV_8UC1, const_cast<uint8_t *>(mask.data()));

  double level_width = 0;

  // 获取白色区域在car_low_y位置的宽度
  int car_low_y_int = static_cast<int>(car_low_y);
  if (car_low_y_int >= height) {
    car_low_y_int = height - 1;
  }

  if (car_low_y_int >= 0 && car_low_y_int < height) {
    // 获取指定行的数据
    std::vector<int> white_indices;
    for (int col = 0; col < width; col++) {
      if (mask_mat.at<uint8_t>(car_low_y_int, col) == 255) {
        white_indices.push_back(col);
      }
    }

    if (!white_indices.empty()) {
      level_width = white_indices.back() - white_indices.front();
    } else {
      level_width = 0;
    }
  }

  // 如果level_width为0，则无法计算p_interval
  if (level_width == 0) {
    std::cout << "No white pixels found at the specified row, cannot "
                 "calculate p_interval."
              << std::endl;
    return result;
  }

  // 计算间隔比例
  double p_interval = (car_width * times_car_width) / level_width;

  // 检查最后一行是否有白色像素
  std::vector<int> white_indices;
  for (int col = 0; col < width; col++) {
    if (mask_mat.at<uint8_t>(height - 1, col) == 255) {
      white_indices.push_back(col);
    }
  }

  if (white_indices.empty()) {
    return result;
  }

  int start_col = white_indices.front();
  int end_col = white_indices.back();

  std::vector<Point> left_border_points;
  std::vector<Point> right_border_points;
  std::vector<Point> left_quarter_points;
  std::vector<Point> right_quarter_points;

  // 遍历每一行
  for (int y = 0; y < height; y++) {
    white_indices.clear();

    // 找到当前行的白色像素
    for (int col = 0; col < width; col++) {
      if (mask_mat.at<uint8_t>(y, col) == 255) {
        white_indices.push_back(col);
      }
    }

    if (!white_indices.empty()) {
      start_col = white_indices.front();
      end_col = white_indices.back();

      // 左右边界点
      left_border_points.push_back(Point(start_col, y));
      right_border_points.push_back(Point(end_col, y));

      // 计算四分之一点
      int left_quarter_col =
          start_col + static_cast<int>((end_col - start_col) * p_interval);
      left_quarter_points.push_back(Point(left_quarter_col, y));

      int right_quarter_col =
          end_col - static_cast<int>((end_col - start_col) * p_interval);
      right_quarter_points.push_back(Point(right_quarter_col, y));
    }
  }

  // 构建结果
  result.left_quarter_points = left_quarter_points;
  result.right_quarter_points = right_quarter_points;

  // 构建左车道区域 (left_border_points + left_quarter_points[::-1])
  result.left_lane_region = left_border_points;
  result.left_lane_region.insert(result.left_lane_region.end(),
                                 left_quarter_points.rbegin(),
                                 left_quarter_points.rend());

  // 构建右车道区域 (right_border_points + right_quarter_points[::-1])
  result.right_lane_region = right_border_points;
  result.right_lane_region.insert(result.right_lane_region.end(),
                                  right_quarter_points.rbegin(),
                                  right_quarter_points.rend());

  // 构建中间车道区域 (left_quarter_points + right_quarter_points[::-1])
  result.middle_lane_region = left_quarter_points;
  result.middle_lane_region.insert(result.middle_lane_region.end(),
                                   right_quarter_points.rbegin(),
                                   right_quarter_points.rend());

  result.is_valid = true;
  return result;
}

EmergencyLaneResult get_Emergency_Lane(const cv::Mat &mask, double car_width,
                                       double car_low_y,
                                       float times_car_width) {
  // 将 cv::Mat 转换为 std::vector<uint8_t>
  std::vector<uint8_t> mask_vector;
  if (mask.isContinuous()) {
    mask_vector.assign(mask.data, mask.data + mask.total());
  } else {
    mask_vector.reserve(mask.rows * mask.cols);
    for (int i = 0; i < mask.rows; i++) {
      for (int j = 0; j < mask.cols; j++) {
        mask_vector.push_back(mask.at<uint8_t>(i, j));
      }
    }
  }

  return get_Emergency_Lane(mask_vector, mask.rows, mask.cols, car_width,
                            car_low_y, times_car_width);
}

EmergencyLaneResult get_Emergency_Lane(const cv::Mat &mask, float p_interval) {
  // 根据分割图得到双向的应急车道线，并返回中间区域
  std::vector<Point> left_border_points;
  std::vector<Point> right_border_points;
  std::vector<Point> left_quarter_points;
  std::vector<Point> right_quarter_points;

  // const double p_interval = 0.25; // 四分之一点的比例

  int height = mask.rows;
  int width = mask.cols;

  // 检查最后一行是否有白色像素
  cv::Mat last_row = mask.row(height - 1);
  std::vector<int> white_indices;
  for (int col = 0; col < width; col++) {
    if (last_row.at<uchar>(col) == 255) {
      white_indices.push_back(col);
    }
  }

  if (white_indices.empty()) {
    return EmergencyLaneResult(); // 返回空结果
  }

  int start_col = white_indices.front();
  int end_col = white_indices.back();

  // 遍历每一行
  for (int y = 0; y < height; y++) {
    white_indices.clear();

    // 找到当前行的白色像素
    for (int col = 0; col < width; col++) {
      if (mask.at<uchar>(y, col) == 255) {
        white_indices.push_back(col);
      }
    }

    if (!white_indices.empty()) {
      start_col = white_indices.front();
      end_col = white_indices.back();

      // 左右边界点
      left_border_points.push_back(Point(start_col, y));
      right_border_points.push_back(Point(end_col, y));

      // 计算四分之一点
      int left_quarter_col =
          start_col + static_cast<int>((end_col - start_col) * p_interval);
      left_quarter_points.push_back(Point(left_quarter_col, y));

      int right_quarter_col =
          end_col - static_cast<int>((end_col - start_col) * p_interval);
      right_quarter_points.push_back(Point(right_quarter_col, y));
    }
  }

  // 构建结果
  EmergencyLaneResult result;
  result.left_quarter_points = left_quarter_points;
  result.right_quarter_points = right_quarter_points;

  // 构建左车道区域 (left_border_points + left_quarter_points[::-1])
  result.left_lane_region = left_border_points;
  result.left_lane_region.insert(result.left_lane_region.end(),
                                 left_quarter_points.rbegin(),
                                 left_quarter_points.rend());

  // 构建右车道区域 (right_border_points + right_quarter_points[::-1])
  result.right_lane_region = right_border_points;
  result.right_lane_region.insert(result.right_lane_region.end(),
                                  right_quarter_points.rbegin(),
                                  right_quarter_points.rend());

  // 构建中间车道区域 (left_quarter_points + right_quarter_points[::-1])
  result.middle_lane_region = left_quarter_points;
  result.middle_lane_region.insert(result.middle_lane_region.end(),
                                   right_quarter_points.rbegin(),
                                   right_quarter_points.rend());

  result.is_valid = true;
  return result;
}

DetectRegion crop_detect_region_optimized(const cv::Mat &img, int height,
                                          int width) {
  double start_row_p = 0.0;

  // 使用OpenCV的findNonZero函数找到所有非零像素
  std::vector<cv::Point> white_pixels;
  cv::findNonZero(img, white_pixels);

  if (white_pixels.empty()) {
    return DetectRegion();
  }

  // 使用OpenCV的boundingRect找到边界框
  cv::Rect bounding_rect = cv::boundingRect(white_pixels);

  int y_min = bounding_rect.y;
  int y_max = bounding_rect.y + bounding_rect.height - 1;
  int white_height = y_max - y_min;
  int y_start = y_min + static_cast<int>(white_height * start_row_p);
  int y_end = y_max;

  // 过滤在指定y范围内的像素
  std::vector<cv::Point> retained_pixels;
  for (const auto &pixel : white_pixels) {
    if (pixel.y >= y_start && pixel.y <= y_end) {
      retained_pixels.push_back(pixel);
    }
  }

  if (retained_pixels.empty()) {
    return DetectRegion();
  }

  // 计算保留像素的边界框
  cv::Rect retained_rect = cv::boundingRect(retained_pixels);

  // 添加1像素的边界并确保在图像范围内
  int x1 = std::max(0, retained_rect.x - 1);
  int y1 = std::max(0, retained_rect.y - 1);
  int x2 = std::min(width, retained_rect.x + retained_rect.width + 1);
  int y2 = std::min(height, retained_rect.y + retained_rect.height + 1);

  return DetectRegion(y1, y2, x1, x2);
}

cv::Mat remove_small_white_regions(const cv::Mat &mask) {
  /**
   * 仅保留最大的白色区域，其余全部去除
   */

  // 创建椭圆形态学核
  // int kernel_size = 5;
  // cv::Mat kernel = cv::getStructuringElement(
  //     cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));

  // // 形态学闭操作
  // cv::Mat closed;
  // cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, kernel);

  // 填充孔洞的函数
  auto fill_holes = [](const cv::Mat &img) -> cv::Mat {
    cv::Mat floodfill = img.clone();
    int h = img.rows;
    int w = img.cols;

    cv::Mat mask_ff = cv::Mat::zeros(h + 2, w + 2, CV_8UC1);

    // 从四个角开始flood fill
    cv::floodFill(floodfill, mask_ff, cv::Point(0, 0), cv::Scalar(255));
    cv::floodFill(floodfill, mask_ff, cv::Point(w - 1, 0), cv::Scalar(255));
    cv::floodFill(floodfill, mask_ff, cv::Point(0, h - 1), cv::Scalar(255));
    cv::floodFill(floodfill, mask_ff, cv::Point(w - 1, h - 1), cv::Scalar(255));

    cv::Mat result;
    cv::bitwise_not(floodfill, result);
    return result;
  };

  // 填充孔洞
  cv::Mat holes_mask = fill_holes(mask);
  cv::Mat filled;
  cv::bitwise_or(mask, holes_mask, filled);

  // 二值化
  cv::Mat binary_img;
  cv::threshold(filled, binary_img, 200, 255, cv::THRESH_BINARY);

  // 查找轮廓
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // 创建最终mask
  cv::Mat final_mask = cv::Mat::zeros(filled.size(), filled.type());

  // 只保留最大面积的轮廓
  if (!contours.empty()) {
    auto max_contour_it = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
          return cv::contourArea(a) < cv::contourArea(b);
        });

    std::vector<std::vector<cv::Point>> max_contour = {*max_contour_it};
    cv::drawContours(final_mask, max_contour, -1, cv::Scalar(255), -1);
  }

  return final_mask;
}

DetectionBox get_low_level_box(const std::vector<DetectionBox> &det_result,
                               const DetectRegion &borders, float percentage) {
  /**
   * 获取画面下方指定区域中目标框宽度最小的目标框
   * @param det_result: 检测结果列表
   * @param borders: 边界区域 (y1, y2, x1, x2)
   * @param percentage: 下方区域比例
   * @return: 最小宽度的检测框
   */

  // 检查输入是否为空
  if (det_result.empty()) {
    return DetectionBox(0, 0, 0, 0, 0.0, -1); // 返回无效框
  }

  // 计算阈值位置：画面下方指定比例区域
  double lowest_y_threshold =
      borders.y2 - (borders.y2 - borders.y1) * percentage;

  double lowest_box_width = std::numeric_limits<double>::infinity();
  double lowest_y = 0.0;
  bool found_valid_box = false;

  // 遍历所有检测框
  DetectionBox box_min = DetectionBox(0, 0, 0, 0, 0.0, -1);
  for (const auto &box : det_result) {
    // 如果框的底部位置在阈值之上，跳过
    if (box.y2 < lowest_y_threshold) {
      continue;
    }

    // 计算框的宽度
    double box_width = box.x2 - box.x1;

    // 如果找到更小宽度的框，更新记录
    if (box_width < lowest_box_width) {
      lowest_box_width = box_width;
      lowest_y = (box.y1 + box.y2) / 2.0; // 框的中心y坐标
      found_valid_box = true;
      box_min = box; // 记录当前最小宽度的框
    }
  }

  return box_min;
}

DetectionBox
get_low_level_box(const std::vector<std::vector<double>> &det_result,
                  const DetectRegion &borders) {
  // 转换为 DetectionBox 格式
  std::vector<DetectionBox> boxes;
  for (const auto &box_data : det_result) {
    if (box_data.size() >= 6) {
      boxes.emplace_back(box_data[0], box_data[1], box_data[2], box_data[3],
                         box_data[4], static_cast<int>(box_data[5]));
    }
  }

  return get_low_level_box(boxes, borders, 0.5);
}

/**
 * @brief 绘制应急车道四分之一点
 * @param image 输入图像
 * @param emergency_lane 应急车道结果
 * @param left_color 左侧四分之一点颜色，默认为绿色
 * @param right_color 右侧四分之一点颜色，默认为蓝色
 * @param point_size 点的大小，默认为5
 */
void drawEmergencyLaneQuarterPoints(cv::Mat &image,
                                    const EmergencyLaneResult &emergency_lane,
                                    const cv::Scalar &left_color,
                                    const cv::Scalar &right_color,
                                    int point_size) {
  if (!emergency_lane.is_valid) {
    return;
  }

  // 绘制左侧四分之一点
  if (!emergency_lane.left_quarter_points.empty()) {
    for (const auto &point : emergency_lane.left_quarter_points) {
      // 检查点是否在图像范围内
      if (point.x >= 0 && point.x < image.cols && point.y >= 0 &&
          point.y < image.rows) {
        cv::circle(image, cv::Point(point.x, point.y), point_size, left_color,
                   -1);
      }
    }
  }

  // 绘制右侧四分之一点
  if (!emergency_lane.right_quarter_points.empty()) {
    for (const auto &point : emergency_lane.right_quarter_points) {
      // 检查点是否在图像范围内
      if (point.x >= 0 && point.x < image.cols && point.y >= 0 &&
          point.y < image.rows) {
        cv::circle(image, cv::Point(point.x, point.y), point_size, right_color,
                   -1);
      }
    }
  }

  // 可选：绘制连接线来显示车道线
  if (!emergency_lane.left_quarter_points.empty() &&
      emergency_lane.left_quarter_points.size() > 1) {
    for (size_t i = 1; i < emergency_lane.left_quarter_points.size(); ++i) {
      const auto &p1 = emergency_lane.left_quarter_points[i - 1];
      const auto &p2 = emergency_lane.left_quarter_points[i];

      // 检查点是否在图像范围内
      if (p1.x >= 0 && p1.x < image.cols && p1.y >= 0 && p1.y < image.rows &&
          p2.x >= 0 && p2.x < image.cols && p2.y >= 0 && p2.y < image.rows) {
        cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y),
                 left_color, 2);
      }
    }
  }

  if (!emergency_lane.right_quarter_points.empty() &&
      emergency_lane.right_quarter_points.size() > 1) {
    for (size_t i = 1; i < emergency_lane.right_quarter_points.size(); ++i) {
      const auto &p1 = emergency_lane.right_quarter_points[i - 1];
      const auto &p2 = emergency_lane.right_quarter_points[i];

      // 检查点是否在图像范围内
      if (p1.x >= 0 && p1.x < image.cols && p1.y >= 0 && p1.y < image.rows &&
          p2.x >= 0 && p2.x < image.cols && p2.y >= 0 && p2.y < image.rows) {
        cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y),
                 right_color, 2);
      }
    }
  }
}
