#pragma once

#include "image_processor.h"

/**
 * 目标检测处理器
 * 继承自ImageProcessor基类，专门负责对输入图像进行目标检测处理
 * 支持多线程并发处理
 */
class ObjectDetection : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  ObjectDetection(int num_threads = 1);

  // 虚析构函数
  virtual ~ObjectDetection() = default;

protected:
  // 重写基类的纯虚函数：执行目标检测算法（模拟）
  virtual void process_image(ImageDataPtr image, int thread_id) override;

  // 重写处理开始前的准备工作
  virtual void on_processing_start(ImageDataPtr image, int thread_id) override;

  // 重写处理完成后的清理工作
  virtual void on_processing_complete(ImageDataPtr image,
                                      int thread_id) override;

private:
  // 具体的目标检测算法实现
  void perform_object_detection(ImageDataPtr image, int thread_id);
};
