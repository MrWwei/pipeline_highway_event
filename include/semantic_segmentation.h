#pragma once

#include "image_processor.h"

/**
 * 语义分割处理器
 * 继承自ImageProcessor基类，专门负责对输入图像进行语义分割处理
 * 支持多线程并发处理
 */
class SemanticSegmentation : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  SemanticSegmentation(int num_threads = 1);

  // 虚析构函数
  virtual ~SemanticSegmentation() = default;

protected:
  // 重写基类的纯虚函数：执行语义分割算法（模拟）
  virtual void process_image(ImageDataPtr image, int thread_id) override;

  // 重写处理开始前的准备工作
  virtual void on_processing_start(ImageDataPtr image, int thread_id) override;

  // 重写处理完成后的清理工作
  virtual void on_processing_complete(ImageDataPtr image,
                                      int thread_id) override;

private:
  // 具体的语义分割算法实现
  void perform_semantic_segmentation(ImageDataPtr image, int thread_id);
};
