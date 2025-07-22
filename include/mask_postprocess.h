#pragma once

#include "image_processor.h"
#include <opencv2/opencv.hpp>

/**
 * Mask后处理模块
 * 负责对语义分割结果进行后处理，如去除小的白色区域等
 */
class MaskPostProcess : public ImageProcessor {
public:
  explicit MaskPostProcess(int num_threads);
  ~MaskPostProcess() = default;

protected:
  void process_image(ImageDataPtr image, int thread_id) override;
  void on_processing_start(ImageDataPtr image, int thread_id) override;
  void on_processing_complete(ImageDataPtr image, int thread_id) override;

private:
  void perform_mask_postprocess(ImageDataPtr image, int thread_id);
};
