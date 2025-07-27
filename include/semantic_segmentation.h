#pragma once

#include "image_processor.h"
#include "road_seg.h"
#include "event_utils.h"
#include <future>
#include <string>
#include <mutex>

// 前向声明
struct PipelineConfig;

/**
 * 语义分割处理器
 * 继承自ImageProcessor基类，专门负责对输入图像进行语义分割处理
 * 支持多线程并发处理
 */
class SemanticSegmentation : public ImageProcessor {
public:
  // 构造函数，可指定线程数量，默认为1
  SemanticSegmentation(int num_threads = 1, const PipelineConfig* config = nullptr);

  // 析构函数（自定义实现，负责线程清理）
  virtual ~SemanticSegmentation();
  
  // 设置分割结果保存间隔（帧数）
  void set_seg_show_interval(int interval);
  
  // 重写参数更新方法
  virtual void change_params(const PipelineConfig &config) override;

protected:
  // 重写基类的纯虚函数：执行语义分割算法（模拟）
  virtual void process_image(ImageDataPtr image, int thread_id) override;

  // 重写处理开始前的准备工作
  virtual void on_processing_start(ImageDataPtr image, int thread_id) override;

  // 重写处理完成后的清理工作
  virtual void on_processing_complete(ImageDataPtr image,
                                      int thread_id) override;
  
public:
  // 启动处理线程
  void start() override {
    ImageProcessor::start(); // 调用基类的start
    if (!worker_thread_.joinable()) {
      stop_worker_ = false;
      worker_thread_ =
          std::thread(&SemanticSegmentation::segmentation_worker, this);
    }
  }


  // 停止处理线程
  void stop() override {
    std::cout << "  停止语义分割工作线程..." << std::endl;
    stop_worker_ = true;
    
    // 向队列添加多个空数据来确保唤醒所有可能在等待的操作
    for (int i = 0; i < 5; ++i) {
      segmentation_queue_->push(nullptr);
    }
    
    std::cout << "  等待语义分割工作线程退出..." << std::endl;
    if (worker_thread_.joinable()) {
      worker_thread_.join();
      std::cout << "  ✅ 语义分割工作线程已正常退出" << std::endl;
    }
    
    // 清空队列中剩余的图像，避免阻塞
    std::cout << "  清理语义分割队列..." << std::endl;
    ImageDataPtr remaining_img;
    while (segmentation_queue_->try_pop(remaining_img)) {
      if (remaining_img) {
        try {
          if (remaining_img->segmentation_promise && 
              remaining_img->segmentation_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            remaining_img->segmentation_promise->set_value();
          }
        } catch (const std::future_error&) {
          // Promise已经被设置，忽略
        }
      }
    }
    
    std::cout << "  调用基类停止..." << std::endl;
    ImageProcessor::stop(); // 调用基类的stop
    std::cout << "  语义分割模块停止完成" << std::endl;
  }

private:
  void segmentation_worker();

  std::unique_ptr<ThreadSafeQueue<ImageDataPtr>> segmentation_queue_;
  std::thread worker_thread_;
  std::atomic<bool> stop_worker_;
  IRoadSeg *road_seg_instance_; // 支持多线程的SDK实例列表

  bool enable_seg_show_; // 是否启用分割结果可视化
  std::string seg_show_image_path_; // 分割结果图像保存路径
  int seg_show_interval_ = 200; // 分割结果保存间隔（帧数）
  mutable int seg_frame_counter_ = 0; // 帧计数器
  
  // 互斥锁保护多线程访问
  mutable std::mutex seg_show_mutex_; // 保护分割结果显示相关变量的互斥锁
};
