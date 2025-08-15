#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <chrono>
#include <memory>
#include <mutex>
#include <fstream>
#include <vector>
#include <functional>

/**
 * 内存使用统计信息
 */
struct MemoryStats {
    size_t process_memory_mb;      // 进程内存使用量 (MB)
    size_t virtual_memory_mb;      // 虚拟内存使用量 (MB)
    size_t resident_memory_mb;     // 驻留内存使用量 (MB)
    size_t shared_memory_mb;       // 共享内存使用量 (MB)
    size_t gpu_memory_used_mb;     // GPU内存使用量 (MB)
    size_t gpu_memory_total_mb;    // GPU总内存量 (MB)
    double cpu_usage_percent;      // CPU使用率 (%)
    double memory_usage_percent;   // 内存使用率 (%)
    std::chrono::system_clock::time_point timestamp;
    
    MemoryStats() : process_memory_mb(0), virtual_memory_mb(0), 
                   resident_memory_mb(0), shared_memory_mb(0),
                   gpu_memory_used_mb(0), gpu_memory_total_mb(0),
                   cpu_usage_percent(0.0), memory_usage_percent(0.0),
                   timestamp(std::chrono::system_clock::now()) {}
};

/**
 * 内存监控器
 * 实时监控系统和进程内存使用情况，检测内存泄漏
 */
class MemoryMonitor {
public:
    explicit MemoryMonitor(const std::string& log_file = "memory_monitor.log", 
                          int monitor_interval_ms = 1000);
    virtual ~MemoryMonitor();
    
    // 启动内存监控
    void start();
    
    // 停止内存监控
    void stop();
    
    // 获取当前内存统计信息
    MemoryStats get_current_stats();
    
    // 获取内存增长率 (MB/second)
    double get_memory_growth_rate();
    
    // 检查是否存在内存泄漏
    bool is_memory_leak_detected();
    
    // 设置内存泄漏阈值 (MB/minute)
    void set_leak_detection_threshold(double threshold_mb_per_min);
    
    // 添加自定义内存监控点
    void add_memory_checkpoint(const std::string& name);
    
    // 设置内存告警回调函数
    void set_memory_warning_callback(std::function<void(const MemoryStats&)> callback);
    
    // 打印内存状态报告
    void print_memory_report();
    
    // 导出内存使用历史到CSV文件
    void export_to_csv(const std::string& csv_file);
    
    // 重置统计信息
    void reset_statistics();

private:
    // 内存监控线程函数
    void monitor_thread_func();
    
    // 获取进程内存信息
    MemoryStats collect_memory_stats();
    
    // 解析/proc/stat文件获取CPU使用率
    double get_cpu_usage();
    
    // 解析/proc/meminfo文件获取系统内存信息
    void get_system_memory_info(MemoryStats& stats);
    
    // 解析/proc/self/status文件获取进程内存信息
    void get_process_memory_info(MemoryStats& stats);
    
    // 获取GPU内存信息（如果有NVIDIA GPU）
    void get_gpu_memory_info(MemoryStats& stats);
    
    // 写入日志
    void write_log(const MemoryStats& stats);
    
    // 检测内存泄漏
    void check_memory_leak(const MemoryStats& stats);
    
private:
    // 配置参数
    std::string log_file_;
    int monitor_interval_ms_;
    double leak_detection_threshold_mb_per_min_;
    
    // 运行状态
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    std::thread monitor_thread_;
    
    // 统计数据
    mutable std::mutex stats_mutex_;
    std::vector<MemoryStats> memory_history_;
    size_t max_history_size_;
    MemoryStats last_stats_;
    
    // 内存泄漏检测
    std::chrono::steady_clock::time_point leak_detection_start_time_;
    size_t leak_detection_start_memory_;
    bool leak_detected_;
    
    // 日志文件
    std::ofstream log_stream_;
    
    // CPU使用率计算
    long long last_total_cpu_time_;
    long long last_idle_cpu_time_;
    
    // 告警回调
    std::function<void(const MemoryStats&)> warning_callback_;
    
    // 内存检查点
    std::vector<std::pair<std::string, MemoryStats>> checkpoints_;
};

/**
 * 内存监控工具类
 * 提供静态方法进行快速内存检查
 */
class MemoryUtils {
public:
    // 获取当前进程内存使用量 (MB)
    static size_t get_process_memory_mb();
    
    // 获取系统可用内存 (MB)
    static size_t get_available_memory_mb();
    
    // 获取GPU内存使用情况
    static std::pair<size_t, size_t> get_gpu_memory_usage_mb();
    
    // 打印内存使用摘要
    static void print_memory_summary();
    
    // 检查内存是否充足
    static bool is_memory_sufficient(size_t required_mb);
    
    // 格式化内存大小显示
    static std::string format_memory_size(size_t bytes);
    
    // 解析内存数值 (移到public以供MemoryMonitor使用)
    static size_t parse_memory_value(const std::string& line);
    
private:
    // 读取文件内容
    static std::string read_file_content(const std::string& filepath);
};

/**
 * RAII内存监控器
 * 自动在作用域开始和结束时记录内存使用情况
 */
class ScopedMemoryMonitor {
public:
    explicit ScopedMemoryMonitor(const std::string& scope_name, 
                                MemoryMonitor* monitor = nullptr);
    ~ScopedMemoryMonitor();
    
    // 获取内存增长量
    size_t get_memory_delta_mb() const;

private:
    std::string scope_name_;
    MemoryMonitor* monitor_;
    MemoryStats start_stats_;
    std::chrono::steady_clock::time_point start_time_;
};

// 便利宏定义
#define MEMORY_CHECKPOINT(monitor, name) \
    if (monitor) monitor->add_memory_checkpoint(name)

#define SCOPED_MEMORY_MONITOR(name) \
    ScopedMemoryMonitor _scoped_monitor(name)

#define SCOPED_MEMORY_MONITOR_WITH_MONITOR(name, monitor) \
    ScopedMemoryMonitor _scoped_monitor(name, monitor)
