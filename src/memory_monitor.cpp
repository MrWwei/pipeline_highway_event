#include "memory_monitor.h"
#include "logger_manager.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <sys/resource.h>

MemoryMonitor::MemoryMonitor(const std::string& log_file, int monitor_interval_ms)
    : log_file_(log_file), monitor_interval_ms_(monitor_interval_ms),
      leak_detection_threshold_mb_per_min_(50.0), running_(false), stop_requested_(false),
      max_history_size_(3600), leak_detected_(false),
      last_total_cpu_time_(0), last_idle_cpu_time_(0) {
    
    // 打开日志文件
    log_stream_.open(log_file_, std::ios::out | std::ios::app);
    if (log_stream_.is_open()) {
        log_stream_ << "=== 内存监控开始 === " 
                   << std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch()).count() 
                   << std::endl;
        log_stream_ << "时间戳,进程内存(MB),虚拟内存(MB),驻留内存(MB),共享内存(MB),"
                   << "GPU已用(MB),GPU总量(MB),CPU使用率(%),内存使用率(%)" << std::endl;
    }
}

MemoryMonitor::~MemoryMonitor() {
    stop();
    if (log_stream_.is_open()) {
        log_stream_ << "=== 内存监控结束 ===" << std::endl;
        log_stream_.close();
    }
}

void MemoryMonitor::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    stop_requested_.store(false);
    
    // 初始化泄漏检测
    leak_detection_start_time_ = std::chrono::steady_clock::now();
    auto initial_stats = collect_memory_stats();
    leak_detection_start_memory_ = initial_stats.process_memory_mb;
    
    // 启动监控线程
    monitor_thread_ = std::thread(&MemoryMonitor::monitor_thread_func, this);
    
    std::cout << "✅ 内存监控已启动，监控间隔: " << monitor_interval_ms_ << "ms" << std::endl;
}

void MemoryMonitor::stop() {
    if (!running_.load()) {
        return;
    }
    
    stop_requested_.store(true);
    running_.store(false);
    
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    LOG_INFO("🛑 内存监控已停止");
}

MemoryStats MemoryMonitor::get_current_stats() {
    return collect_memory_stats();
}

double MemoryMonitor::get_memory_growth_rate() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (memory_history_.size() < 2) {
        return 0.0;
    }
    
    // 计算最近一分钟的内存增长率
    auto now = std::chrono::steady_clock::now();
    auto one_minute_ago = now - std::chrono::minutes(1);
    
    auto recent_start = std::find_if(memory_history_.rbegin(), memory_history_.rend(),
        [one_minute_ago](const MemoryStats& stats) {
            auto stats_time = std::chrono::steady_clock::time_point(
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    stats.timestamp.time_since_epoch()));
            return stats_time <= one_minute_ago;
        });
    
    if (recent_start == memory_history_.rend()) {
        return 0.0;
    }
    
    size_t start_memory = recent_start->process_memory_mb;
    size_t current_memory = memory_history_.back().process_memory_mb;
    
    auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        now - one_minute_ago).count();
    
    if (duration_seconds == 0) {
        return 0.0;
    }
    
    return static_cast<double>(current_memory - start_memory) / duration_seconds;
}

bool MemoryMonitor::is_memory_leak_detected() {
    return leak_detected_;
}

void MemoryMonitor::set_leak_detection_threshold(double threshold_mb_per_min) {
    leak_detection_threshold_mb_per_min_ = threshold_mb_per_min;
}

void MemoryMonitor::add_memory_checkpoint(const std::string& name) {
    auto stats = collect_memory_stats();
    std::lock_guard<std::mutex> lock(stats_mutex_);
    checkpoints_.emplace_back(name, stats);
    
    std::cout << "📍 内存检查点 [" << name << "]: " << stats.process_memory_mb << " MB" << std::endl;
}

void MemoryMonitor::set_memory_warning_callback(std::function<void(const MemoryStats&)> callback) {
    warning_callback_ = callback;
}

void MemoryMonitor::print_memory_report() {
    auto stats = collect_memory_stats();
    
    LOG_INFO("\n📊 内存使用报告:");
    std::cout << "├─ 进程内存: " << stats.process_memory_mb << " MB" << std::endl;
    std::cout << "├─ 虚拟内存: " << stats.virtual_memory_mb << " MB" << std::endl;
    std::cout << "├─ 驻留内存: " << stats.resident_memory_mb << " MB" << std::endl;
    std::cout << "├─ 共享内存: " << stats.shared_memory_mb << " MB" << std::endl;
    
    if (stats.gpu_memory_total_mb > 0) {
        std::cout << "├─ GPU内存: " << stats.gpu_memory_used_mb << "/" 
                  << stats.gpu_memory_total_mb << " MB ("
                  << std::fixed << std::setprecision(1) 
                  << 100.0 * stats.gpu_memory_used_mb / stats.gpu_memory_total_mb << "%)" << std::endl;
    }
    
    std::cout << "├─ CPU使用率: " << std::fixed << std::setprecision(1) 
              << stats.cpu_usage_percent << "%" << std::endl;
    std::cout << "├─ 内存使用率: " << std::fixed << std::setprecision(1) 
              << stats.memory_usage_percent << "%" << std::endl;
    
    double growth_rate = get_memory_growth_rate();
    std::cout << "├─ 内存增长率: " << std::fixed << std::setprecision(2) 
              << growth_rate << " MB/s" << std::endl;
    
    if (leak_detected_) {
        LOG_INFO("⚠️  检测到内存泄漏!");
    }
    
    // 显示检查点信息
    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (!checkpoints_.empty()) {
        LOG_INFO("\n📍 内存检查点:");
        for (const auto& checkpoint : checkpoints_) {
            std::cout << "   " << checkpoint.first << ": " 
                      << checkpoint.second.process_memory_mb << " MB" << std::endl;
        }
    }
    std::cout << std::endl;
}

void MemoryMonitor::export_to_csv(const std::string& csv_file) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::ofstream csv_stream(csv_file);
    if (!csv_stream.is_open()) {
        std::cerr << "❌ 无法打开CSV文件: " << csv_file << std::endl;
        return;
    }
    
    // 写入CSV头部
    csv_stream << "Timestamp,ProcessMemory(MB),VirtualMemory(MB),ResidentMemory(MB),"
               << "SharedMemory(MB),GPUUsed(MB),GPUTotal(MB),CPUUsage(%),MemoryUsage(%)" << std::endl;
    
    // 写入历史数据
    for (const auto& stats : memory_history_) {
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            stats.timestamp.time_since_epoch()).count();
        
        csv_stream << timestamp << "," << stats.process_memory_mb << ","
                   << stats.virtual_memory_mb << "," << stats.resident_memory_mb << ","
                   << stats.shared_memory_mb << "," << stats.gpu_memory_used_mb << ","
                   << stats.gpu_memory_total_mb << "," << stats.cpu_usage_percent << ","
                   << stats.memory_usage_percent << std::endl;
    }
    
    csv_stream.close();
    std::cout << "✅ 内存历史数据已导出到: " << csv_file << std::endl;
}

void MemoryMonitor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    memory_history_.clear();
    checkpoints_.clear();
    leak_detected_ = false;
    leak_detection_start_time_ = std::chrono::steady_clock::now();
    leak_detection_start_memory_ = collect_memory_stats().process_memory_mb;
    
    LOG_INFO("🔄 内存监控统计信息已重置");
}

void MemoryMonitor::monitor_thread_func() {
    while (running_.load() && !stop_requested_.load()) {
        try {
            auto stats = collect_memory_stats();
            
            // 保存到历史记录
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                memory_history_.push_back(stats);
                
                // 限制历史记录大小
                if (memory_history_.size() > max_history_size_) {
                    memory_history_.erase(memory_history_.begin(), 
                                        memory_history_.begin() + (memory_history_.size() - max_history_size_));
                }
                
                last_stats_ = stats;
            }
            
            // 写入日志
            write_log(stats);
            
            // 检测内存泄漏
            check_memory_leak(stats);
            
            // 内存告警回调
            if (warning_callback_ && (stats.memory_usage_percent > 80.0 || 
                                    stats.process_memory_mb > 2000)) {
                warning_callback_(stats);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "❌ 内存监控异常: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(monitor_interval_ms_));
    }
}

MemoryStats MemoryMonitor::collect_memory_stats() {
    MemoryStats stats;
    
    // 获取系统内存信息
    get_system_memory_info(stats);
    
    // 获取进程内存信息
    get_process_memory_info(stats);
    
    // 获取CPU使用率
    stats.cpu_usage_percent = get_cpu_usage();
    
    // 获取GPU内存信息
    get_gpu_memory_info(stats);
    
    stats.timestamp = std::chrono::system_clock::now();
    
    return stats;
}

double MemoryMonitor::get_cpu_usage() {
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        return 0.0;
    }
    
    std::string line;
    std::getline(stat_file, line);
    
    std::istringstream iss(line);
    std::string cpu;
    long long user, nice, system, idle, iowait, irq, softirq, steal;
    
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    
    long long total = user + nice + system + idle + iowait + irq + softirq + steal;
    long long total_idle = idle + iowait;
    
    if (last_total_cpu_time_ == 0) {
        last_total_cpu_time_ = total;
        last_idle_cpu_time_ = total_idle;
        return 0.0;
    }
    
    long long total_diff = total - last_total_cpu_time_;
    long long idle_diff = total_idle - last_idle_cpu_time_;
    
    double cpu_usage = 0.0;
    if (total_diff > 0) {
        cpu_usage = 100.0 * (total_diff - idle_diff) / total_diff;
    }
    
    last_total_cpu_time_ = total;
    last_idle_cpu_time_ = total_idle;
    
    return cpu_usage;
}

void MemoryMonitor::get_system_memory_info(MemoryStats& stats) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return;
    }
    
    std::string line;
    size_t mem_total = 0, mem_available = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            mem_total = MemoryUtils::parse_memory_value(line) / 1024; // KB to MB
        } else if (line.find("MemAvailable:") == 0) {
            mem_available = MemoryUtils::parse_memory_value(line) / 1024; // KB to MB
        }
    }
    
    if (mem_total > 0) {
        stats.memory_usage_percent = 100.0 * (mem_total - mem_available) / mem_total;
    }
}

void MemoryMonitor::get_process_memory_info(MemoryStats& stats) {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return;
    }
    
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("VmSize:") == 0) {
            stats.virtual_memory_mb = MemoryUtils::parse_memory_value(line) / 1024; // KB to MB
        } else if (line.find("VmRSS:") == 0) {
            stats.resident_memory_mb = MemoryUtils::parse_memory_value(line) / 1024; // KB to MB
        } else if (line.find("RssAnon:") == 0) {
            stats.process_memory_mb = MemoryUtils::parse_memory_value(line) / 1024; // KB to MB
        } else if (line.find("RssShmem:") == 0) {
            stats.shared_memory_mb = MemoryUtils::parse_memory_value(line) / 1024; // KB to MB
        }
    }
    
    // 如果没有RssAnon，使用VmRSS作为进程内存
    if (stats.process_memory_mb == 0) {
        stats.process_memory_mb = stats.resident_memory_mb;
    }
}

void MemoryMonitor::get_gpu_memory_info(MemoryStats& stats) {
    // 尝试通过nvidia-smi获取GPU内存信息
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) {
        return;
    }
    
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        size_t comma_pos = line.find(',');
        if (comma_pos != std::string::npos) {
            try {
                stats.gpu_memory_used_mb = std::stoul(line.substr(0, comma_pos));
                stats.gpu_memory_total_mb = std::stoul(line.substr(comma_pos + 1));
            } catch (const std::exception&) {
                // 解析失败，忽略GPU信息
            }
        }
    }
    
    pclose(pipe);
}

void MemoryMonitor::write_log(const MemoryStats& stats) {
    if (!log_stream_.is_open()) {
        return;
    }
    
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        stats.timestamp.time_since_epoch()).count();
    
    log_stream_ << timestamp << "," << stats.process_memory_mb << ","
               << stats.virtual_memory_mb << "," << stats.resident_memory_mb << ","
               << stats.shared_memory_mb << "," << stats.gpu_memory_used_mb << ","
               << stats.gpu_memory_total_mb << "," << std::fixed << std::setprecision(2)
               << stats.cpu_usage_percent << "," << stats.memory_usage_percent << std::endl;
    
    log_stream_.flush();
}

void MemoryMonitor::check_memory_leak(const MemoryStats& stats) {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_minutes = std::chrono::duration_cast<std::chrono::minutes>(
        current_time - leak_detection_start_time_).count();
    
    if (elapsed_minutes < 1) {
        return; // 至少运行1分钟后才开始检测
    }
    
    size_t memory_growth = stats.process_memory_mb - leak_detection_start_memory_;
    double growth_rate_per_minute = static_cast<double>(memory_growth) / elapsed_minutes;
    
    if (growth_rate_per_minute > leak_detection_threshold_mb_per_min_) {
        if (!leak_detected_) {
            leak_detected_ = true;
            LOG_INFO("⚠️  检测到疑似内存泄漏!");
            std::cout << "   内存增长率: " << std::fixed << std::setprecision(2) 
                      << growth_rate_per_minute << " MB/分钟" << std::endl;
            std::cout << "   阈值: " << leak_detection_threshold_mb_per_min_ << " MB/分钟" << std::endl;
        }
    }
}

// MemoryUtils 实现
size_t MemoryUtils::get_process_memory_mb() {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return 0;
    }
    
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            return parse_memory_value(line) / 1024; // KB to MB
        }
    }
    
    return 0;
}

size_t MemoryUtils::get_available_memory_mb() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0;
    }
    
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            return parse_memory_value(line) / 1024; // KB to MB
        }
    }
    
    return 0;
}

std::pair<size_t, size_t> MemoryUtils::get_gpu_memory_usage_mb() {
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) {
        return {0, 0};
    }
    
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        size_t comma_pos = line.find(',');
        if (comma_pos != std::string::npos) {
            try {
                size_t used = std::stoul(line.substr(0, comma_pos));
                size_t total = std::stoul(line.substr(comma_pos + 1));
                pclose(pipe);
                return {used, total};
            } catch (const std::exception&) {
                // 解析失败
            }
        }
    }
    
    pclose(pipe);
    return {0, 0};
}

void MemoryUtils::print_memory_summary() {
    size_t process_mb = get_process_memory_mb();
    size_t available_mb = get_available_memory_mb();
    auto gpu_memory = get_gpu_memory_usage_mb();
    
    LOG_INFO("\n💾 内存使用摘要:");
    std::cout << "├─ 进程内存: " << format_memory_size(process_mb * 1024 * 1024) << std::endl;
    std::cout << "├─ 系统可用内存: " << format_memory_size(available_mb * 1024 * 1024) << std::endl;
    
    if (gpu_memory.second > 0) {
        std::cout << "├─ GPU内存: " << gpu_memory.first << "/" << gpu_memory.second 
                  << " MB (" << std::fixed << std::setprecision(1)
                  << 100.0 * gpu_memory.first / gpu_memory.second << "%)" << std::endl;
    }
    std::cout << std::endl;
}

bool MemoryUtils::is_memory_sufficient(size_t required_mb) {
    size_t available_mb = get_available_memory_mb();
    return available_mb >= required_mb;
}

std::string MemoryUtils::format_memory_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

std::string MemoryUtils::read_file_content(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return "";
    }
    
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

size_t MemoryUtils::parse_memory_value(const std::string& line) {
    std::istringstream iss(line);
    std::string name, value, unit;
    iss >> name >> value >> unit;
    
    try {
        return std::stoul(value);
    } catch (const std::exception&) {
        return 0;
    }
}

// ScopedMemoryMonitor 实现
ScopedMemoryMonitor::ScopedMemoryMonitor(const std::string& scope_name, MemoryMonitor* monitor)
    : scope_name_(scope_name), monitor_(monitor), start_time_(std::chrono::steady_clock::now()) {
    
    start_stats_.process_memory_mb = MemoryUtils::get_process_memory_mb();
    
    std::cout << "🔍 [" << scope_name_ << "] 开始 - 内存: " 
              << start_stats_.process_memory_mb << " MB" << std::endl;
    
    if (monitor_) {
        monitor_->add_memory_checkpoint(scope_name_ + " - 开始");
    }
}

ScopedMemoryMonitor::~ScopedMemoryMonitor() {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
    
    size_t end_memory = MemoryUtils::get_process_memory_mb();
    size_t memory_delta = (end_memory > start_stats_.process_memory_mb) ? 
                         (end_memory - start_stats_.process_memory_mb) : 0;
    
    std::cout << "✅ [" << scope_name_ << "] 结束 - 耗时: " << duration.count() 
              << "ms, 内存增长: " << memory_delta << " MB" << std::endl;
    
    if (monitor_) {
        monitor_->add_memory_checkpoint(scope_name_ + " - 结束");
    }
}

size_t ScopedMemoryMonitor::get_memory_delta_mb() const {
    size_t current_memory = MemoryUtils::get_process_memory_mb();
    return (current_memory > start_stats_.process_memory_mb) ? 
           (current_memory - start_stats_.process_memory_mb) : 0;
}
