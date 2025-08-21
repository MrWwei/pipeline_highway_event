#ifndef LOGGER_MANAGER_H
#define LOGGER_MANAGER_H

#include <log4cplus/log4cplus.h>
#include <memory>
#include <string>

/**
 * 日志管理器单例类
 * 提供统一的日志接口，所有模块通过此类进行日志输出
 */
class LoggerManager {
public:
    /**
     * 获取日志管理器单例
     */
    static LoggerManager& getInstance();

    /**
     * 初始化日志系统
     * @param log_file_path 日志文件路径
     * @param enable_console 是否同时输出到控制台
     * @param log_level 日志级别 (DEBUG, INFO, WARN, ERROR)
     */
    bool initialize(const std::string& log_file_path = "highway_event.log", 
                   bool enable_console = true, 
                   const std::string& log_level = "INFO");

    /**
     * 获取日志器
     */
    log4cplus::Logger& getLogger();

    /**
     * 便捷的日志宏定义
     */
    void logDebug(const std::string& message);
    void logInfo(const std::string& message);
    void logWarn(const std::string& message);
    void logError(const std::string& message);

    /**
     * 关闭日志系统
     */
    void shutdown();

private:
    LoggerManager() = default;
    ~LoggerManager();
    
    // 禁止拷贝和赋值
    LoggerManager(const LoggerManager&) = delete;
    LoggerManager& operator=(const LoggerManager&) = delete;

    log4cplus::Logger logger_;
    bool initialized_ = false;
};

// 便捷的日志宏定义
#define LOG_DEBUG(msg) LoggerManager::getInstance().logDebug(msg)
#define LOG_INFO(msg) LoggerManager::getInstance().logInfo(msg)
#define LOG_WARN(msg) LoggerManager::getInstance().logWarn(msg)
#define LOG_ERROR(msg) LoggerManager::getInstance().logError(msg)

// 格式化日志宏
#define LOG_DEBUG_F(fmt, ...) do { \
    char buffer[1024]; \
    snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__); \
    LOG_DEBUG(std::string(buffer)); \
} while(0)

#define LOG_INFO_F(fmt, ...) do { \
    char buffer[1024]; \
    snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__); \
    LOG_INFO(std::string(buffer)); \
} while(0)

#define LOG_WARN_F(fmt, ...) do { \
    char buffer[1024]; \
    snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__); \
    LOG_WARN(std::string(buffer)); \
} while(0)

#define LOG_ERROR_F(fmt, ...) do { \
    char buffer[1024]; \
    snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__); \
    LOG_ERROR(std::string(buffer)); \
} while(0)

#endif // LOGGER_MANAGER_H
