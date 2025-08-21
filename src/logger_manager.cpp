#include "logger_manager.h"
#include <log4cplus/configurator.h>
#include <log4cplus/fileappender.h>
#include <log4cplus/consoleappender.h>
#include <log4cplus/layout.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

LoggerManager& LoggerManager::getInstance() {
    static LoggerManager instance;
    return instance;
}

bool LoggerManager::initialize(const std::string& log_file_path, 
                              bool enable_console, 
                              const std::string& log_level) {
    if (initialized_) {
        return true;
    }

    try {
        // 初始化log4cplus
        log4cplus::initialize();

        // 创建日志器
        logger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("HighwayEvent"));

        // 设置日志级别
        log4cplus::LogLevel level = log4cplus::INFO_LOG_LEVEL;
        if (log_level == "DEBUG") {
            level = log4cplus::DEBUG_LOG_LEVEL;
        } else if (log_level == "INFO") {
            level = log4cplus::INFO_LOG_LEVEL;
        } else if (log_level == "WARN") {
            level = log4cplus::WARN_LOG_LEVEL;
        } else if (log_level == "ERROR") {
            level = log4cplus::ERROR_LOG_LEVEL;
        }
        logger_.setLogLevel(level);

        // 创建日志布局模式
        log4cplus::tstring pattern = LOG4CPLUS_TEXT("%D{%Y-%m-%d %H:%M:%S.%q} [%p] %c - %m%n");
        std::unique_ptr<log4cplus::Layout> layout(new log4cplus::PatternLayout(pattern));

        // 创建文件Appender
        log4cplus::SharedAppenderPtr fileAppender(
            new log4cplus::FileAppender(log4cplus::tstring(log_file_path.begin(), log_file_path.end()))
        );
        fileAppender->setName(LOG4CPLUS_TEXT("FileAppender"));
        fileAppender->setLayout(std::unique_ptr<log4cplus::Layout>(new log4cplus::PatternLayout(pattern)));
        logger_.addAppender(fileAppender);

        // 如果启用控制台输出，创建控制台Appender
        if (enable_console) {
            log4cplus::SharedAppenderPtr consoleAppender(new log4cplus::ConsoleAppender());
            consoleAppender->setName(LOG4CPLUS_TEXT("ConsoleAppender"));
            consoleAppender->setLayout(std::unique_ptr<log4cplus::Layout>(new log4cplus::PatternLayout(pattern)));
            logger_.addAppender(consoleAppender);
        }

        initialized_ = true;
        
        // 记录初始化成功
        LOG4CPLUS_INFO(logger_, "HighwayEvent 日志系统初始化成功，日志文件: " << log_file_path);
        
        return true;
        
    } catch (const std::exception& e) {
        // 如果日志初始化失败，使用标准输出
        std::cerr << "❌ 日志系统初始化失败: " << e.what() << std::endl;
        return false;
    }
}

log4cplus::Logger& LoggerManager::getLogger() {
    return logger_;
}

void LoggerManager::logDebug(const std::string& message) {
    if (initialized_) {
        LOG4CPLUS_DEBUG(logger_, message);
    }
}

void LoggerManager::logInfo(const std::string& message) {
    if (initialized_) {
        LOG4CPLUS_INFO(logger_, message);
    } else {
        std::cout << "[INFO] " << message << std::endl;
    }
}

void LoggerManager::logWarn(const std::string& message) {
    if (initialized_) {
        LOG4CPLUS_WARN(logger_, message);
    } else {
        std::cout << "[WARN] " << message << std::endl;
    }
}

void LoggerManager::logError(const std::string& message) {
    if (initialized_) {
        LOG4CPLUS_ERROR(logger_, message);
    } else {
        std::cerr << "[ERROR] " << message << std::endl;
    }
}

void LoggerManager::shutdown() {
    if (initialized_) {
        LOG4CPLUS_INFO(logger_, "HighwayEvent 日志系统关闭");
        log4cplus::Logger::shutdown();
        initialized_ = false;
    }
}

LoggerManager::~LoggerManager() {
    shutdown();
}
