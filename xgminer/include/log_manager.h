#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <ctime>
#include <iomanip>
#include <map>
#include <set>
#include <filesystem>
#include <cstdlib>  // for getenv

enum class LogLevel {
    INFO,
    WARNING,
    ERROR
};


class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    Logger() : consoleEnabled_(true) {}
    ~Logger() {
        if (currentFileStream_.is_open()) {
            currentFileStream_.close();
        }
    }

    // 设置当前日志文件（运行期间第一次打开时截断，之后追加）
    void setFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 获取 ROOT_PATH 环境变量
        const char* rootPath = std::getenv("XGMINER_ROOT");
        if (!rootPath) {
            std::cerr << "ERROR: ROOT_PATH environment variable not set!" << std::endl;
            return;
        }
        std::cout << "XGMINER_ROOT: " << rootPath << std::endl;

        // 构建完整路径：$ROOT_PATH/{relativePath}
        std::filesystem::path fullPath(rootPath);
        fullPath /= filename;  // 自动处理路径分隔符

        // 创建目录（如果不存在）
        // std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::filesystem::create_directories(fullPath.parent_path());

        // 检查是否是运行期间第一次打开该文件
        bool is_first_open = false;
        {
            std::lock_guard<std::mutex> files_lock(files_mutex_);
            is_first_open = opened_files_.find(fullPath.string()) == opened_files_.end();
        }

        // 关闭当前文件流（如果已打开）
        if (currentFileStream_.is_open()) {
            currentFileStream_.close();
        }

        // 打开模式：第一次用截断，之后用追加
        std::ios_base::openmode mode = std::ios_base::out;
        if (is_first_open) {
            mode |= std::ios_base::trunc;  // 截断（覆盖）
            // 记录该文件已被打开过
            std::lock_guard<std::mutex> files_lock(files_mutex_);
            opened_files_.insert(fullPath.string());
        } else {
            mode |= std::ios_base::app;    // 追加
        }

        // 打开文件
        currentFileStream_.open(fullPath, mode);
        if (!currentFileStream_.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    // 日志记录方法（使用当前设置的文件）
    void log(LogLevel level, const std::string& message, const char* file, int line) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string formattedMessage = formatMessage(level, message, file, line);
        
        // 输出到控制台
        if (consoleEnabled_) {
            std::cout << formattedMessage << std::endl;
        }

        // 输出到当前文件
        if (currentFileStream_.is_open()) {
            currentFileStream_ << formattedMessage << std::endl;
        }
    }

    // 启用/禁用控制台输出
    void setConsoleEnabled(bool enabled) {
        consoleEnabled_ = enabled;
    }

    std::string formatMessage(LogLevel level, const std::string& message, 
                             const char* file, int line) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;

        // 时间戳
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");

        // 日志级别和颜色
        std::string levelStr;
        std::string colorCode;
        switch (level) {
            case LogLevel::INFO:
                levelStr = "INFO";
                colorCode = "\033[32m"; // green
                break;
            case LogLevel::WARNING:
                levelStr = "WARNING";
                colorCode = "\033[34m"; // blue
                break;
            case LogLevel::ERROR:
                levelStr = "ERROR";
                colorCode = "\033[31m"; // red
                break;
        }

        // 文件位置
        std::string fileStr = file;
        if (fileStr.size() > 30) {
            fileStr = "..." + fileStr.substr(fileStr.size()-30);
        }

        // 组合消息
        std::string resetColor = "\033[0m";
        ss << " [" << colorCode << levelStr << resetColor 
           << "] " << fileStr << ":" << line << " - " << message;
        return ss.str();
    }
private:
    bool consoleEnabled_;
    std::ofstream currentFileStream_;
    std::mutex mutex_;
    std::set<std::string> opened_files_;  // 记录运行期间已打开过的文件
    std::mutex files_mutex_;              // 保护 opened_files_
};

// 定义日志宏
#define LOG(level, msg) Logger::getInstance().log(level, msg, __FILE__, __LINE__)
#define LOG_INFO(msg) LOG(LogLevel::INFO, msg)
#define LOG_WARNING(msg) LOG(LogLevel::WARNING, msg)
#define LOG_ERROR(msg) LOG(LogLevel::ERROR, msg)


#endif // LOG_MANAGER_H