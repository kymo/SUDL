 /*
 * @file sub_log.h
 * @date 2017/06/12 14:49:27
 * @version $Revision$ 
 * @brief 
 *  
 **/
#ifndef __SUB_LOG_H
#define __SUB_LOG_H

#include <iostream>
#include <vector>
#include <time.h>
#include <pthread.h>
#include <stdarg.h>
#include <string>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

namespace sub_dl {

#define DEBUG 0
#define WARN 1
#define NOTICE 2
#define FATAL 3
#define DEBUG_LOG(fmt, ...) SubLogger<std::string>::_get_instance()->_write_log(DEBUG, fmt, ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) SubLogger<std::string>::_get_instance()->_write_log(WARN, fmt, ##__VA_ARGS__)
#define NOTICE_LOG(fmt, ...) SubLogger<std::string>::_get_instance()->_write_log(NOTICE, fmt, ##__VA_ARGS__)
#define FATAL_LOG(fmt, ...) SubLogger<std::string>::_get_instance()->_write_log(FATAL, fmt, ##__VA_ARGS__)


template <typename T>
class SubLogger {
    
public:
    static SubLogger* _get_instance() {
        if (NULL == _instance) {
            _instance = new SubLogger();
        }
        return _instance;
    }

    void _write_log(int log_type, const char* format, ...) {
        char log_line[MAX_LOG_LEN];
        log_line[0] = '\0';
        switch (log_type) {
        case DEBUG:
            strcat(log_line, " [DEBUG] ");
            break;
        case WARN:
            strcat(log_line, " [WARNING] ");
            break;
        case NOTICE:
            strcat(log_line, " [NOTICE] ");
            break;
        case FATAL:
            strcat(log_line, " [FATAL] ");
            break;
        default:
            return;
        }
        va_list arg_ptr;
        va_start(arg_ptr, format);
        vsnprintf(log_line + strlen(log_line), MAX_LOG_LEN - strlen(log_line), format, arg_ptr);
        va_end(arg_ptr);
        std::cout << log_line << std::endl;
    }

    ~SubLogger() {
    }
private:
    static SubLogger* _instance;
    SubLogger() {}
    
    const static int MAX_LOG_LEN = 1024;
    const static int LOG_OK = 1;
    const static int LOG_FAIL = 0;

};

template <typename T>
SubLogger<T>* SubLogger<T>::_instance = NULL;

}


#endif  


/* vim: set ts=4 sw=4 sts=4 tw=100 */
