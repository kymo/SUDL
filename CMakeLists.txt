
#cmake最低版本需求，不加入此行会受到警告信息
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(HELLO) #项目名称
#把当前目录(.)下所有源代码文件和头文件加入变量SRC_LIST
#生成应用程序 hello (在windows下会自动生成hello.exe)

include_directories(matrix)
include_directories(layers)
include_directories(utils)
include_directories(net)

aux_source_directory(./layers DIR_SRCS)
add_library(Lib ${DIR_SRCS})

# 生成wordseg可执行程序
aux_source_directory(./example/wordseg WS_SRCS)
add_executable(wordseg ${WS_SRCS})
target_link_libraries(wordseg Lib)

# 生成number digital 可执行
aux_source_directory(./example/digital_recognition DR_SRCS)
add_executable(dr ${DR_SRCS})
target_link_libraries(dr Lib)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/bin)

#SET_TARGET_PROPERTIES(dr wordseg PROPERTIES RUNTIME_OUTPUT_DIRECTORY output/bin)
