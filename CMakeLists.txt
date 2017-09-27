CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(SUDL)

include_directories(matrix)
include_directories(layers)
include_directories(utils)
include_directories(net)

aux_source_directory(./layers DIR_SRCS)
add_library(Lib ${DIR_SRCS})

aux_source_directory(./example/wordseg WS_SRCS)
add_executable(wordseg ${WS_SRCS})
target_link_libraries(wordseg Lib)

aux_source_directory(./example/digital_recognition DR_SRCS)
add_executable(dr ${DR_SRCS})
target_link_libraries(dr Lib)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/bin)
