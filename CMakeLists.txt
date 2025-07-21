cmake_minimum_required(VERSION 3.16)
project(PipelineHighwayEvent)


include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Find required packages
find_package(Threads REQUIRED)

# thirdparty
set(ThirdParty /home/ubuntu/ThirdParty)
set(tensorRT_libs_DIR ${ThirdParty}/TensorRT-8.5.1.7/lib)
set(tensorRT_headers_DIR ${ThirdParty}/TensorRT-8.5.1.7/include)
include_directories( ${tensorRT_headers_DIR} )
link_directories(${tensorRT_libs_DIR})

include_directories(${ThirdParty}/opencv-4.5.4/build/install/include/opencv4)
link_directories(${ThirdParty}/opencv-4.5.4/build/install/lib)

include_directories(/home/ubuntu/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/include)
link_directories(/home/ubuntu/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib)

# object_detection
add_library(detect_lib SHARED IMPORTED)
set_target_properties( detect_lib PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/object_detect/build/libdetect_yl_nc11_trt_X86_V6.1.5_u.so )
include_directories(/home/ubuntu/wtwei/object_detect/include)

# object_track
add_library(track_lib SHARED IMPORTED)
set_target_properties( track_lib PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/object_track/build/libtrack_btk_n_x86_V6.3.so )
include_directories(/home/ubuntu/wtwei/object_track/include)

add_library(road_seg_interface SHARED IMPORTED)
set_target_properties( road_seg_interface PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/highway_event/road_seg_sdk/build/libroad_seg_sdk_V1.0.so )
include_directories(/home/ubuntu/wtwei/highway_event/road_seg_sdk)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/image_processor.cpp
    src/semantic_segmentation.cpp
    src/object_detection.cpp
    src/pipeline_manager.cpp
    src/image_data.cpp
)

# Headers
set(HEADERS
    include/image_processor.h
    include/semantic_segmentation.h
    include/object_detection.h
    include/pipeline_manager.h
    include/image_data.h
    include/thread_safe_queue.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} Threads::Threads)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -O2
)
