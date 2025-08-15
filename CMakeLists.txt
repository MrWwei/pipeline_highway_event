cmake_minimum_required(VERSION 3.16)
project(PipelineHighwayEvent)

enable_language(CUDA)
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
# cuda
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
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
set_target_properties( road_seg_interface PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/seg_trt/build/libroad_seg_SDK_V1.0.so )
include_directories(/home/ubuntu/wtwei/seg_trt/include)

add_library(vehicle_parking SHARED IMPORTED)
set_target_properties( vehicle_parking PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/vehicle_parking/build/libvehicle_parking_detect_V1.0.so )
include_directories(/home/ubuntu/wtwei/vehicle_parking/include)


# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/highway_event.cpp
    src/image_processor.cpp
    src/process_mask.cu
    src/semantic_segmentation.cpp
    src/mask_postprocess.cpp
    src/object_detection.cpp
    src/object_tracking.cpp
    src/event_determine.cpp
    src/pipeline_manager.cpp
    src/image_data.cpp
    src/event_utils.cc
    src/thread_pool.cpp
    # 新增批次处理模块
    src/batch_data.cpp
    src/batch_semantic_segmentation.cpp
    src/batch_mask_postprocess.cpp
    src/batch_object_detection.cpp
    src/batch_object_tracking.cpp
    src/batch_event_determine.cpp
    src/batch_pipeline_manager.cpp
    # 内存监控模块
    src/memory_monitor.cpp
)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/process_mask.cu  )


set(sdk_target_name highway_event_sdk_v1.0)
add_library(${sdk_target_name} SHARED ${SOURCES})
target_link_libraries(${sdk_target_name}

    Threads::Threads 
    detect_lib
    track_lib
    road_seg_interface
    vehicle_parking
    opencv_world
    nvinfer
    nvinfer_plugin
    cudnn
    cuda          
   cudart    
   cublas    
   cublasLt  
   curand    
   cusolver  
   cusparse  
   cufft  
)
# #################### jni #################
include_directories(${ThirdParty}/jdk1.8.0_381/include)
include_directories(${ThirdParty}/jdk1.8.0_381/include/linux)

# V2.0
set(jni_target_name highway_event_X86_SDK_V1.0_JNI_V1.0 )
include_directories(${CMAKE_SOURCE_DIR}/jni)
add_library(${jni_target_name} SHARED ${CMAKE_SOURCE_DIR}/jni/cn_xtkj_jni_algor_HighwayAlgors.cpp )
target_link_libraries(${jni_target_name} ${sdk_target_name}) 


add_executable(HighwayEventDemo highway_event_demo.cpp)

target_link_libraries(HighwayEventDemo Threads::Threads 
    ${sdk_target_name}
)
