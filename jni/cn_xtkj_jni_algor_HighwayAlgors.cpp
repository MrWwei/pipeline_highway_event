#include "cn_xtkj_jni_algor_HighwayAlgors.h"
#include "highway_event.h"
#include <opencv2/opencv.hpp>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>

// 全局实例管理
static std::mutex g_instance_mutex;
static std::map<int, std::unique_ptr<HighwayEventDetector>> g_detectors;
static int g_next_instance_id = 1;

// 辅助函数：从Java字符串获取C++字符串
std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// 辅助函数：创建Java字符串
jstring string_to_jstring(JNIEnv* env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

// 辅助函数：检查并清理JNI异常
bool check_and_clear_exception(JNIEnv* env, const char* context) {
    if (env->ExceptionCheck()) {
        std::cerr << "❌ JNI异常发生在: " << context << std::endl;
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

// 辅助函数：从MatRef获取OpenCV Mat
cv::Mat get_mat_from_matref(JNIEnv* env, jobject matRef) {
    jclass matRefClass = env->GetObjectClass(matRef);
    if (check_and_clear_exception(env, "get_mat_from_matref - GetObjectClass")) {
        return cv::Mat();
    }
    
    // 获取宽度和高度
    jfieldID colsField = env->GetFieldID(matRefClass, "matCols", "I");
    jfieldID rowsField = env->GetFieldID(matRefClass, "matRows", "I");
    jfieldID dataAddrField = env->GetFieldID(matRefClass, "matDataAddr", "J");
    
    if (check_and_clear_exception(env, "get_mat_from_matref - GetFieldID")) {
        env->DeleteLocalRef(matRefClass);
        return cv::Mat();
    }
    
    jint cols = env->GetIntField(matRef, colsField);
    jint rows = env->GetIntField(matRef, rowsField);
    jlong dataAddr = env->GetLongField(matRef, dataAddrField);
    
    if (check_and_clear_exception(env, "get_mat_from_matref - Get*Field")) {
        env->DeleteLocalRef(matRefClass);
        return cv::Mat();
    }
    
    // 从数据地址创建OpenCV Mat
    cv::Mat mat(rows, cols, CV_8UC3, reinterpret_cast<void*>(dataAddr));
    
    env->DeleteLocalRef(matRefClass);
    return mat; // 克隆以确保数据安全
}

// 辅助函数：从HighwayAlgorParam获取配置
HighwayEventConfig get_config_from_param(JNIEnv* env, jobject param) {
    HighwayEventConfig config;
    
    if (!param) return config;
    
    jclass paramClass = env->GetObjectClass(param);
    if (check_and_clear_exception(env, "get_config_from_param - GetObjectClass")) {
        return config;
    }
    
    // 获取功能开关参数
    jfieldID enableEmergencyField = env->GetFieldID(paramClass, "enableEmergencyLaneDetection", "Z");
    jfieldID enableSegShowField = env->GetFieldID(paramClass, "enableSegShow", "Z");
    jfieldID enableLaneShowField = env->GetFieldID(paramClass, "enableLaneShow", "Z");
    
    if (check_and_clear_exception(env, "get_config_from_param - GetFieldID for switches")) {
        env->DeleteLocalRef(paramClass);
        return config;
    }
    
    if (enableSegShowField) {
        config.enable_seg_show = env->GetBooleanField(param, enableSegShowField);
    }
    
    // 获取路径参数
    jfieldID segShowPathField = env->GetFieldID(paramClass, "segShowImagePathString", "Ljava/lang/String;");
    if (segShowPathField) {
        jstring segShowPath = (jstring)env->GetObjectField(param, segShowPathField);
        if (segShowPath) {
            config.seg_show_image_path = jstring_to_string(env, segShowPath);
            env->DeleteLocalRef(segShowPath); // 释放局部引用
        }
    }

    if(enableLaneShowField) {
        config.enable_lane_show = env->GetBooleanField(param, enableLaneShowField);
    }
    jfieldID laneShowPathField = env->GetFieldID(paramClass, "laneShowImagePathString", "Ljava/lang/String;");
    if (laneShowPathField) {
        jstring laneShowPath = (jstring)env->GetObjectField(param, laneShowPathField);
        if (laneShowPath) {
            config.lane_show_image_path = jstring_to_string(env, laneShowPath);
            env->DeleteLocalRef(laneShowPath); // 释放局部引用
        }
    }
    
    // 获取应急车道参数
    jfieldID emergencyWidthField = env->GetFieldID(paramClass, "emergencyLaneWidth", "F");
    jfieldID emergencyHeightField = env->GetFieldID(paramClass, "emergencyLaneHeight", "F");
    
    if (emergencyWidthField) {
        float emergencyWidth = env->GetFloatField(param, emergencyWidthField);
        // 根据应急车道参数调整目标框筛选区域
        config.box_filter_top_fraction = 4.0f / 7.0f;
        config.box_filter_bottom_fraction = 8.0f / 9.0f;
        config.times_car_width = emergencyWidth;
    }
    
    // 设置默认线程配置（可以根据需要调整）
    config.semantic_threads = 4;
    config.mask_threads = 4;
    config.detection_threads = 4;
    config.tracking_threads = 1;
    config.filter_threads = 2;
    config.enable_debug_log = true;
    
    env->DeleteLocalRef(paramClass);
    return config;
}

// 辅助函数：获取事件类型ID
int get_event_type_id(ObjectStatus status) {
    switch (status) {
        case ObjectStatus::OCCUPY_EMERGENCY_LANE:
            return 3; // OCCUPY_EMERGENCY_LANE
        case ObjectStatus::NORMAL:
        default:
            return 0; // 无事件
    }
}

// 辅助函数：创建EventYoloCoor对象（从DetectionBox）
jobject create_event_yolo_coor(JNIEnv* env, const DetectionBox& box) {
    // 获取EventYoloCoor类
    jclass coorClass = env->FindClass("cn/xtkj/jni/algor/helper/EventYoloCoor");
    if (!coorClass) return nullptr;
    
    // 获取构造函数
    jmethodID constructor = env->GetMethodID(coorClass, "<init>", "()V");
    if (!constructor) {
        env->DeleteLocalRef(coorClass);
        return nullptr;
    }
    
    // 创建对象
    jobject coorObj = env->NewObject(coorClass, constructor);
    if (!coorObj) {
        env->DeleteLocalRef(coorClass);
        return nullptr;
    }
    
    // 设置YoloCoor基类字段（坐标信息）
    jfieldID leftField = env->GetFieldID(coorClass, "coorNorthwestLeftPx", "Ljava/lang/Integer;");
    jfieldID topField = env->GetFieldID(coorClass, "coorNorthwestTopPx", "Ljava/lang/Integer;");
    jfieldID rightField = env->GetFieldID(coorClass, "coorSoutheastLeftPx", "Ljava/lang/Integer;");
    jfieldID bottomField = env->GetFieldID(coorClass, "coorSoutheastTopPx", "Ljava/lang/Integer;");
    jfieldID reliabilityField = env->GetFieldID(coorClass, "reliability", "Ljava/lang/Integer;");
    jfieldID typeField = env->GetFieldID(coorClass, "type", "Ljava/lang/Integer;");
    jfieldID trackIdField = env->GetFieldID(coorClass, "trackId", "Ljava/lang/Integer;");
    
    if (check_and_clear_exception(env, "create_event_yolo_coor - GetFieldID")) {
        env->DeleteLocalRef(coorClass);
        return nullptr;
    }
    
    // 获取Integer类和构造函数
    jclass integerClass = env->FindClass("java/lang/Integer");
    if (!integerClass) {
        env->DeleteLocalRef(coorClass);
        return nullptr;
    }
    
    jmethodID integerConstructor = env->GetMethodID(integerClass, "<init>", "(I)V");
    if (!integerConstructor) {
        env->DeleteLocalRef(integerClass);
        env->DeleteLocalRef(coorClass);
        return nullptr;
    }
    
    if (check_and_clear_exception(env, "create_event_yolo_coor - Integer class/constructor")) {
        env->DeleteLocalRef(integerClass);
        env->DeleteLocalRef(coorClass);
        return nullptr;
    }
    
    // 设置坐标字段
    if (leftField && integerConstructor) {
        jobject leftObj = env->NewObject(integerClass, integerConstructor, (jint)box.left);
        env->SetObjectField(coorObj, leftField, leftObj);
        env->DeleteLocalRef(leftObj);
    }
    
    if (topField && integerConstructor) {
        jobject topObj = env->NewObject(integerClass, integerConstructor, (jint)box.top);
        env->SetObjectField(coorObj, topField, topObj);
        env->DeleteLocalRef(topObj);
    }
    
    if (rightField && integerConstructor) {
        jobject rightObj = env->NewObject(integerClass, integerConstructor, (jint)box.right);
        env->SetObjectField(coorObj, rightField, rightObj);
        env->DeleteLocalRef(rightObj);
    }
    
    if (bottomField && integerConstructor) {
        jobject bottomObj = env->NewObject(integerClass, integerConstructor, (jint)box.bottom);
        env->SetObjectField(coorObj, bottomField, bottomObj);
        env->DeleteLocalRef(bottomObj);
    }
    
    if (reliabilityField && integerConstructor) {
        jobject reliabilityObj = env->NewObject(integerClass, integerConstructor, (jint)(box.confidence * 100));
        env->SetObjectField(coorObj, reliabilityField, reliabilityObj);
        env->DeleteLocalRef(reliabilityObj);
    }
    
    if (typeField && integerConstructor) {
        jobject typeObj = env->NewObject(integerClass, integerConstructor, (jint)box.class_id);
        env->SetObjectField(coorObj, typeField, typeObj);
        env->DeleteLocalRef(typeObj);
    }
    
    if (trackIdField && integerConstructor) {
        jobject trackIdObj = env->NewObject(integerClass, integerConstructor, (jint)box.track_id);
        env->SetObjectField(coorObj, trackIdField, trackIdObj);
        env->DeleteLocalRef(trackIdObj);
    }
    
    // 设置EventYoloCoor特有字段（事件ID）
    jfieldID eventIdField = env->GetFieldID(coorClass, "eventId", "I");
    if (eventIdField) {
        int eventId = get_event_type_id(box.status);
        env->SetIntField(coorObj, eventIdField, eventId);
    }
    
    env->DeleteLocalRef(integerClass);
    env->DeleteLocalRef(coorClass);
    return coorObj;
}

/*
 * Class:     cn_xtkj_jni_algor_HighwayAlgors
 * Method:    getVersion
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_cn_xtkj_jni_algor_HighwayAlgors_getVersion
  (JNIEnv *env, jobject) {
    return string_to_jstring(env, "HighwayEvent Pipeline v1.0.0");
}

/*
 * Class:     cn_xtkj_jni_algor_HighwayAlgors
 * Method:    createInstanceCollections
 * Signature: (Lcn/xtkj/jni/algor/HighwayAlgorParam;[Lcn/xtkj/jni/algor/HighwayExample;)[I
 */
JNIEXPORT jintArray JNICALL Java_cn_xtkj_jni_algor_HighwayAlgors_createInstanceCollections
  (JNIEnv *env, jobject, jobject param, jobjectArray examples) {
    
    std::lock_guard<std::mutex> lock(g_instance_mutex);
    
    try {
        // 从参数获取配置
        HighwayEventConfig config = get_config_from_param(env, param);
        
        // 从examples数组获取模型路径
        if (examples) {
            jsize exampleCount = env->GetArrayLength(examples);
            if (exampleCount > 0) {
                jobject firstExample = env->GetObjectArrayElement(examples, 0);
                if (firstExample) {
                    jclass exampleClass = env->GetObjectClass(firstExample);
                    if (check_and_clear_exception(env, "createInstanceCollections - GetObjectClass")) {
                        env->DeleteLocalRef(firstExample);
                        return nullptr;
                    }
                    
                    // 获取车道分割模型路径 (对应 seg_model_path)
                    jfieldID laneSegmentModelPathField = env->GetFieldID(exampleClass, "laneSegmentModelPath", "Ljava/lang/String;");
                    if (laneSegmentModelPathField) {
                        jstring laneSegmentModelPath = (jstring)env->GetObjectField(firstExample, laneSegmentModelPathField);
                        if (laneSegmentModelPath) {
                            config.seg_model_path = jstring_to_string(env, laneSegmentModelPath);
                            env->DeleteLocalRef(laneSegmentModelPath); // 释放局部引用
                        }
                    }
                    
                    // 获取机动车检测模型路径 (对应 det_model_path)
                    jfieldID vehTargetModelPathField = env->GetFieldID(exampleClass, "vehTargetModelPath", "Ljava/lang/String;");
                    if (vehTargetModelPathField) {
                        jstring vehTargetModelPath = (jstring)env->GetObjectField(firstExample, vehTargetModelPathField);
                        if (vehTargetModelPath) {
                            config.det_model_path = jstring_to_string(env, vehTargetModelPath);
                            env->DeleteLocalRef(vehTargetModelPath); // 释放局部引用
                        }
                    }
                    
                    // 获取跟踪模型路径 (如果需要的话)
                    jfieldID trackModelPathField = env->GetFieldID(exampleClass, "trackModelPath", "Ljava/lang/String;");
                    if (trackModelPathField) {
                        jstring trackModelPath = (jstring)env->GetObjectField(firstExample, trackModelPathField);
                        if (trackModelPath) {
                            // 这里可以设置到config的跟踪模型路径字段，如果有的话
                            env->DeleteLocalRef(trackModelPath); // 释放局部引用
                        }
                    }
                    
                    // 获取行人检测模型路径 (如果需要的话)
                    jfieldID personTargetModelPathField = env->GetFieldID(exampleClass, "personTargetModelPath", "Ljava/lang/String;");
                    if (personTargetModelPathField) {
                        jstring personTargetModelPath = (jstring)env->GetObjectField(firstExample, personTargetModelPathField);
                        if (personTargetModelPath) {
                            // 这里可以设置到config的行人检测模型路径字段，如果有的话
                            env->DeleteLocalRef(personTargetModelPath); // 释放局部引用
                        }
                    }
                    
                    env->DeleteLocalRef(exampleClass);
                    env->DeleteLocalRef(firstExample);
                }
            }
        }
        
        // 创建检测器实例
        int instance_id = g_next_instance_id++;
        
        auto detector = create_highway_event_detector();
        if (!detector) {
            std::cerr << "❌ 创建检测器失败" << std::endl;
            return nullptr;
        }
        
        // 初始化检测器
        if (!detector->initialize(config)) {
            std::cerr << "❌ 检测器初始化失败" << std::endl;
            detector->stop(); // 确保清理资源
            return nullptr;
        }
        
        // 启动检测器
        if (!detector->start()) {
            std::cerr << "❌ 检测器启动失败" << std::endl;
            detector->stop(); // 确保清理资源
            return nullptr;
        }
        
        g_detectors[instance_id] = std::move(detector);
        
        // 创建Java int数组返回（只返回一个实例）
        jintArray result = env->NewIntArray(1);
        if (result) {
            jint instance_array[] = {instance_id};
            env->SetIntArrayRegion(result, 0, 1, instance_array);
        }
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 创建实例时发生异常: " << e.what() << std::endl;
        return nullptr;
    }
}

/*
 * Class:     cn_xtkj_jni_algor_HighwayAlgors
 * Method:    changeParam
 * Signature: (Lcn/xtkj/jni/algor/HighwayAlgorParam;)I
 */
JNIEXPORT jint JNICALL Java_cn_xtkj_jni_algor_HighwayAlgors_changeParam
  (JNIEnv *env, jobject, jobject param) {

    std::lock_guard<std::mutex> lock(g_instance_mutex);
    if (!param) {
        std::cerr << "❌ 参数为null" << std::endl;
        return -1; // 错误状态
    }
    // 从参数获取配置
    HighwayEventConfig config = get_config_from_param(env, param);
    if (config.seg_model_path.empty() || config.det_model_path.empty()) {
        std::cerr << "❌ 模型路径不能为空" << std::endl;
        return -1; // 错误状态
    }
    // 遍历所有实例，更新配置
    for (auto& pair : g_detectors) {
        int instanceId = pair.first;
        auto& detector = pair.second;
        if (!detector) {
            std::cerr << "❌ 检测器实例 " << instanceId << " 为空" << std::endl;
            continue; // 跳过空实例
        }   
        // 更新检测器配置
        if (!detector->change_params(config)) {
            std::cerr << "❌ 更新检测器实例 " << instanceId << " 配置失败" << std::endl;
            return -1; // 错误状态
        }
        std::cout << "✅ 检测器实例 " << instanceId << " 配置更新成功" << std::endl;
    }
    
    // 当前实现：参数变更需要重新创建实例
    // 这里可以返回一个状态码表示需要重新创建
    return 1; // 表示需要重新创建实例
}

/*
 * Class:     cn_xtkj_jni_algor_HighwayAlgors
 * Method:    putMat
 * Signature: (ILcn/xtkj/jni/algor/data/MatRef;)J
 */
JNIEXPORT jlong JNICALL Java_cn_xtkj_jni_algor_HighwayAlgors_putMat
  (JNIEnv *env, jobject, jint instanceId, jobject matRef) {
    
    if (!matRef) {
        std::cerr << "❌ MatRef参数为null" << std::endl;
        return -1;
    }
    
    // 先获取图像数据，避免在持锁时进行JNI操作
    cv::Mat image = get_mat_from_matref(env, matRef);
    if (image.empty()) {
        std::cerr << "❌ 获取图像数据失败" << std::endl;
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(g_instance_mutex);
    
    try {
        // 查找检测器实例
        auto it = g_detectors.find(instanceId);
        if (it == g_detectors.end()) {
            std::cerr << "❌ 找不到实例 " << instanceId << std::endl;
            return -1;
        }
        
        auto& detector = it->second;
        if (!detector) {
            std::cerr << "❌ 检测器实例 " << instanceId << " 为空" << std::endl;
            return -1;
        }
        
        // 添加图像到检测器
        int64_t frame_id = detector->add_frame(std::move(image));

        
        if (frame_id < 0) {
            std::cerr << "❌ 添加图像到检测器失败" << std::endl;
            return -1;
        }
        
        return static_cast<jlong>(frame_id); // 返回帧ID
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 添加图像时发生异常: " << e.what() << std::endl;
        return -1;
    }
}

/*
 * Class:     cn_xtkj_jni_algor_HighwayAlgors
 * Method:    takeRes
 * Signature: (IJ)[Lcn/xtkj/jni/algor/helper/EventYoloCoor;
 */
JNIEXPORT jobjectArray JNICALL Java_cn_xtkj_jni_algor_HighwayAlgors_takeRes
  (JNIEnv *env, jobject, jint instanceId, jlong frameId) {
    
    std::lock_guard<std::mutex> lock(g_instance_mutex);
    
    try {
        // 查找检测器实例
        auto it = g_detectors.find(instanceId);
        if (it == g_detectors.end()) {
            std::cerr << "❌ 找不到实例 " << instanceId << std::endl;
            return nullptr;
        }
        
        auto& detector = it->second;
        if (!detector) {
            std::cerr << "❌ 检测器实例 " << instanceId << " 为空" << std::endl;
            return nullptr;
        }
        
        // 获取处理结果
        auto result = detector->get_result(static_cast<uint64_t>(frameId));
        // cv::Mat image = result.srcImage;
        // for(auto & box : result.detections) {
        //     // 确保每个检测框的track_id唯一
        //     cv::rectangle(image, 
        //                 cv::Point(box.left, box.top), 
        //                 cv::Point(box.right, box.bottom), 
        //                 cv::Scalar(0, 255, 0), 2);
        // }
        // int image_id = static_cast<int>(frameId);
        // cv::imwrite("output_" + std::to_string(image_id) + ".jpg", image);
        
        if (result.status != ResultStatus::SUCCESS) {
            std::cerr << "❌ 获取帧 " << frameId << " 结果失败，状态: " << static_cast<int>(result.status) << std::endl;
            return nullptr;
        }
        
        // 使用ProcessResult中的detections
        const auto& boxes = result.detections;
        
        // 创建Java数组
        jclass coorClass = env->FindClass("cn/xtkj/jni/algor/helper/EventYoloCoor");
        if (!coorClass) {
            std::cerr << "❌ 找不到EventYoloCoor类" << std::endl;
            return nullptr;
        }
        
        jobjectArray resultArray = env->NewObjectArray(boxes.size(), coorClass, nullptr);
        
        if (!resultArray) {
            std::cerr << "❌ 创建结果数组失败" << std::endl;
            env->DeleteLocalRef(coorClass);
            return nullptr;
        }
        
        // 填充结果数组
        for (size_t i = 0; i < boxes.size(); i++) {
            jobject coorObj = create_event_yolo_coor(env, boxes[i]);
            if (coorObj) {
                env->SetObjectArrayElement(resultArray, i, coorObj);
                if (check_and_clear_exception(env, "takeRes - SetObjectArrayElement")) {
                    env->DeleteLocalRef(coorObj);
                    env->DeleteLocalRef(coorClass);
                    return nullptr;
                }
                env->DeleteLocalRef(coorObj);
            } else {
                std::cerr << "⚠️ 创建EventYoloCoor对象失败，索引: " << i << std::endl;
            }
        }
        
        env->DeleteLocalRef(coorClass);
        
        return resultArray;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 获取结果时发生异常: " << e.what() << std::endl;
        return nullptr;
    }
}

/*
 * Class:     cn_xtkj_jni_algor_HighwayAlgors
 * Method:    releaseInstanceCollection
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_cn_xtkj_jni_algor_HighwayAlgors_releaseInstanceCollection
  (JNIEnv *, jobject, jint instanceId) {
    
    std::lock_guard<std::mutex> lock(g_instance_mutex);
    
    try {
        auto it = g_detectors.find(instanceId);
        if (it == g_detectors.end()) {
            std::cerr << "❌ 找不到要释放的实例 " << instanceId << std::endl;
            return -1;
        }
        
        auto& detector = it->second;
        if (detector) {
            detector->stop();
        }
        
        g_detectors.erase(it);
        
        return 0; // 成功
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 释放实例时发生异常: " << e.what() << std::endl;
        return -1;
    }
}

// 添加一个全局清理函数
static void cleanup_all_instances() {
    std::lock_guard<std::mutex> lock(g_instance_mutex);
    
    if (!g_detectors.empty()) {
        for (auto& pair : g_detectors) {
            if (pair.second) {
                pair.second->stop();
            }
        }
        
        g_detectors.clear();
    }
}

// 添加JNI_OnLoad和JNI_OnUnload来管理生命周期
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    cleanup_all_instances();
}
