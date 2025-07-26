package cn.xtkj.jni.algor;


import com.alibaba.fastjson.annotation.JSONField;

import java.util.concurrent.ExecutorService;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: HighwayExample
 * @date 2025年07月14日 17:07:52
 */
public class HighwayExample {
    private int ExampleId;//应用侧设置的实例ID
    private int gpuId=0;//使用的GPUID
    private int instanceId;//SDK层实例ID
    private String highwayJniPath;//JNI路径
    private String vehTargetModelPath;//机动车检测模型文件路径
    private String personTargetModelPath;//行人检测模型文件路径
    private String trackModelPath;//跟踪检测模型文件路径
    private String laneSegmentModelPath;//车道分割模型文件路径

    @JSONField(serialize = false)
    private ExecutorService executorService;

    public int getInstanceId() {
        return instanceId;
    }

    public void setInstanceId(int instanceId) {
        this.instanceId = instanceId;
    }

    public String getHighwayJniPath() {
        return highwayJniPath;
    }

    public void setHighwayJniPath(String highwayJniPath) {
        this.highwayJniPath = highwayJniPath;
    }

    public String getVehTargetModelPath() {
        return vehTargetModelPath;
    }

    public void setVehTargetModelPath(String vehTargetModelPath) {
        this.vehTargetModelPath = vehTargetModelPath;
    }

    public String getPersonTargetModelPath() {
        return personTargetModelPath;
    }

    public void setPersonTargetModelPath(String personTargetModelPath) {
        this.personTargetModelPath = personTargetModelPath;
    }

    public String getTrackModelPath() {
        return trackModelPath;
    }

    public void setTrackModelPath(String trackModelPath) {
        this.trackModelPath = trackModelPath;
    }

    public String getLaneSegmentModelPath() {
        return laneSegmentModelPath;
    }

    public void setLaneSegmentModelPath(String laneSegmentModelPath) {
        this.laneSegmentModelPath = laneSegmentModelPath;
    }

    public boolean isLoaded() {
        return instanceId>0;
    }

    public int getExampleId() {
        return ExampleId;
    }

    public void setExampleId(int exampleId) {
        ExampleId = exampleId;
    }

    public ExecutorService getExecutorService() {
        return executorService;
    }

    public void setExecutorService(ExecutorService executorService) {
        this.executorService = executorService;
    }

    public int getGpuId() {
        return gpuId;
    }

    public void setGpuId(int gpuId) {
        this.gpuId = gpuId;
    }
}