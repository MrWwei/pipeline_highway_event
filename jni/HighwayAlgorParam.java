package cn.xtkj.jni.algor;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: HighwayAlgorParam
 * @date 2025年07月14日 17:17:46
 */
public class HighwayAlgorParam {

    // 功能开关参数
    private boolean enableEmergencyLaneDetection = true; // 是否启用应急车道检测
    private boolean enableLicensePlateRecognition = true; // 是否启用车牌识别
    private boolean enablePersonDetection = true;        // 是否启用行人检测
    private boolean enableLaneShow = false;              // 是否启用车道线可视化
    private boolean enableSegShow = false;               // 是否启用分割可视化（如车道线分割）


    // 违停检测参数
    private boolean enableParkingDetection = true; // 是否启用违停检测
    private int staticThreshold = 5;               // 禁止阈值
    private int minStaticDuration = 2;             // 最小运动持续帧数

    // 应急车道判断参数
    private float emergencyLaneWidth = 1.0f;  // 几倍的车辆宽度作为应急车道宽度
    private float emergencyLaneHeight = 0.5f; // 应急车道高度占比

    private String segShowImagePathString = ""; // 分割可视化图片路径
    private String laneShowImagePathString = ""; // 车道线可视化图片路径


    public boolean getEnableEmergencyLaneDetection() {
        return enableEmergencyLaneDetection;
    }

    public void setEnableEmergencyLaneDetection(boolean enableEmergencyLaneDetection) {
        this.enableEmergencyLaneDetection = enableEmergencyLaneDetection;
    }

    public boolean getEnableLicensePlateRecognition() {
        return enableLicensePlateRecognition;
    }

    public void setEnableLicensePlateRecognition(boolean enableLicensePlateRecognition) {
        this.enableLicensePlateRecognition = enableLicensePlateRecognition;
    }

    public boolean getEnablePersonDetection() {
        return enablePersonDetection;
    }

    public void setEnablePersonDetection(boolean enablePersonDetection) {
        this.enablePersonDetection = enablePersonDetection;
    }

    public boolean getEnableLaneShow() {
        return enableLaneShow;
    }

    public void setEnableLaneShow(boolean enableLaneShow) {
        this.enableLaneShow = enableLaneShow;
    }

    public boolean getEnableParkingDetection() {
        return enableParkingDetection;
    }

    public void setEnableParkingDetection(boolean enableParkingDetection) {
        this.enableParkingDetection = enableParkingDetection;
    }

    public int getStaticThreshold() {
        return staticThreshold;
    }

    public void setStaticThreshold(int staticThreshold) {
        this.staticThreshold = staticThreshold;
    }

    public int getMinStaticDuration() {
        return minStaticDuration;
    }

    public void setMinStaticDuration(int minStaticDuration) {
        this.minStaticDuration = minStaticDuration;
    }

    public float getEmergencyLaneWidth() {
        return emergencyLaneWidth;
    }

    public void setEmergencyLaneWidth(float emergencyLaneWidth) {
        this.emergencyLaneWidth = emergencyLaneWidth;
    }

    public float getEmergencyLaneHeight() {
        return emergencyLaneHeight;
    }

    public void setEmergencyLaneHeight(float emergencyLaneHeight) {
        this.emergencyLaneHeight = emergencyLaneHeight;
    }

    public boolean getEnableSegShow() {
        return enableSegShow;
    }

    public void setEnableSegShow(boolean enableSegShow) {
        this.enableSegShow = enableSegShow;
    }

    public String getSegShowImagePathString() {
        return segShowImagePathString;
    }

    public void setSegShowImagePathString(String segShowImagePathString) {
        this.segShowImagePathString = segShowImagePathString;
    }

    public String getLaneShowImagePathString() {
        return laneShowImagePathString;
    }

    public void setLaneShowImagePathString(String laneShowImagePathString) {
        this.laneShowImagePathString = laneShowImagePathString;
    }
}