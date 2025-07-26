// 测试JNI模型路径解析功能
import cn.xtkj.jni.algor.HighwayExample;

public class TestModelPathParsing {
    
    public static void main(String[] args) {
        // 创建HighwayExample实例
        HighwayExample example = new HighwayExample();
        
        // 设置模型路径
        example.setVehTargetModelPath("/path/to/vehicle/detection/model.onnx");
        example.setLaneSegmentModelPath("/path/to/lane/segmentation/model");
        example.setPersonTargetModelPath("/path/to/person/detection/model.onnx");
        example.setTrackModelPath("/path/to/tracking/model.onnx");
        
        // 打印设置的路径
        System.out.println("=== 设置的模型路径 ===");
        System.out.println("车辆检测模型: " + example.getVehTargetModelPath());
        System.out.println("车道分割模型: " + example.getLaneSegmentModelPath());
        System.out.println("行人检测模型: " + example.getPersonTargetModelPath());
        System.out.println("跟踪模型: " + example.getTrackModelPath());
        
        // 创建数组
        HighwayExample[] examples = {example};
        
        System.out.println("\n=== 准备传递给JNI ===");
        System.out.println("数组长度: " + examples.length);
        System.out.println("第一个实例的车辆检测模型路径: " + examples[0].getVehTargetModelPath());
        
        // 注意：实际的JNI调用需要先加载native库
        // System.loadLibrary("highway_event_jni");
        // HighwayAlgors algors = new HighwayAlgors();
        // algors.createInstanceCollections(param, examples);
    }
}
