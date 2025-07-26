package cn.xtkj.jni.algor;

import cn.xtkj.jni.algor.data.MatRef;
import cn.xtkj.jni.algor.helper.EventYoloCoor;
import cn.xtkj.jni.util.LibLoader;

import java.util.*;
import java.util.concurrent.*;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: HighwayAlgors
 * @date 2025年07月14日 16:56:29
 */
public class HighwayAlgors{
    private static Set<HighwayExample> canUsedHighwayExample=new CopyOnWriteArraySet<>();
    private static Map<HighwayExample,ExecutorService> executorService=new ConcurrentHashMap<>();
    private static HighwayAlgors self;
    private static String version;
    /**
     * 获取jni,SDK版本号
     * @return
     */
    private native String getVersion();

    //初始化多个实例集，返回数组中每个元素是一个实例集ID(一个实例集在C++底层包含：一个机动车目标检测实例、一个行人目标检测实例、一个跟踪实例、一个分割实例)
    private native int[] createInstanceCollections(HighwayAlgorParam highwayAlgorParam,HighwayExample... highwayExample);

    //统一变更算法阈值参数 大于0表示变更成功
    private native int changeParam(HighwayAlgorParam highwayAlgorParam);

    //将数据推入算法层（未resize），返回这帧数据在算法层的数据id
    private native long putMat(int instanceCollectionId, MatRef matRefs);

    //从指定的实例集中获取某帧的推理结果
    private native EventYoloCoor[] takeRes(int instanceCollectionId,long algorsMatResourceId);

    //释放一个实例集 大于0表示为释放成功
    private native int releaseInstanceCollection(int instanceCollectionId);

    private HighwayAlgors(){}

    public static HighwayAlgors instance(String jniPath,HighwayAlgorParam highwayAlgorParam,HighwayExample... highwayExample){
        synchronized (HighwayAlgors.class){
            if(self==null){
                self=new HighwayAlgors();
                LibLoader.load(jniPath);
            }
        }
       int[] netIds=self.createInstanceCollections(highwayAlgorParam,highwayExample);
        if(netIds!=null && netIds.length==highwayExample.length){
            for(int x=0;x<netIds.length;x++){
                if(netIds[x]>0){
                    executorService.put(highwayExample[x],Executors.newFixedThreadPool(32));
                    highwayExample[x].setInstanceId(netIds[x]);
                    highwayExample[x].setExecutorService(Executors.newFixedThreadPool(1));
                    canUsedHighwayExample.add(highwayExample[x]);
                }
            }
        }
        return self;
    }

    public boolean flushParams(HighwayAlgorParam highwayAlgorParam){
        return changeParam(highwayAlgorParam)>0;
    }

    public EventYoloCoor[][] checkMats(MatRef[] matRefs, HighwayExample example){
        if(example!=null && example.isLoaded() && matRefs!=null && matRefs.length>0){
            ExecutorService service=executorService.get(example);
            if(service==null){
                return null;
            }
            List<AlgorResHold> algorResHolds=new LinkedList<>();
            Phaser phaser=new Phaser(matRefs.length);
            for(MatRef matRef:matRefs){
                long matId=putMat(example.getInstanceId(),matRef);
                AlgorResHold algorResHold=new AlgorResHold(matId);
                algorResHolds.add(algorResHold);
                service.submit(new Runnable() {
                    @Override
                    public void run() {
                        algorResHold.setAlgorRes(self.takeRes(example.getInstanceId(),algorResHold.getMatId()));
                        phaser.arrive();
                    }
                });
            }
            phaser.awaitAdvance(0);
            EventYoloCoor[][] yoloCoors=new EventYoloCoor[algorResHolds.size()][];
            for(int x=0;x<yoloCoors.length;x++){
                yoloCoors[x]=algorResHolds.get(x).getAlgorRes();
            }
            return yoloCoors;
        }
        return null;
    }

    public boolean releaseInstance(HighwayExample example){
        if(example.getInstanceId()>0){
            ExecutorService service=executorService.remove(example);
            if(service!=null){
                service.shutdown();
            }
            int x=releaseInstanceCollection(example.getInstanceId());
            if(x>0){
                example.setInstanceId(-1);
                return true;
            }
            return false;
        }
        return false;
    }

    public String getVersionName(){
        return getVersion();
    }

    public List<HighwayExample> getCanUsedNetExampleIds(){
        return new ArrayList<>(canUsedHighwayExample);
    }

    class AlgorResHold{
        private long matId;
        private EventYoloCoor[] algorRes;

        public AlgorResHold(long matId) {
            this.matId = matId;
        }

        public long getMatId() {
            return matId;
        }

        public void setMatId(long matId) {
            this.matId = matId;
        }

        public EventYoloCoor[] getAlgorRes() {
            return algorRes;
        }

        public void setAlgorRes(EventYoloCoor[] algorRes) {
            this.algorRes = algorRes;
        }
    }


}