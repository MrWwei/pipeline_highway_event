package cn.xtkj.jni.algor.helper;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: EventYoloCoor
 * @date 2025年07月14日 17:25:13
 * 事件目标，继承自YoloCoor并添加事件类型
 */
public class EventYoloCoor extends YoloCoor {
    private int eventId;//事件ID，算法给出; 参考HighwayEventType枚举定义的eventTypeId
    private HighwayEventType eventType;//高速事件类型，引擎层解析

    public int getEventId() {
        return eventId;
    }

    public void setEventId(int eventId) {
        this.eventId = eventId;
    }

    public HighwayEventType getEventType() {
        return eventType;
    }

    public void setEventType(HighwayEventType eventType) {
        this.eventType = eventType;
    }
}