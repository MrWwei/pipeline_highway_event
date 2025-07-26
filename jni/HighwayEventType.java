package cn.xtkj.jni.algor.helper;

public enum HighwayEventType {
    PARKING_LANE(1,"parking-lane","行车道停车"),
    PARKING_EMERGENCY_LANE(2,"parking-emergency-lane ","应急车道停车"),
    OCCUPY_EMERGENCY_LANE(3,"occupy-emergency-lane","占用应急车道行驶"),
    WALK_HIGHWAY(4,"walk-highway","高速路有行人"),
    HIGHWAY_JAM(5,"highway-jam","高速路拥堵"),
    TRAFFIC_ACCIDENT(6,"traffic-accident","交通事故");

    private int eventTypeId;//事件类别ID
    private String eventTypeCode;//事件类别编码
    private String eventTypeDesc;//事件类别描述

    HighwayEventType() {
    }

    HighwayEventType(int eventTypeId, String eventTypeCode, String eventTypeDesc) {
        this.eventTypeId = eventTypeId;
        this.eventTypeCode = eventTypeCode;
        this.eventTypeDesc = eventTypeDesc;
    }

    public static HighwayEventType parseByEventTypeId(int eventTypeId){
        for(HighwayEventType type:HighwayEventType.values()){
            if(type.getEventTypeId()==eventTypeId){
                return type;
            }
        }
        return null;
    }

    public static HighwayEventType parseByEventTypeId(String eventTypeCode){
        for(HighwayEventType type:HighwayEventType.values()){
            if(type.getEventTypeCode().equalsIgnoreCase(eventTypeCode)){
                return type;
            }
        }
        return null;
    }


    public int getEventTypeId() {
        return eventTypeId;
    }

    public void setEventTypeId(int eventTypeId) {
        this.eventTypeId = eventTypeId;
    }

    public String getEventTypeCode() {
        return eventTypeCode;
    }

    public void setEventTypeCode(String eventTypeCode) {
        this.eventTypeCode = eventTypeCode;
    }

    public String getEventTypeDesc() {
        return eventTypeDesc;
    }

    public void setEventTypeDesc(String eventTypeDesc) {
        this.eventTypeDesc = eventTypeDesc;
    }
}
