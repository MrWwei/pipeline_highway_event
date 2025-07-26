#ifndef EVENT_TYPE_H
#define EVENT_TYPE_H
// 目标状态枚举
enum class ObjectStatus {
  UNKNOWN = -1,            // 未知状态
  NORMAL = 0,                 // 正常状态
  PARKING_LANE = 1,           // 违停
  PARKING_EMERGENCY_LANE = 2, // 应急车道停车
  OCCUPY_EMERGENCY_LANE = 3,  // 占用应急车道
  WALK_HIGHWAY = 4,           // 高速行人
  HIGHWAY_JAM = 5,            // 高速拥堵
  TRAFFIC_ACCIDENT = 6        // 交通事故
};
#endif // EVENT_TYPE_H