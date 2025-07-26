#ifndef BOX_EVENT_H
#define BOX_EVENT_H
#include "event_type.h"
struct DetectionBox {
    int left, top, right, bottom;           // 边界框坐标
    float confidence;                       // 置信度
    int class_id;                          // 类别ID
    int track_id;                          // 跟踪ID
    ObjectStatus status;                   // 目标状态
    
    DetectionBox() : left(0), top(0), right(0), bottom(0), 
                    confidence(0.0f), class_id(0), track_id(0), 
                    status(ObjectStatus::UNKNOWN) {}
    DetectionBox(int l, int t, int r, int b, float conf, int cls = 0, int tid = 0, ObjectStatus stat = ObjectStatus::UNKNOWN)
        : left(l), top(t), right(r), bottom(b), confidence(conf),
          class_id(cls), track_id(tid), status(stat) {}
    bool is_valid() const {
        return left < right && top < bottom && confidence > 0.0f;
    }
};
#endif // BOX_H