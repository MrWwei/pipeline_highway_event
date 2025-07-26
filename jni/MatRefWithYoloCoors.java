package cn.xtkj.jni.algor.data;

import cn.xtkj.jni.algor.helper.YoloCoor;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: MatRefWithYoloCoors
 * @date 2023年09月21日 15:22:32
 */
public class MatRefWithYoloCoors extends MatRef {
private YoloCoor[] subCoors;//一个mat中识别到的多个目标

    public YoloCoor[] getSubCoors() {
        return subCoors;
    }

    public void setSubCoors(YoloCoor[] subCoors) {
        this.subCoors = subCoors;
    }
}