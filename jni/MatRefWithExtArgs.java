package cn.xtkj.jni.algor.data;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: MatRefWithExtArgs
 * @date 2023年09月21日 16:05:56
 */
public class MatRefWithExtArgs extends MatRef {
    private int panelCount;//灯组中灯盘数量
    private int direction;//灯组内灯盘摆布 0:纵向，1:横向
    private int needKeelPoints;//分割算法使用，是否需要输出龙骨信息，大于0表示需要，否则不需要

    public int getPanelCount() {
        return panelCount;
    }

    public void setPanelCount(int panelCount) {
        this.panelCount = panelCount;
    }

    public int getDirection() {
        return direction;
    }

    public void setDirection(int direction) {
        this.direction = direction;
    }

    public int getNeedKeelPoints() {
        return needKeelPoints;
    }

    public void setNeedKeelPoints(int needKeelPoints) {
        this.needKeelPoints = needKeelPoints;
    }
}