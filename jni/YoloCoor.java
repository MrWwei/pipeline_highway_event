package cn.xtkj.jni.algor.helper;

import java.io.Serializable;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: YoloCoor 用于接收目标识别算法和分类算法的输出
 * @date 2022年08月24日 16:44:07
 */
public class YoloCoor implements Serializable {
    /**
     * 检测目标区域的宽度
     */
    private Integer coorWidthPx;

    /**
     * 检测目标区域的高度
     */
    private Integer coorHeightPx;

    /**
     * 左上角的左边距像素
     */
    private Integer coorNorthwestLeftPx;

    /**
     * 左上角的上边距像素
     */
    private Integer coorNorthwestTopPx;

    /**
     * 右下角的左边距像素
     */
    private Integer coorSoutheastLeftPx;

    /**
     * 右下角的上边距像素
     */
    private Integer coorSoutheastTopPx;

    /**
     * 可信度
     */
    private Integer reliability;
    /**
     * 类别
     */
    private Integer type;

    private Integer trackId;//目标跟踪id

    public Integer getCoorWidthPx() {
        if(coorWidthPx==null && coorSoutheastLeftPx!=null && coorNorthwestLeftPx!=null){
            coorWidthPx=coorSoutheastLeftPx-coorNorthwestLeftPx;
        }
        return coorWidthPx;
    }

    public void setCoorWidthPx(Integer coorWidthPx) {
        this.coorWidthPx = coorWidthPx;
    }

    public Integer getCoorHeightPx() {
        if(coorHeightPx==null  && coorSoutheastTopPx!=null && coorNorthwestTopPx!=null){
            coorHeightPx=coorSoutheastTopPx-coorNorthwestTopPx;
        }
        return coorHeightPx;
    }

    public void setCoorHeightPx(Integer coorHeightPx) {
        this.coorHeightPx = coorHeightPx;
    }

    public Integer getCoorNorthwestLeftPx() {
        return coorNorthwestLeftPx;
    }

    public void setCoorNorthwestLeftPx(Integer coorNorthwestLeftPx) {
        this.coorNorthwestLeftPx = coorNorthwestLeftPx;
    }

    public Integer getCoorNorthwestTopPx() {
        return coorNorthwestTopPx;
    }

    public void setCoorNorthwestTopPx(Integer coorNorthwestTopPx) {
        this.coorNorthwestTopPx = coorNorthwestTopPx;
    }

    public Integer getCoorSoutheastLeftPx() {
        return coorSoutheastLeftPx;
    }

    public void setCoorSoutheastLeftPx(Integer coorSoutheastLeftPx) {
        this.coorSoutheastLeftPx = coorSoutheastLeftPx;
    }

    public Integer getCoorSoutheastTopPx() {
        return coorSoutheastTopPx;
    }

    public void setCoorSoutheastTopPx(Integer coorSoutheastTopPx) {
        this.coorSoutheastTopPx = coorSoutheastTopPx;
    }

    public Integer getReliability() {
        return reliability;
    }

    public void setReliability(Integer reliability) {
        this.reliability = reliability;
    }

    public Integer getType() {
        return type;
    }

    public void setType(Integer type) {
        this.type = type;
    }

    public Integer getTrackId() {
        return trackId;
    }

    public void setTrackId(Integer trackId) {
        this.trackId = trackId;
    }
}