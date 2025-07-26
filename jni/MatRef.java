package cn.xtkj.jni.algor.data;

import java.io.Serializable;

/**
 * @author htchen
 * @version 1.0
 * @ClassName: MatRef
 * @date 2022年10月29日 10:28:16
 */
public class MatRef implements Serializable {
    private long matDataAddr;//底层数据地址
    private int matCols;
    private int matRows;
    private long nativeObjAddr;//底层对象地址

    public MatRef() {
    }

    public MatRef(long matDataAddr, int matCols, int matRows, long nativeObjAddr) {
        this.matDataAddr = matDataAddr;
        this.matCols = matCols;
        this.matRows = matRows;
        this.nativeObjAddr = nativeObjAddr;
    }

    public long getMatDataAddr() {
        return matDataAddr;
    }

    public void setMatDataAddr(long matDataAddr) {
        this.matDataAddr = matDataAddr;
    }

    public int getMatCols() {
        return matCols;
    }

    public void setMatCols(int matCols) {
        this.matCols = matCols;
    }

    public int getMatRows() {
        return matRows;
    }

    public void setMatRows(int matRows) {
        this.matRows = matRows;
    }

    public long getNativeObjAddr() {
        return nativeObjAddr;
    }

    public void setNativeObjAddr(long nativeObjAddr) {
        this.nativeObjAddr = nativeObjAddr;
    }
}