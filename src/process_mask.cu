#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cstdio>

#define CUDA_CHECK(err)                                                                    \
    do                                                                                     \
    {                                                                                      \
        cudaError_t e = (err);                                                             \
        if (e != cudaSuccess)                                                              \
        {                                                                                  \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

/* ================= CUDA kernels ================= */

/* 第一次扫描：4-邻域等价标记 */
__global__ void first_scan(const uint8_t *mask,
                           int *labels,
                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;
    if (mask[idx] == 0)
    {
        labels[idx] = 0;
        return;
    }

    /* 上、左根标签 */
    int up = (y > 0) ? labels[idx - width] : 0;
    int left = (x > 0) ? labels[idx - 1] : 0;

    if (up == 0 && left == 0)
        labels[idx] = idx + 1; // 新标签
    else if (up == 0)
        labels[idx] = left;
    else if (left == 0)
        labels[idx] = up;
    else
        labels[idx] = min(up, left); // 合并
}

/* 第二次扫描：路径压缩 */
__global__ void second_scan(int *labels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int lab = labels[idx];
    if (lab == 0)
        return;

    while (labels[lab - 1] != lab)
        lab = labels[lab - 1];
    labels[idx] = lab;
}

/* 只保留最大域 */
__global__ void keep_max(const int *labels,
                         uint8_t *dst,
                         int max_label,
                         int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    dst[idx] = (labels[idx] == max_label) ? 0 : 1;
}
/* 种子填充：迭代扫描，8 邻域 */
__global__ void flood_fill_step(uint8_t *mask, int w, int h, bool *changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    const int idx = y * w + x;
    if (mask[idx] != 0)
        return; // 已经是前景跳过

    /* 8 邻域只要有一个背景点 → 仍是背景 */
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                continue;
            if (mask[ny * w + nx] == 0)
                return;
        }
    /* 所有邻域都是前景 → 该点置前景 */
    mask[idx] = 1;
    *changed = true;
}
/* ================= 主接口 ================= */
cv::Mat remove_small_white_regions_cuda1(const cv::Mat &src)
{
    CV_Assert(src.type() == CV_8UC1);
    const int w = src.cols;
    const int h = src.rows;
    const int n = w * h;

    /* GPU 上传 */
    cv::cuda::GpuMat d_src, d_tmp, d_labels;
    d_src.upload(src);
    d_src.copyTo(d_tmp); // 工作副本，用于孔洞填充
    d_labels.create(h, w, CV_32SC1);

    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    /* ---------- 1. 填充孔洞 ---------- */
    bool *d_changed;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(bool)));
    do
    {
        bool h_changed = false;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
        flood_fill_step<<<grid, block>>>(d_tmp.ptr<uint8_t>(), w, h, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        if (!h_changed)
            break;
    } while (true);
    CUDA_CHECK(cudaFree(d_changed));

    /* ---------- 2. 连通域标记 ---------- */
    first_scan<<<grid, block>>>(d_tmp.ptr<uint8_t>(),
                                reinterpret_cast<int *>(d_labels.ptr<uint8_t>()),
                                w, h);
    CUDA_CHECK(cudaDeviceSynchronize());
    second_scan<<<grid, block>>>(reinterpret_cast<int *>(d_labels.ptr<uint8_t>()),
                                 w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---------- 3. 统计最大域 ---------- */
    thrust::device_ptr<int> d_lab(reinterpret_cast<int *>(d_labels.ptr<uint8_t>()));
    thrust::device_vector<int> labs(d_lab, d_lab + n);
    thrust::sort(labs.begin(), labs.end());

    thrust::device_vector<int> keys(n);
    thrust::device_vector<int> areas(n);
    auto new_end = thrust::reduce_by_key(labs.begin(), labs.end(),
                                         thrust::constant_iterator<int>(1),
                                         keys.begin(),
                                         areas.begin());

    int num_labels = new_end.second - areas.begin();
    auto max_it = thrust::max_element(areas.begin(), areas.begin() + num_labels);
    int max_label = keys[max_it - areas.begin()];

    /* ---------- 4. 只保留最大域 ---------- */
    keep_max<<<grid, block>>>(reinterpret_cast<int *>(d_labels.ptr<uint8_t>()),
                              d_tmp.ptr<uint8_t>(),
                              max_label,
                              w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Mat dst;
    d_tmp.download(dst);
    return dst;
}