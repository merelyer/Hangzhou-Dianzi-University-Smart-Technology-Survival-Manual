/*
程序流程：
1、基于透视变换，划分18个色块的大致区域
2、筛选每个区域内的所有像素，转换为hsv，使用二维直方图方式提取主色调
*/
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>
#include "color_detect.h"
#include "esp_log.h"
static const char *TAG = "color_detect";

// 高斯消元法求解8x8线性方程组
static int gaussian_elimination(int n, float A[][8], float b[], float x[]) {
    for (int k = 0; k < n; k++) {
        // 部分选主元
        int p = k;
        float max_val = fabsf(A[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (fabsf(A[i][k]) > max_val) {
                max_val = fabsf(A[i][k]);
                p = i;
            }
        }
        // 检查主元是否为零
        if (fabsf(max_val) < 1e-10f) {
            return -1; // 矩阵奇异
        }
        // 交换行
        if (p != k) {
            for (int j = k; j < n; j++) {
                float temp = A[k][j];
                A[k][j] = A[p][j];
                A[p][j] = temp;
            }
            float temp_b = b[k];
            b[k] = b[p];
            b[p] = temp_b;
        }
        // 消元
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k] / A[k][k];
            for (int j = k; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    // 回代
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    return 0;
}

// 计算四边形面积（鞋带公式）
// int calculate_quad_area(Point quad[4]) {
//     int area = 0;
//     for (int i = 0; i < 4; i++) {
//         int j = (i + 1) % 4;
//         area += quad[i].x * quad[j].y - quad[j].x * quad[i].y;
//     }
//     return abs(area) / 2;
// }

// 区域分割函数，根据透视变换，将2个大四边形拆分为18个小四边形
int get_2x3x3_quads(const Point point_xy[6], Point output_quads[18][4])
{
    // 定义两个四边形
    Point quads[2][4] = {
        {
            // 四边形1: {0,1,4,3}
            {point_xy[0].x, point_xy[0].y}, // P0 (左下)
            {point_xy[1].x, point_xy[1].y}, // P1 (右下)
            {point_xy[4].x, point_xy[4].y}, // P4 (右上)
            {point_xy[3].x, point_xy[3].y}  // P3 (左上)
        },
        {
            // 四边形2: {1,2,5,4}
            {point_xy[1].x, point_xy[1].y}, // P1 (左下)
            {point_xy[2].x, point_xy[2].y}, // P2 (右下)
            {point_xy[5].x, point_xy[5].y}, // P5 (右上)
            {point_xy[4].x, point_xy[4].y}  // P4 (左上)
        }};
    const int divisions = 3;             // 每条边三等分
    const int grid_size = divisions + 1; // 网格大小：4x4

    // 处理两个四边形
    for (int quad_index = 0; quad_index < 2; quad_index++)
    {
        // 获取当前四边形的四个角点
        Point A = quads[quad_index][0];
        Point B = quads[quad_index][1];
        Point C = quads[quad_index][2];
        Point D = quads[quad_index][3];

        // 构建8x8系数矩阵和右侧向量
        float coefA[8][8] = {0};
        float b[8] = {0};

        // 点A对应目标(0,0)
        // 方程1: h00 * 0 + h01 * 0 + h02 = A.x
        coefA[0][2] = 1;
        b[0] = A.x;
        // 方程2: h10 * 0 + h11 * 0 + h12 = A.y
        coefA[1][5] = 1;
        b[1] = A.y;

        // 点B对应目标(1,0)
        // 方程3: h00 * 1 + h01 * 0 + h02 - h20 * 1*B.x - h21 * 0*B.x = B.x
        coefA[2][0] = 1;
        coefA[2][2] = 1;
        coefA[2][6] = -B.x;
        b[2] = B.x;
        // 方程4: h10 * 1 + h11 * 0 + h12 - h20 * 1*B.y - h21 * 0*B.y = B.y
        coefA[3][3] = 1;
        coefA[3][5] = 1;
        coefA[3][6] = -B.y;
        b[3] = B.y;

        // 点C对应目标(1,1)
        // 方程5: h00 * 1 + h01 * 1 + h02 - h20 * 1*C.x - h21 * 1*C.x = C.x
        coefA[4][0] = 1;
        coefA[4][1] = 1;
        coefA[4][2] = 1;
        coefA[4][6] = -C.x;
        coefA[4][7] = -C.x;
        b[4] = C.x;
        // 方程6: h10 * 1 + h11 * 1 + h12 - h20 * 1*C.y - h21 * 1*C.y = C.y
        coefA[5][3] = 1;
        coefA[5][4] = 1;
        coefA[5][5] = 1;
        coefA[5][6] = -C.y;
        coefA[5][7] = -C.y;
        b[5] = C.y;

        // 点D对应目标(0,1)
        // 方程7: h00 * 0 + h01 * 1 + h02 - h20 * 0*D.x - h21 * 1*D.x = D.x
        coefA[6][1] = 1;
        coefA[6][2] = 1;
        coefA[6][7] = -D.x;
        b[6] = D.x;
        // 方程8: h10 * 0 + h11 * 1 + h12 - h20 * 0*D.y - h21 * 1*D.y = D.y
        coefA[7][4] = 1;
        coefA[7][5] = 1;
        coefA[7][7] = -D.y;
        b[7] = D.y;

        // 解方程组获取变换参数
        float H[8];
        if (gaussian_elimination(8, coefA, b, H) != 0)
        {
            ESP_LOGE(TAG, "Error: Singular matrix in Gaussian elimination.");
            return -1;
        }

        // 创建4x4的网格点
        Point grid[grid_size][grid_size];
        for (int i = 0; i < grid_size; i++)
        {
            float t = (float)i / divisions;
            for (int j = 0; j < grid_size; j++)
            {
                float s = (float)j / divisions;
                // 计算齐次坐标的权重
                float w = H[6] * s + H[7] * t + 1.0f;
                // 计算原图中的x坐标（+0.5用于四舍五入）
                float px = (H[0] * s + H[1] * t + H[2]) / w + 0.5f;
                float py = (H[3] * s + H[4] * t + H[5]) / w + 0.5f;
                grid[i][j] = (Point){(int)px, (int)py};
            }
        }

        // 输出9个小四边形
        for (int i = 0; i < divisions; i++)
        {
            for (int j = 0; j < divisions; j++)
            {
                // 小四边形的四个顶点
                int index = quad_index * 9 + i * 3 + j;
                output_quads[index][0] = grid[i][j];
                output_quads[index][1] = grid[i][j + 1];
                output_quads[index][2] = grid[i + 1][j + 1];
                output_quads[index][3] = grid[i + 1][j];
            }
        }
    }
    return 0;
}

// RGB转HSV转换（整数运算）
static void rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b, uint8_t *h, uint8_t *s, uint8_t *v) {
    uint8_t max = r > g ? (r > b ? r : b) : (g > b ? g : b);
    uint8_t min = r < g ? (r < b ? r : b) : (g < b ? g : b);

    *v = max;
    
    if (max == 0) {
        *s = 0;
        *h = 0;
        return;
    }
    
    // 计算饱和度
    *s = (uint8_t)(255 * (max - min) / max);
    
    if (max == min) {
        *h = 0;
        return;
    }
    
    int delta = max - min;
    int h_val;
    
    if (max == r) {
        h_val = 60 * (g - b) / delta;
    } else if (max == g) {
        h_val = 60 * (b - r) / delta + 120;
    } else {
        h_val = 60 * (r - g) / delta + 240;
    }
    
    if (h_val < 0) h_val += 360;
    *h = (uint8_t)(h_val / 2); // 转换为0-179范围
}
// HSV转RGB转换（整数运算）
static void hsv_to_rgb(uint8_t h, uint8_t s, uint8_t v, uint8_t *r, uint8_t *g, uint8_t *b) {
    // 灰色（无颜色）
    if (s == 0) {
        *r = v;
        *g = v;
        *b = v;
        return;
    }

    // 转换h回0-359范围
    int h_val = h * 2;
    int sector = h_val / 60;    // 0-5的分区
    int f = h_val % 60;         // 余数 (0-59)

    // 计算中间值
    int c = (v * s) / 255;      // 颜色分量
    int m = v - c;               // 亮度调整
    int p = (c * f) / 60;        // 过渡计算值
    int q = (c * (60 - f)) / 60; // 反方向过渡值

    // 根据扇区分量计算RGB
    switch (sector) {
        case 0:
            *r = c + m;
            *g = p + m;
            *b = m;
            break;
        case 1:
            *r = q + m;
            *g = c + m;
            *b = m;
            break;
        case 2:
            *r = m;
            *g = c + m;
            *b = p + m;
            break;
        case 3:
            *r = m;
            *g = q + m;
            *b = c + m;
            break;
        case 4:
            *r = p + m;
            *g = m;
            *b = c + m;
            break;
        default: // case 5:
            *r = c + m;
            *g = m;
            *b = q + m;
    }
}

uint8_t hs_distanse(uint8_t h1, uint8_t s1, uint8_t h2, uint8_t s2) {
    uint8_t r1, g1, b1;
    uint8_t r2, g2, b2;
    
    // 固定亮度为255转换到RGB
    hsv_to_rgb(h1, s1, 255, &r1, &g1, &b1);
    hsv_to_rgb(h2, s2, 255, &r2, &g2, &b2);
    
    // 计算曼哈顿距离并缩放至0-255
    int dr = abs(r1 - r2);
    int dg = abs(g1 - g2);
    int db = abs(b1 - b2);
    int dist = dr + dg + db;
    
    // 平均通道距离（返回0-255范围）
    return (uint8_t)(dist / 3);
}
// RGB565转RGB888
static inline void rgb565_to_rgb888(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b) 
{
    // 组合成16位像素值（大端序：color_0是高字节，color_1是低字节）
    // uint16_t pixel = ((uint16_t)color_0 << 8) | color_1;
    // 组合成16位像素值（小端序）
    // uint16_t pixel = ((uint16_t)color_1 << 8) | color_0;
    
    // 提取RGB分量
    uint8_t r5 = (pixel >> 11) & 0x1F;  // 高5位是红色
    uint8_t g6 = (pixel >> 5)  & 0x3F;  // 中间6位是绿色
    uint8_t b5 = pixel & 0x1F;          // 低5位是蓝色

    // 扩展到8位
    *b = b5 << 3;  // 5位->8位
    *g = g6 << 2;  // 6位->8位
    *r = r5 << 3;  // 5位->8位
}


#define H_BINS  60
#define S_BINS  8
bool hs_equal(uint8_t h1, uint8_t s1, uint8_t h2, uint8_t s2, int h_threshold_bin_width) {
    const int h_threshold = h_threshold_bin_width * ((180 + H_BINS/2) / H_BINS); 
    
    // 如果都是白色
    if (s1 <= 48 && s2 <= 48) {
        return true;
    }
    
    // 计算色相差值（考虑环状特性）
    int h_diff = abs((int)h1 - (int)h2);
    if (h_diff > 90) { // 如果差值大于90度，取其环状补数
        h_diff = 180 - h_diff;
    }
    
    // 判断条件：色相差 <= h_threshold_bin_width个色相BIN宽度
    return h_diff <= h_threshold;
}

// 提取四边形内的像素并计算主色调 (RGB565版本)
// 提取四边形内的像素并计算主色调 (RGB565版本)
void extract_dominant_color_from_quad(Point quad[4], const uint8_t *img, int width, int height, 
                                      uint8_t *h, uint8_t *s, uint8_t *v) {
    // 步骤1：第一次扫描 - 计算边界和直方图
    int min_x = width, max_x = 0;
    int min_y = height, max_y = 0;
    
    // 计算四边形边界
    for (int i = 0; i < 4; i++) {
        if (quad[i].x < min_x) min_x = quad[i].x;
        if (quad[i].x > max_x) max_x = quad[i].x;
        if (quad[i].y < min_y) min_y = quad[i].y;
        if (quad[i].y > max_y) max_y = quad[i].y;
    }
    
    // 确保边界在图像范围内
    if (min_x < 0) min_x = 0;
    if (max_x >= width) max_x = width - 1;
    if (min_y < 0) min_y = 0;
    if (max_y >= height) max_y = height - 1;
    
    int total_pixels = 0;
    int hist[H_BINS][S_BINS] = {0};  // HSV直方图

    // 第一次扫描：填充直方图
    for (int y = min_y; y <= max_y; y++) {
        // 计算当前扫描线与四边形的交点
        int x_intersects[4];
        int intersect_count = 0;
        
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            int y1 = quad[i].y, y2 = quad[j].y;
            
            if ((y1 <= y && y <= y2) || (y2 <= y && y <= y1)) {
                if (y1 == y2) {
                    x_intersects[intersect_count++] = quad[i].x;
                    x_intersects[intersect_count++] = quad[j].x;
                } else {
                    float x = quad[i].x + (float)(quad[j].x - quad[i].x) * (y - y1) / (y2 - y1);
                    x_intersects[intersect_count++] = (int)(x + 0.5f);
                }
            }
        }
        
        // 对交点排序
        for (int i = 0; i < intersect_count - 1; i++) {
            for (int j = i + 1; j < intersect_count; j++) {
                if (x_intersects[i] > x_intersects[j]) {
                    int temp = x_intersects[i];
                    x_intersects[i] = x_intersects[j];
                    x_intersects[j] = temp;
                }
            }
        }
        
        // 处理交点之间的像素
        for (int i = 0; i < intersect_count; i += 2) {
            int start_x = x_intersects[i];
            int end_x = x_intersects[i + 1];
            
            if (start_x < 0) start_x = 0;
            if (end_x >= width) end_x = width - 1;
            
            for (int x = start_x; x <= end_x; x++) {
                const uint16_t *pixel_ptr = (uint16_t *)img;
                uint16_t pixel = pixel_ptr[y * width + x];
                total_pixels++;
                
                // 转换为RGB888并更新直方图
                uint8_t r, g, b;
                rgb565_to_rgb888(pixel, &r, &g, &b);
                
                uint8_t h_val, s_val, v_val;
                rgb_to_hsv(r, g, b, &h_val, &s_val, &v_val);
                
                int h_bin = (h_val * H_BINS) / 180;
                int s_bin = (s_val * S_BINS) / 256;
                
                if (h_bin >= H_BINS) h_bin = H_BINS - 1;
                if (s_bin >= S_BINS) s_bin = S_BINS - 1;
                
                hist[h_bin][s_bin]++;
            }
        }
    }
    
    // 如果没有像素，返回默认值
    if (total_pixels == 0) {
        *h = 0; *s = 0; *v = 0;
        return;
    }
    
    // 提取直方图峰值
    int max_val = 0;
    int max_h = 0, max_s = 0;
    for (int i = 0; i < H_BINS; i++) {
        for (int j = 0; j < S_BINS; j++) {
            if (hist[i][j] > max_val) {
                max_val = hist[i][j];
                max_h = i;
                max_s = j;
            }
        }
    }
    
    // 计算主色调的HSV值
    const int h_bin_width = 180 / H_BINS;
    const int s_bin_width = 256 / S_BINS;
    uint8_t main_h = (max_h * h_bin_width) + (h_bin_width / 2);
    uint8_t main_s = (max_s * s_bin_width) + (s_bin_width / 2);
    
    // 步骤2：第二次扫描 - 计算平均颜色
    unsigned int sum_r = 0, sum_g = 0, sum_b = 0;
    unsigned int cnt = 0;
    
    for (int y = min_y; y <= max_y; y++) {
        // 重新计算扫描线交点 (与第一次扫描相同)
        int x_intersects[4];
        int intersect_count = 0;
        
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            int y1 = quad[i].y, y2 = quad[j].y;
            
            if ((y1 <= y && y <= y2) || (y2 <= y && y <= y1)) {
                if (y1 == y2) {
                    x_intersects[intersect_count++] = quad[i].x;
                    x_intersects[intersect_count++] = quad[j].x;
                } else {
                    float x = quad[i].x + (float)(quad[j].x - quad[i].x) * (y - y1) / (y2 - y1);
                    x_intersects[intersect_count++] = (int)(x + 0.5f);
                }
            }
        }
        
        // 对交点排序 (与第一次扫描相同)
        for (int i = 0; i < intersect_count - 1; i++) {
            for (int j = i + 1; j < intersect_count; j++) {
                if (x_intersects[i] > x_intersects[j]) {
                    int temp = x_intersects[i];
                    x_intersects[i] = x_intersects[j];
                    x_intersects[j] = temp;
                }
            }
        }
        
        // 处理交点之间的像素
        for (int i = 0; i < intersect_count; i += 2) {
            int start_x = x_intersects[i];
            int end_x = x_intersects[i + 1];
            
            if (start_x < 0) start_x = 0;
            if (end_x >= width) end_x = width - 1;
            
            for (int x = start_x; x <= end_x; x++) {
                const uint16_t *pixel_ptr = (uint16_t *)img;
                uint16_t pixel = pixel_ptr[y * width + x];
                
                // 转换为RGB888和HSV
                uint8_t r, g, b;
                rgb565_to_rgb888(pixel, &r, &g, &b);
                
                uint8_t h_val, s_val, v_val;
                rgb_to_hsv(r, g, b, &h_val, &s_val, &v_val);
                
                // 检查是否匹配主色调
                if (hs_equal(h_val, s_val, main_h, main_s, 2)) {
                    sum_r += r;
                    sum_g += g;
                    sum_b += b;
                    cnt++;
                }
            }
        }
    }
    
    // 计算平均颜色
    uint8_t avg_r, avg_g, avg_b;
    if (cnt > 0) {
        avg_r = sum_r / cnt;
        avg_g = sum_g / cnt;
        avg_b = sum_b / cnt;
    } else {
        avg_r = avg_g = avg_b = 0;
    }
    
    // 转换为HSV输出
    rgb_to_hsv(avg_r, avg_g, avg_b, h, s, v);
}

// void print_2x3x3_quads(const Point output_quads[18][4])
// {
//     for (int i = 0; i < 18; i++)
//     {
//         // 输出顶点坐标（按顺序）
//         printf("%d %d\n", output_quads[i][0].x, output_quads[i][0].y);
//         printf("%d %d\n", output_quads[i][1].x, output_quads[i][1].y);
//         printf("%d %d\n", output_quads[i][2].x, output_quads[i][2].y);
//         printf("%d %d\n", output_quads[i][3].x, output_quads[i][3].y);
//         printf("\n"); // 小四边形之间用空行分隔
//     }
// }
//                           2-----------2------------1
//                           | U1(0)   U2(1)   U3(2)  |
//                           |                        |
//                           3 U4(3)   U5(4)   U6(5)  1
//                           |                        |
//                           | U7(6)   U8(7)   U9(8)  |
//  2-----------3------------3-----------0------------0-----------1------------1------------2------------2
//  | L1(36)  L2(37)  L3(38) | F1(18)  F2(19)  F3(20) | R1(9)   R2(10)  R3(11) |  B1(45)  B2(46)  B3(47) |
//  |                        |                        |                        |                         |
// 11 L4(39)  L5(40)  L6(41) 9 F4(21)  F5(22)  F6(23) 8 R4(12)  R5(13)  R6(14) 10 B4(48)  B5(49)  B6(50) 11
//  |                        |                        |                        |                         |
//  | L7(42)  L8(43)  L9(44) | F7(24)  F8(25)  F9(26) | R7(15)  R8(16)  R9(17) |  B7(51)  B8(52)  B9(53) |
//  3-----------7------------5-----------4------------4-----------5------------7------------6------------3
//                           | D1(27)  D2(28)  D3(29) |
//                           |                        |
//                           7 D4(30)  D5(31)  D6(32) 5
//                           |                        |
//                           | D7(33)  D8(34)  D9(35) |
//                           6-----------6------------7

// 参考开源项目 https://gitee.com/hemn1990/double-armed-rubiks-cube-robot
// 已经修复：聚类数量有时候不是每组9个的问题
// 已经修复：部分魔方配色，红色和橙色的饱和度和色相很接近，只有亮度不同，会出现混淆问题

int color_detect(const uint8_t hsv_color[54][3], char cube_str[55], char white_green_pos[3])
{
    int index[54];
    // 寻找白色,挑选饱和度S最低的9个色块,将索引号存放在index[45] - index[53]
    for(int i=0; i<54; i++)
    {
        index[i] = i;
    }
    // 冒泡排序,只排最低的9个
    for(int i=0; i<9; i++)
    {
        for(int j=0; j<54-i-1;j++)
        {
            if(hsv_color[index[j]][1] < hsv_color[index[j+1]][1])
            {
                int tmp = index[j];
                index[j] = index[j+1];
                index[j+1] = tmp;                
            }
        }
    }
    // 检测白色中心块
    int white_center_index = -1;
    for(int i=45; i<54; i++)
    {
        if(index[i] % 9 == 4)
        {
            white_center_index = index[i];
            break;
        }
    }
    if(white_center_index == -1)
    {
        ESP_LOGE(TAG, "Error: Can not find white center\n");
        return 1;
    }
    
    // 获取5个非白色中心块索引
    int center_indices[5];
    int center_count = 0;
    for(int i=0; i<6; i++)
    {
        int idx = i*9+4;
        if(idx != white_center_index)
            center_indices[center_count++] = idx;
    }
    
    // 初始化聚类中心
    float h_center[5] = {0};
    for(int i=0; i<5; i++)
    {
        h_center[i] = hsv_color[center_indices[i]][0] * 2.0f;
    }
    
    // 存储每个聚类的点
    int groups[5][45] = {{0}};
    int group_sizes[5] = {0};
    const int MAX_LOOP = 10;
    
    for(int loop=0; loop<MAX_LOOP; loop++)
    {
        // 重置聚类
        memset(group_sizes, 0, sizeof(group_sizes));
        
        // 分配点到最近的聚类 (排除白色块)
        for(int i=0; i<45; i++)
        {
            int point_idx = index[i];
            float h_val = hsv_color[point_idx][0] * 2.0f;
            
            float min_dist = FLT_MAX;
            int best_group = -1;
            
            for(int g=0; g<5; g++)
            {
                // 要计算这两个角度之间的绝对值差，首先需要处理两个角度间的差值使其落在0到360度之间，
                // 因为简单地做差可能得出的结果会超过这个范围，例如当h0 = 350而h = 10时，它们的差应该是20度而不是340度。
                float diff = fabsf(h_val - h_center[g]);
                float dist = diff > 180.0f ? 360.0f - diff : diff;
                
                if(dist < min_dist)
                {
                    min_dist = dist;
                    best_group = g;
                }
            }
            
            groups[best_group][group_sizes[best_group]++] = point_idx;
        }
        
        // 调整聚类大小确保每组恰好9个点
        int free_points[45] = {0};
        int free_count = 0;
        
        // 移出多余的点 (保护中心块)
        for(int g=0; g<5; g++)
        {
            if(group_sizes[g] <= 9) continue;
            
            // 计算每个点到中心的距离
            float distances[45];
            for(int i=0; i<group_sizes[g]; i++)
            {
                int idx = groups[g][i];
                float h_val = hsv_color[idx][0] * 2.0f;
                float diff = fabsf(h_val - h_center[g]);
                distances[i] = diff > 180.0f ? 360.0f - diff : diff;
            }
            
            // 移出最远的非中心点
            int remove_count = group_sizes[g] - 9;
            while(remove_count-- > 0)
            {
                int farthest_idx = -1;
                float max_dist = -1;
                
                // 寻找最远的非中心点
                for(int i=0; i<group_sizes[g]; i++)
                {
                    if(groups[g][i] == center_indices[g]) continue; // 保护中心块
                    
                    if(distances[i] > max_dist)
                    {
                        max_dist = distances[i];
                        farthest_idx = i;
                    }
                }
                
                if(farthest_idx == -1) break; // 没有可移出的点
                
                // 移出点
                free_points[free_count++] = groups[g][farthest_idx];
                
                // 从组中移除
                for(int j=farthest_idx; j<group_sizes[g]-1; j++)
                {
                    groups[g][j] = groups[g][j+1];
                    distances[j] = distances[j+1];
            }
                group_sizes[g]--;
            }
        }
        
        // 分配自由点到不足的组
        for(int i=0; i<free_count; i++)
        {
            int point_idx = free_points[i];
            float h_val = hsv_color[point_idx][0] * 2.0f;
            
            float min_dist = FLT_MAX;
            int best_group = -1;
            
            // 寻找最近的未满聚类
            for(int g=0; g<5; g++)
            {
                if(group_sizes[g] >= 9) continue;
                
                float diff = fabsf(h_val - h_center[g]);
                float dist = diff > 180.0f ? 360.0f - diff : diff;
                
                if(dist < min_dist)
                {
                    min_dist = dist;
                    best_group = g;
                }
            }
            
            if(best_group != -1)
            {
                groups[best_group][group_sizes[best_group]++] = point_idx;
            }
        }
        
        // 重新计算聚类中心
        int center_changed = 0;
        for(int g=0; g<5; g++)
        {
            float sum_cos = 0, sum_sin = 0;
            for(int i=0; i<group_sizes[g]; i++)
            {
                float angle = hsv_color[groups[g][i]][0] * (M_PI / 90.0f); // 转换为弧度
                sum_cos += cosf(angle);
                sum_sin += sinf(angle);
            }
            
            float new_center = atan2f(sum_sin, sum_cos) * (180.0f / M_PI);
            if(new_center < 0) new_center += 360.0f;
            
            if(fabsf(new_center - h_center[g]) > 1.0f)
            {
                center_changed = 1;
                h_center[g] = new_center;
            }
        }
        
        if(!center_changed) break; // 中心不再变化
    }

    // 红色和橙色区分逻辑
    float min_center_diff = 360.0f;
    int idx1 = -1, idx2 = -1;
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            float diff = fabsf(h_center[i] - h_center[j]);
            diff = diff > 180.0f ? 360.0f - diff : diff;
            if (diff < min_center_diff) {
                min_center_diff = diff;
                idx1 = i;
                idx2 = j;
            }
        }
    }

    ESP_LOGI(TAG, "最小色相差: %.2f (中心%d和中心%d)", min_center_diff, idx1, idx2);

    // 检查色相区分度
    // 色相区分度大的魔方通常大于11
    // 色相区分度小的魔方通常小于2
    if (min_center_diff < 6.0f) { 
        // printf("检测到色相近似的两个中心(可能是红色和橙色)\n");
        
        int v_values[18];
        int count = 0;
        for (int i = 0; i < group_sizes[idx1]; i++) {
            v_values[count++] = hsv_color[groups[idx1][i]][2];
        }
        for (int i = 0; i < group_sizes[idx2]; i++) {
            v_values[count++] = hsv_color[groups[idx2][i]][2];
        }
        
        // 按亮度排序
        for(int i=0; i<18; i++) {
            for(int j=0; j<18-i-1; j++) {
                if(v_values[j] > v_values[j+1]) {
                    int tmp = v_values[j];
                    v_values[j] = v_values[j+1];
                    v_values[j+1] = tmp;
                }
            }
        }
        
        // 计算前9个和后9个的统计信息，计算区分度，检查能否明确区分为两类
        float mean1 = 0, mean2 = 0;
        for (int i = 0; i < 9; i++) {
            mean1 += v_values[i];
        }
        mean1 /= 9;
        for (int i = 9; i < 18; i++) {
            mean2 += v_values[i];
        }
        mean2 /= 9;
        
        float std1 = 0, std2 = 0;
        for (int i = 0; i < 9; i++) {
            std1 += (v_values[i] - mean1) * (v_values[i] - mean1);
        }
        std1 = sqrtf(std1 / 9);
        for (int i = 9; i < 18; i++) {
            std2 += (v_values[i] - mean2) * (v_values[i] - mean2);
        }
        std2 = sqrtf(std2 / 9);
        
        // printf("低亮度组: 均值=%.2f, 标准差=%.2f\n", mean1, std1);
        // printf("高亮度组: 均值=%.2f, 标准差=%.2f\n", mean2, std2);
        // printf("均值差: %.2f, 标准差和: %.2f, 比值: %.2f\n", 
        //     fabsf(mean1 - mean2), std1 + std2, fabsf(mean1 - mean2) / (std1 + std2));
        
        if (fabsf(mean1 - mean2) > (std1 + std2) * 1.5f) {
            ESP_LOGI(TAG, "按亮度区分红色、橙色");
            
            // 合并两个组的点
            int combined[18];
            memcpy(combined, groups[idx1], group_sizes[idx1] * sizeof(int));
            memcpy(combined + 9, groups[idx2], group_sizes[idx2] * sizeof(int));
            
            // 按亮度排序
            for (int i = 0; i < 18; i++) {
                for (int j = 0; j < 18 - i - 1; j++) {
                    if (hsv_color[combined[j]][2] > hsv_color[combined[j + 1]][2]) {
                        int tmp = combined[j];
                        combined[j] = combined[j + 1];
                        combined[j + 1] = tmp;
                    }
                }
            }
            
            // 获取两个中心块的亮度值
            int v_center1 = hsv_color[center_indices[idx1]][2];
            int v_center2 = hsv_color[center_indices[idx2]][2];
            
            // 根据中心块的亮度决定分配顺序
            if (v_center1 < v_center2) {
                // 中心块1亮度低，所以组合并后亮度低的9个点应该分配给idx1
                memcpy(groups[idx1], combined, 9 * sizeof(int));
                memcpy(groups[idx2], combined + 9, 9 * sizeof(int));
            } else {
                // 中心块2亮度低，所以组合并后亮度低的9个点应该分配给idx2
                memcpy(groups[idx2], combined, 9 * sizeof(int));
                memcpy(groups[idx1], combined + 9, 9 * sizeof(int));
            }
        } else {
            // 亮度区分度不足，保持原分配
        }
    } else {
        // 色相差足够大，不需要亮度区分
    }

    // 构建魔方状态字符串
    const char URFDLB[6] = {'U','R','F','D','L','B'};
    
    // 初始化所有面为未识别('?')
    for(int i=0; i<54; i++)
    {
        cube_str[i] = '?';
    }
    cube_str[54] = '\0';
    
    // 设置白色面
    int white_face = white_center_index / 9;
    for(int i=45; i<54; i++)
    {
        cube_str[index[i]] = URFDLB[white_face];
    }
    
    // 设置其他面
    for(int g=0; g<5; g++)
            {
        int center_idx = center_indices[g];
        int face = center_idx / 9;
        
        for(int i=0; i<group_sizes[g]; i++)
            {
            int point_idx = groups[g][i];
            cube_str[point_idx] = URFDLB[face];
        }
    }
    
    // 验证结果
    int errors = 0;
    for(int i=0; i<54; i++)
    {
        if(cube_str[i] == '?')
            {
            printf("Unassigned point: %d\n", i);
            errors++;
        }
    }
    // 寻找绿色中心块（色相最接近120度的中心块）
    float min_green_diff = 360.0f;
    int green_face = -1;
    for (int g = 0; g < 5; g++) {
        // 计算与绿色（120度）的色相差（考虑环形特性）
        float diff = fabsf(h_center[g] - 120.0f);
        diff = diff > 180.0f ? 360.0f - diff : diff;
        
        if (diff < min_green_diff) {
            min_green_diff = diff;
            green_face = center_indices[g] / 9;
        }
    }

    // 赋值结果
    white_green_pos[0] = URFDLB[white_face];   // 白色中心块位置
    white_green_pos[1] = URFDLB[green_face];   // 绿色中心块位置
    white_green_pos[2] = '\0';
    return errors;
}