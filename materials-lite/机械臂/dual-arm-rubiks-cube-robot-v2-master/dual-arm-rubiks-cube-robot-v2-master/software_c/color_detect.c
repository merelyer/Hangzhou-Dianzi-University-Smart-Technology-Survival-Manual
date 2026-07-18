/*
程序流程：
1、基于透视变换，划分18个色块的大致区域
2、筛选每个区域内的所有像素，转换为hsv，使用二维直方图方式提取主色调
*/
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "color_detect.h"

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
// static int calculate_quad_area(Point quad[4]) {
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
            perror("Error: Singular matrix in Gaussian elimination.\n");
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
static void rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b, uint8_t *h, uint8_t *s) {
    uint8_t max = r > g ? (r > b ? r : b) : (g > b ? g : b);
    uint8_t min = r < g ? (r < b ? r : b) : (g < b ? g : b);

    // 移除v分量的计算
    //*v = max;
    
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

// 提取四边形内的像素并计算主色调
// 当 avg_mode = false时计算主色调
// 当 avg_mode = true时计算平均色调
void extract_dominant_color_from_quad(Point quad[4], const uint8_t *img, int width, int height, 
                                     uint8_t *h, uint8_t *s, uint8_t *dominant_percentage, 
                                    bool avg_mode) {
    const int H_BINS = 60;
    const int S_BINS = 8;
    int hist[H_BINS][S_BINS] = {0};
    int total_pixels = 0;
    unsigned int sum_r = 0;
    unsigned int sum_g = 0;
    unsigned int sum_b = 0;
    // 寻找四边形边界
    int min_x = width, max_x = 0;
    int min_y = height, max_y = 0;
    
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
    
    // 扫描线填充算法
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
        
        // 填充交点之间的像素
        for (int i = 0; i < intersect_count; i += 2) {
            int start_x = x_intersects[i];
            int end_x = x_intersects[i + 1];
            
            if (start_x < 0) start_x = 0;
            if (end_x >= width) end_x = width - 1;
            
            for (int x = start_x; x <= end_x; x++) {
                const uint8_t *pixel = &img[3 * (y * width + x)];
                if(avg_mode){
                    // 对h求平均计算较为复杂，因此对rgb求平均值
                    sum_r += pixel[2];
                    sum_g += pixel[1];
                    sum_b += pixel[0];
                }else{
                    uint8_t h_val, s_val;
                    // 转换为HSV
                    rgb_to_hsv(pixel[2], pixel[1], pixel[0], &h_val, &s_val);
                    // 映射到直方图bin
                    int h_bin = (h_val * H_BINS) / 180;
                    int s_bin = (s_val * S_BINS) / 256;
            
                    if (h_bin >= H_BINS) h_bin = H_BINS - 1;
                    if (s_bin >= S_BINS) s_bin = S_BINS - 1;
            
                    // 更新直方图
                    hist[h_bin][s_bin]++;
                }
                total_pixels++;
            }
        }
    }
    
    // 如果没有像素，返回默认值
    if (total_pixels == 0) {
        *h = 0;
        *s = 0;
        *dominant_percentage = 0;
        return;
    }
    if(avg_mode){
        // 求区域的平均颜色
        sum_r /= total_pixels;
        sum_g /= total_pixels;
        sum_b /= total_pixels;
        *dominant_percentage = 100;
        rgb_to_hsv(sum_r, sum_g, sum_b, h, s);
    }else{
        // 提取前4个最大值的bin
        int top_values[4] = {0};
        int top_h[4] = {0};
        int top_s[4] = {0};
        int found = 0;
        int cumulative = 0;
        
        for (int k = 0; k < 4; k++) {
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
            
            if (max_val == 0) break;
            
            top_values[found] = max_val;
            top_h[found] = max_h;
            top_s[found] = max_s;
            cumulative += max_val;
            found++;
            
            // 如果累计占比已超过33%，提前退出（如果取50%，摄像头歪斜测试用例用例4会fail）
            if (cumulative >= total_pixels / 3) {
                break;
            }
            
            // 将最大值位置清零，避免重复选择
            hist[max_h][max_s] = 0;
        }
        
        // 计算主色占比
        *dominant_percentage = cumulative * 100 / total_pixels;
        
        // 计算加权平均
        int total_weight = 0;
        int weighted_h = 0;
        int weighted_s = 0;
        
        for (int i = 0; i < found; i++) {
            total_weight += top_values[i];
            weighted_h += top_h[i] * top_values[i];
            weighted_s += top_s[i] * top_values[i];
        }
        
        if (total_weight > 0) {
            *h = (uint8_t)((weighted_h * 180 + total_weight * H_BINS / 2) / (total_weight * H_BINS));
            *s = (uint8_t)((weighted_s * 256 + total_weight * S_BINS / 2) / (total_weight * S_BINS));
        } else {
            *h = 0;
            *s = 0;
        }
    }
}

// BMP读取
uint8_t* load_bmp_image(const char* filename, int* width, int* height) {
    FILE *bmp_file = fopen(filename, "rb");
    if (!bmp_file) {
        perror("Error opening image");
        return NULL;
    }
    
    // 读取BMP头
    uint8_t header[54];
    if (fread(header, 1, 54, bmp_file) != 54) {
        perror("Error reading header");
        fclose(bmp_file);
        return NULL;
    }
    
    // 提取图像参数
    *width = *(int32_t *)&header[18];
    *height = *(int32_t *)&header[22];
    uint16_t bpp = *(uint16_t *)&header[28];
    
    if (bpp != 24) {
        fprintf(stderr, "Only 24-bit BMP supported\n");
        fclose(bmp_file);
        return NULL;
    }
    
    // 计算行填充（每行字节数必须是4的倍数）
    int32_t row_size = (*width * 3 + 3) & ~3;
    int32_t image_size = row_size * *height;
    
    // 分配内存
    uint8_t *bmp_data = (uint8_t *)malloc(image_size);
    if (!bmp_data) {
        perror("Memory allocation failed");
        fclose(bmp_file);
        return NULL;
    }
    
    // 读取图像数据（从下到上）
    fseek(bmp_file, 54, SEEK_SET);
    for (int y = *height - 1; y >= 0; y--) {
        if (fread(bmp_data + y * row_size, 1, row_size, bmp_file) != (size_t)row_size) {
            perror("Error reading image data");
            free(bmp_data);
            fclose(bmp_file);
            return NULL;
        }
    }
    fclose(bmp_file);
    
    // 转换为RGB格式（从BGR）
    uint8_t *bgr_image = (uint8_t *)malloc(*width * *height * 3);
    if (!bgr_image) {
        perror("Memory allocation failed");
        free(bmp_data);
        return NULL;
    }
    
    for (int y = 0; y < *height; y++) {
        uint8_t *src = bmp_data + y * row_size;
        for (int x = 0; x < *width; x++) {
            // 像素顺序是BGR
            bgr_image[3 * (y * *width + x) + 0] = *src++;
            bgr_image[3 * (y * *width + x) + 1] = *src++;
            bgr_image[3 * (y * *width + x) + 2] = *src++;
        }
    }
    free(bmp_data);
    
    return bgr_image;
}

void print_2x3x3_quads(const Point output_quads[18][4])
{
    for (int i = 0; i < 18; i++)
    {
        // 输出顶点坐标（按顺序）
        printf("%d %d\n", output_quads[i][0].x, output_quads[i][0].y);
        printf("%d %d\n", output_quads[i][1].x, output_quads[i][1].y);
        printf("%d %d\n", output_quads[i][2].x, output_quads[i][2].y);
        printf("%d %d\n", output_quads[i][3].x, output_quads[i][3].y);
        printf("\n"); // 小四边形之间用空行分隔
    }
}
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
// 修复聚类数量有时候不是每组9个的问题
int color_detect(const uint8_t hsv_color[54][2], char cube_str[55])
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
        printf("Error: Can not find white center\n");
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
    
    return errors;
}