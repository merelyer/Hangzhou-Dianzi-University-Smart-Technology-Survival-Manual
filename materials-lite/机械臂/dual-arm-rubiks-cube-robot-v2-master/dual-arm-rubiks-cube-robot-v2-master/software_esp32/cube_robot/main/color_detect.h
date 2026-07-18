#pragma once

/* 二维坐标点结构体 */
typedef struct {
    int16_t x;  // 点的X坐标
    int16_t y;  // 点的Y坐标
} Point;

/**
 * @brief 将两个大四边形划分为18个小四边形
 * @param point_xy 输入点数组(6个点，定义两个大四边形)
 * @param output_quads 输出的小四边形顶点数组(18个四边形，每个含4个点)
 * @return 成功返回0，失败返回-1
 */
int get_2x3x3_quads(const Point point_xy[6], Point output_quads[18][4]);

/**
 * @brief 从四边形区域内提取主色调
 * @param quad 四边形区域的4个顶点
 * @param img 图像数据指针(RGB565格式)
 * @param width 图像宽度
 * @param height 图像高度
 * @param h 输出的主色调色相分量(0-179)
 * @param s 输出的主色调饱和度分量(0-255)
 * @param v 输出的主色调亮度分量(0-255)
 */
void extract_dominant_color_from_quad(Point quad[4], const uint8_t *img, int width, int height, 
    uint8_t *h, uint8_t *s, uint8_t *v);

/**
 * @brief 魔方颜色识别主函数
 * @param hsv_color 54个色块的HSV颜色数组([h,s]格式)
 * @param cube_str 输出的魔方状态字符串(54字符+结束符)
 * @param white_green_pos 指示白色中心块和绿色中心块对应的标志，用URFDLB表示
 * @return 未分配色块数量(0表示成功)
 */
int color_detect(const uint8_t hsv_color[54][3], char cube_str[55], char white_green_pos[3]);

/**
 * @brief 计算两个HSV颜色的距离
 * @return RGB空间的曼哈顿距离均值(0-255)
 */
uint8_t hs_distanse(uint8_t h1, uint8_t s1, uint8_t h2, uint8_t s2);

/**
 * @brief 判断两个HSV颜色是否相等
 * @return true=颜色相似视为相等，false=颜色差异较大
 */
bool hs_equal(uint8_t h1, uint8_t s1, uint8_t h2, uint8_t s2, int h_threshold_bin_width);

// int calculate_quad_area(Point quad[4]);