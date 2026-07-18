#ifndef COLOR_DETECT_H
#define COLOR_DETECT_H

typedef struct {
    int16_t x;
    int16_t y;
} Point;

int get_2x3x3_quads(const Point point_xy[6], Point output_quads[18][4]);
void print_2x3x3_quads(const Point output_quads[18][4]);
void extract_dominant_color_from_quad(Point quad[4], const uint8_t *img, int width, int height, 
    uint8_t *h, uint8_t *s, uint8_t *dominant_percentage, bool avg_mode);
uint8_t* load_bmp_image(const char* filename, int* width, int* height);
int color_detect(const uint8_t hsv_color[54][2], char cube_str[55]);

#endif