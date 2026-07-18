#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>    
#include "solve.h"
#include "optimizer.h"
#include "color_detect.h"

#ifdef _WIN32
    #define API_EXPORT __declspec(dllexport)
#else
    #define API_EXPORT __attribute__((visibility("default")))
#endif

typedef enum route_enum {
    L=0, L3, L2, R, R3, R2, U, U3, U2, D, D3, D2, F, F3, F2, B, B3, B2
}route_t;

typedef struct cube_struct{
    uint8_t ep[12];
    uint8_t er[12];
    uint8_t cp[8];
    uint8_t cr[8];
} cube_t;

// 用于转换两种表示方式 20个棱块角块 <---> 54个面
// [UF, UR, UB, UL,DF, DR, DB, DL, FR, FL, BR, BL] [UFR, URB, UBL, ULF, DRF, DFL, DLB, DBR]
// UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
static const char edge_to_face[12][2] = {
    {7, 19},  {5, 10},  {1, 46},  {3, 37},
    {28, 25}, {32, 16}, {34, 52}, {30, 43},
    {23, 12}, {21, 41}, {48, 14}, {50, 39}
};
static const char corner_to_face[8][3] = {
    {8, 20, 9}, {2, 11, 45}, {0, 47, 36}, {6, 38, 18},
    {29, 15, 26}, {27, 24, 44}, {33, 42, 53}, {35, 51, 17}
};
static const char edge_index[24][2] = {
    "UF","UR","UB","UL","DF","DR","DB","DL","FR","FL","BR","BL",
    "FU","RU","BU","LU","FD","RD","BD","LD","RF","LF","RB","LB"
};
static const char corner_index[24][3] = {
    "UFR","URB","UBL","ULF","DRF","DFL","DLB","DBR",
    "FRU","RBU","BLU","LFU","RFD","FLD","LBD","BRD",
    "RUF","BUR","LUB","FUL","FDR","LDF","BDL","RDB"
};
// 54个面的表示方式，转换为棱块角块表示方式
// 正常 返回1 错误 返回0
static int cube_from_face_54(cube_t *c, const char *cube_str)
{
    int sum = 0;
    // 棱块
    for(int i=0; i<12; i++){
        int index_a = edge_to_face[i][0];
        int index_b = edge_to_face[i][1];
        int tmp;
        for(tmp=0;tmp<24;tmp++){
            if( edge_index[tmp][0] == cube_str[index_a] &&
                edge_index[tmp][1] == cube_str[index_b] ){
                break;
            }
        }
        if(tmp < 24){
            c -> ep[i] = tmp % 12;
            c -> er[i] = tmp / 12;
        }else{
            printf("Error: invalid edge %c%c.\n",
                   cube_str[index_a],cube_str[index_b]);
            return 0;
        }
    }
    for(int i=0; i<12; i++){
        for(int j=0; j<12; j++){
            if(i != j && c -> ep[i] == c -> ep[j]){
                printf("Error: repetition edge.\n");
                return 0;
            }
        }
    }
    sum = 0;
    for(int i=0; i<12; i++){
        sum += c -> er[i];
    }
    if(sum%2 != 0){
        // 单个棱块翻转，不影响后续程序运行
        // 测试用例BRBBUUFLFUFULRLDDBLBRDFULBRFRFFDBBLRUUURLRDDDRFLDBUDFL
        printf("Flipped edge, ignore.\n");
    }
    // 角块
    for(int i=0; i<8; i++){
        int index_a = corner_to_face[i][0];
        int index_b = corner_to_face[i][1];
        int index_c = corner_to_face[i][2];
        int tmp;
        for(tmp=0;tmp<24;tmp++){
            if( corner_index[tmp][0] == cube_str[index_a] &&
                corner_index[tmp][1] == cube_str[index_b] &&
                corner_index[tmp][2] == cube_str[index_c]){
                break;
            }
        }
        if(tmp < 24){
            c -> cp[i] = tmp % 8;
            c -> cr[i] = tmp / 8;
        }else{
            printf("Error: invalid corner %c%c%c.\n",
                   cube_str[index_a],cube_str[index_b],cube_str[index_c]);
            return 0;
        }
    }
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(i != j && c -> cp[i] == c -> cp[j]){
                printf("Error: repetition corner.\n");
                return 0;
            }
        }
    }
    sum = 0;
    for(int i=0; i<8; i++){
        sum += c -> cr[i];
    }
    if(sum%3 != 0){
        // 单个角块旋转，不影响后续程序运行
        printf("Twisted corner, ignore.\n");
    }
    // 校验棱块和角块之间的位置关系
    int cornerParity = 0;
    for (int i = 7; i >= 1; i--)
        for (int j = i - 1; j >= 0; j--)
            if (c->cp[j] > c->cp[i])
                cornerParity++;
    cornerParity = cornerParity % 2;
    int edgeParity = 0;
    for (int i = 11; i >= 1; i--)
        for (int j = i - 1; j >= 0; j--)
            if (c->ep[j] > c->ep[i])
                edgeParity++;
    edgeParity = edgeParity % 2;
    if ((edgeParity ^ cornerParity) != 0) {
        printf("ERROR: parity error.\n");
        return 0;
    }
    return 1;
}
// 用于验证正确性的测试代码
static const uint8_t route_tab_ep[18][12]={
    {0,1,2,11,4,5,6,9,8,3,10,7},// L  0
    {0,1,2,9,4,5,6,11,8,7,10,3},// L' 1
    {0,1,2,7,4,5,6,3,8,11,10,9},// L2 2
    {0,8,2,3,4,10,6,7,5,9,1,11},// R  3
    {0,10,2,3,4,8,6,7,1,9,5,11},// R' 4
    {0,5,2,3,4,1,6,7,10,9,8,11},// R2 5
    {1,2,3,0,4,5,6,7,8,9,10,11},// U  6
    {3,0,1,2,4,5,6,7,8,9,10,11},// U' 7
    {2,3,0,1,4,5,6,7,8,9,10,11},// U2 8
    {0,1,2,3,7,4,5,6,8,9,10,11},// D  9
    {0,1,2,3,5,6,7,4,8,9,10,11},// D' 10
    {0,1,2,3,6,7,4,5,8,9,10,11},// D2 11
    {9,1,2,3,8,5,6,7,0,4,10,11},// F  12
    {8,1,2,3,9,5,6,7,4,0,10,11},// F' 13
    {4,1,2,3,0,5,6,7,9,8,10,11},// F2 14
    {0,1,10,3,4,5,11,7,8,9,6,2},// B  15
    {0,1,11,3,4,5,10,7,8,9,2,6},// B' 16
    {0,1,6,3,4,5,2,7,8,9,11,10} // B2 17
};
static const uint8_t route_tab_er[18][12]={
    {0,0,0,0,0,0,0,0,0,0,0,0},// L  0
    {0,0,0,0,0,0,0,0,0,0,0,0},// L' 1
    {0,0,0,0,0,0,0,0,0,0,0,0},// L2 2
    {0,0,0,0,0,0,0,0,0,0,0,0},// R  3
    {0,0,0,0,0,0,0,0,0,0,0,0},// R' 4
    {0,0,0,0,0,0,0,0,0,0,0,0},// R2 5
    {0,0,0,0,0,0,0,0,0,0,0,0},// U  6
    {0,0,0,0,0,0,0,0,0,0,0,0},// U' 7
    {0,0,0,0,0,0,0,0,0,0,0,0},// U2 8
    {0,0,0,0,0,0,0,0,0,0,0,0},// D  9
    {0,0,0,0,0,0,0,0,0,0,0,0},// D' 10
    {0,0,0,0,0,0,0,0,0,0,0,0},// D2 11
    {1,0,0,0,1,0,0,0,1,1,0,0},// F  12
    {1,0,0,0,1,0,0,0,1,1,0,0},// F' 13
    {0,0,0,0,0,0,0,0,0,0,0,0},// F2 14
    {0,0,1,0,0,0,1,0,0,0,1,1},// B  15
    {0,0,1,0,0,0,1,0,0,0,1,1},// B' 16
    {0,0,0,0,0,0,0,0,0,0,0,0} // B2 17
};
static const uint8_t route_tab_cp[18][8]={
    {0,1,6,2,4,3,5,7},// L  0
    {0,1,3,5,4,6,2,7},// L' 1
    {0,1,5,6,4,2,3,7},// L2 2
    {4,0,2,3,7,5,6,1},// R  3
    {1,7,2,3,0,5,6,4},// R' 4
    {7,4,2,3,1,5,6,0},// R2 5
    {1,2,3,0,4,5,6,7},// U  6
    {3,0,1,2,4,5,6,7},// U' 7
    {2,3,0,1,4,5,6,7},// U2 8
    {0,1,2,3,5,6,7,4},// D  9
    {0,1,2,3,7,4,5,6},// D' 10
    {0,1,2,3,6,7,4,5},// D2 11
    {3,1,2,5,0,4,6,7},// F  12
    {4,1,2,0,5,3,6,7},// F' 13
    {5,1,2,4,3,0,6,7},// F2 14
    {0,7,1,3,4,5,2,6},// B  15
    {0,2,6,3,4,5,7,1},// B' 16
    {0,6,7,3,4,5,1,2} // B2 17
};
static const uint8_t route_tab_cr[18][8]={
    {0,0,2,1,0,2,1,0},// L  0
    {0,0,2,1,0,2,1,0},// L' 1
    {0,0,0,0,0,0,0,0},// L2 2
    {2,1,0,0,1,0,0,2},// R  3
    {2,1,0,0,1,0,0,2},// R' 4
    {0,0,0,0,0,0,0,0},// R2 5
    {0,0,0,0,0,0,0,0},// U  6
    {0,0,0,0,0,0,0,0},// U' 7
    {0,0,0,0,0,0,0,0},// U2 8
    {0,0,0,0,0,0,0,0},// D  9
    {0,0,0,0,0,0,0,0},// D' 10
    {0,0,0,0,0,0,0,0},// D2 11
    {1,0,0,2,2,1,0,0},// F  12
    {1,0,0,2,2,1,0,0},// F' 13
    {0,0,0,0,0,0,0,0},// F2 14
    {0,2,1,0,0,0,2,1},// B  15
    {0,2,1,0,0,0,2,1},// B' 16
    {0,0,0,0,0,0,0,0} // B2 17
};
static void cube_route(cube_t *c1, route_t d)
{
    uint8_t ep_tmp[12], er_tmp[12], cp_tmp[12], cr_tmp[12];
    for(int i=0; i<12; i++){
        ep_tmp[i] = c1 -> ep[route_tab_ep[d][i]];
        er_tmp[i] = c1 -> er[route_tab_ep[d][i]] ^ route_tab_er[d][i];
    }
    for(int i=0; i<8; i++){
        cp_tmp[i] = c1 -> cp[route_tab_cp[d][i]];
        cr_tmp[i] = (c1 -> cr[route_tab_cp[d][i]] + route_tab_cr[d][i] ) % 3;
    }
    for(int i=0; i<12; i++){
        c1 -> ep[i] = ep_tmp[i];
        c1 -> er[i] = er_tmp[i];
    }
    for(int i=0; i<8; i++){
        c1 -> cp[i] = cp_tmp[i];
        c1 -> cr[i] = cr_tmp[i];
    }
}
// 输出54个面的状态
static void cube_to_face_54(cube_t *c, char *cube_str)
{
    cube_str[4]  = 'U';
    cube_str[13] = 'R';
    cube_str[22] = 'F';
    cube_str[31] = 'D';
    cube_str[40] = 'L';
    cube_str[49] = 'B';
    cube_str[54] = '\0';
    for(int i=0; i<12; i++){
        int index_a = edge_to_face[i][0];
        int index_b = edge_to_face[i][1];
        const char *s = edge_index[c->ep[i] + c->er[i] * 12];
        cube_str[index_a] = s[0];
        cube_str[index_b] = s[1];
    }
    for(int i=0; i<8; i++){
        int index_a = corner_to_face[i][0];
        int index_b = corner_to_face[i][1];
        int index_c = corner_to_face[i][2];
        const char *s = corner_index[c->cp[i] + c->cr[i] * 8];
        cube_str[index_a] = s[0];
        cube_str[index_b] = s[1];
        cube_str[index_c] = s[2];
    }
}
static int cube_route_str(cube_t *c1, const char *str)
{
    const char *p = str;
    int count = 0;
    // 整理格式，顺便统计数量
    while(1)
    {
        if(*p == 'U' || *p == 'R' || *p == 'F' || *p == 'D' || *p == 'L' || *p == 'B'){
            route_t r;
            switch(*p){
                case 'L':
                    r = L;
                    break;
                case 'R':
                    r = R;
                    break;
                case 'U':
                    r = U;
                    break;
                case 'D':
                    r = D;
                    break;
                case 'F':
                    r = F;
                    break;
                case 'B':
                    r = B;
                    break;
                default:
                    assert(0);
            }
            p++;
            switch(*p){
                case '\'':
                    r += 1;
                    break;
                case '2':
                    r += 2;
                    break;
            }
            cube_route(c1, r);
            count++;
            if(*p == 0){
                break;
            }
        }
        p++;
        if(*p == 0){
            break;
        }
    }
    return count;
}
static int verify(const char *str, const char *solution)
{
    cube_t cube; 
    // 测试结果的正确性
    cube_from_face_54(&cube, str);
    cube_route_str(&cube, solution);
    // printf("totel steps = %d ",count);
    char cube_str[55];
    cube_to_face_54(&cube, cube_str);
    if(strcmp(cube_str,"UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB") == 0){
        // printf("PASS\n");
        return 1;
    }else{
        printf("FAIL %s\n", cube_str);
        return 0;
    }
}
API_EXPORT int solve_motion(const char *facelets, const char* initial_state, char* output_sequence)
{
    char solution[128] = {0}; 
    output_sequence[0] = '\0';
    // 魔方求解
    int len = solve(facelets, solution);
    if(len < 0){
        printf("魔方求解出错，报错代码: %d\n", len);
        return -1;
    }
    printf("魔方解法: %s (%d步)\n", solution, len);
    // 魔方解到机械步骤的转换
    int total_time = solution_to_motion(initial_state, solution, output_sequence);
    if(total_time < 0){
        return -1;
    }
    printf("机械步骤: %s (%dms)\n", output_sequence, total_time);
    return total_time;
}

const uint8_t index_mapping[54] = {
    6,3,0,7,4,1,8,5,2,
    47,50,53,46,49,52,45,48,51,
    17,16,15,14,13,12,11,10,9,
    26,25,24,23,22,21,20,19,18,
    38,41,44,37,40,43,36,39,42,
    27,28,29,30,31,32,33,34,35
};

// typedef struct {
//     int16_t x;
//     int16_t y;
// } Point;

API_EXPORT int color_detect_img(const int width, const int height, 
    const uint8_t *img1, const uint8_t *img2, const uint8_t *img3, 
    const Point point_xy[6], char cube_str[55])
{
    Point output_quads[18][4];
    uint8_t dominant_hsv[54][2];

    // 分割为2x3x3个子区域
    if (get_2x3x3_quads(point_xy, output_quads)) {
        perror("Error generating quads\n");
        return 1;
    }

    const uint8_t *img[3] = {img1, img2, img3};
    for(int img_index=0; img_index<3; img_index++){
        // 加载图像
        const uint8_t *bgr_image = img[img_index];
        if (!bgr_image) {
            return 1;
        }
        // 处理每个子区域
        for (int i = 0; i < 18; i++) {
            uint8_t dominant_h, dominant_s;
            uint8_t dominant_percentage;
                
            // 直接提取主色调
            extract_dominant_color_from_quad(output_quads[i], bgr_image, width, height, 
                    &dominant_h, &dominant_s, &dominant_percentage, false);
            // 查询存放的位置
            int index = index_mapping[img_index*18 + i];
            dominant_hsv[index][0] = dominant_h;
            dominant_hsv[index][1] = dominant_s;
        }
    }
    for(int i=0; i<54; i++){
        printf("%2d: H=%3d, S=%3d\n", i, dominant_hsv[i][0], dominant_hsv[i][1]);
    }
    int result = color_detect(dominant_hsv, cube_str);
    return result;
}

API_EXPORT int avg_color_detect_img(const int width, const int height, 
    const uint8_t *bgr_image, const Point point_xy[6], int color_hs[18][2])
{
    Point output_quads[18][4];
    // 分割为2x3x3个子区域
    if (get_2x3x3_quads(point_xy, output_quads)) {
        perror("Error generating quads\n");
        return 1;
    }
    if (!bgr_image) {
        return 1;
    }
    // 处理每个子区域
    for (int i = 0; i < 18; i++) {
        uint8_t dominant_h, dominant_s;
        uint8_t dominant_percentage;
            
        // 直接提取主色调
        extract_dominant_color_from_quad(output_quads[i], bgr_image, width, height, 
                &dominant_h, &dominant_s, &dominant_percentage, true);
        // 查询存放的位置
        color_hs[i][0] = dominant_h;// 色相，范围0-179，对比差异时注意边界问题 dist = diff > 90 ? 180 - diff : diff;
        color_hs[i][1] = dominant_s;// 饱和度，范围0-255
    }
    
    return 0;
}
int color_detect_bmp(char **file_name, Point point_xy[6], char cube_str[55])
{
    Point output_quads[18][4];
    uint8_t dominant_hsv[54][2];

    // 分割为2x3x3个子区域
    if (get_2x3x3_quads(point_xy, output_quads)) {
        perror("Error generating quads\n");
        return 1;
    }
    // 输出调试信息
    print_2x3x3_quads(output_quads);

    for(int bmp_index=0; bmp_index<3; bmp_index++){
        // 加载BMP图像
        int width, height;
        uint8_t *bgr_image = load_bmp_image(file_name[bmp_index], &width, &height);
        if (!bgr_image) {
            return 1;
        }
        // 处理每个子区域
        for (int i = 0; i < 18; i++) {
            uint8_t dominant_h, dominant_s;
            uint8_t dominant_percentage;
                
            // 直接提取主色调
            extract_dominant_color_from_quad(output_quads[i], bgr_image, width, height, 
                    &dominant_h, &dominant_s, &dominant_percentage, false);
                
            // 输出中间结果（添加主色占比信息）
            printf("Region %2d: H=%3d, S=%3d, dominant=%d%%\n", 
                i, dominant_h, dominant_s, dominant_percentage);
            // 查询存放的位置
            int index = index_mapping[bmp_index*18 + i];
            dominant_hsv[index][0] = dominant_h;
            dominant_hsv[index][1] = dominant_s;
        }
        free(bgr_image);
    }
    printf("----------------\n");
    for(int i=0; i<54; i++){
        printf("%2d: H=%3d, S=%3d\n", i, dominant_hsv[i][0], dominant_hsv[i][1]);
    }
    printf("----------------\n");
    int result = color_detect(dominant_hsv, cube_str);
    if(result == 0){
        printf("cube_str: %s\n", cube_str);
    }
    return result;
}

int run_solve_test(char *testcase)
{
    printf("Run auto test.\n");
    FILE *f = fopen(testcase, "r");
    char line_buffer[80];
    char solution[128] = {0}; 
    int test_case_count = 0;
    char sequence[MOTION_SEQUENCE_LEN] = {0};
    if(f != NULL){
        int min=99,max=0,avg=0;
        for(int i=0;;i++){ 
            char *line = fgets(line_buffer,79,f);
            if(line == 0){
                printf("end of file\n");
                break;
            }
            if(strlen(line_buffer) < 54){
                printf("skip %s\n",line_buffer);
                continue;
            }
            printf("测试用例%d: %s",i, line_buffer);
            test_case_count++;
            
            struct timespec start, end, start_motion, end_motion;  
            double diff, diff_motion;  
            // 测试魔方求解
            clock_gettime(CLOCK_REALTIME, &start); // 获取开始时间  
            int len = solve(line_buffer, solution);
            clock_gettime(CLOCK_REALTIME, &end); // 获取结束时间
            assert(len >= 0);
            diff = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;  
            printf("魔方解法: %s (%d步)\n", solution, len);
            // 测试魔方解到机械步骤的转换
            clock_gettime(CLOCK_REALTIME, &start_motion); // 获取开始时间  
            const char* initial_state = "RU";
            int total_time = solution_to_motion(initial_state, solution, sequence);
            clock_gettime(CLOCK_REALTIME, &end_motion); // 获取结束时间
            diff_motion = (end_motion.tv_sec - start_motion.tv_sec) * 1e3 + (end_motion.tv_nsec - start_motion.tv_nsec) / 1e6; 
            printf("机械步骤: %s (%dms)\n", sequence, total_time);
            printf("solve() 耗时: %.3lfms\n", diff);  
            printf("solution_to_motion() 耗时: %.3lfms\n", diff_motion);  
            printf("-----------------------------------------\n");  

            if(len > max){
                max = len;
            }
            if(len < min){
                min = len;
            }
            avg += len;
            
            if(verify(line_buffer, solution)==0){
                printf("verify Error\n");
                return 1;
            }

        }
        fclose(f);
        printf("min=%d, max=%d, avg=%.2f, test_case_count=%d\n",min,max,avg / (float)test_case_count,test_case_count);
    }else{
        printf("can not open %s\n",testcase);
    }
    print_table_access();
    return 0;
}

int main(int argc ,char **argv)
{
    if(argc == 3 && strcmp(argv[1], "-s") == 0){
        char sequence[MOTION_SEQUENCE_LEN] = {0};
        solve_motion(argv[2], "RU", sequence);
    }else if(argc == 3 && strcmp(argv[1], "-t") == 0){
        run_solve_test(argv[2]);
    }else if(argc == 6 && strcmp(argv[1], "-c") == 0){
        // 输入点坐标
        Point point_xy[6] = {0};
        sscanf(argv[2],"%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd", 
            &point_xy[0].x, &point_xy[0].y,
            &point_xy[1].x, &point_xy[1].y,
            &point_xy[2].x, &point_xy[2].y,
            &point_xy[3].x, &point_xy[3].y,
            &point_xy[4].x, &point_xy[4].y,
            &point_xy[5].x, &point_xy[5].y);
        char cube_str[55];
        char *file_name[3] = {argv[3], argv[4], argv[5]};
        color_detect_bmp(file_name, point_xy, cube_str);
    }else if(argc == 2 && strcmp(argv[1], "-o") == 0){
        binary_table();
    }else{
        printf("使用说明:\n");
        printf("-s: 输入魔方状态字符串，输出解法:\n");
        printf("    示例：%s -s BBBFULRUBUURFRRRDFDFLUFDLRUUUFFDRLDLRRFLLBBLFDLUBBDDBD\n", argv[0]);
        printf("-t: 读取大量测试用例，批量测试魔方求解算法:\n");
        printf("    示例：%s -t test_case.txt\n", argv[0]);
        printf("-c: 读取三张BMP图片，根据输入的魔方区域坐标，解析魔方的色块分布，然后尝试求解:\n");
        printf("    示例：%s -c 461,239,667,173,857,250,448,517,650,595,854,530 ../temp/captured_image_1.bmp ../temp/captured_image_2.bmp ../temp/captured_image_3.bmp\n", argv[0]);
        printf("-m: 输入机械步骤，控制魔方机器人运动:\n");
        printf("    示例：%s -m \"XXXXXX\"\n", argv[0]);
        printf("-o: 输出二进制格式的table表，table.bin\n");
    }
    return 0;
}
