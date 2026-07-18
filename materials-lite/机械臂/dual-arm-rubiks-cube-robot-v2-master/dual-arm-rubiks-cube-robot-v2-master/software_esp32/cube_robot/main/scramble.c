#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>

#include "esp_random.h"
#include "esp_log.h"
#include "scramble.h"

static const char *TAG = "scramble";

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
            ESP_LOGE(TAG, "Error: invalid edge %c%c.",
                   cube_str[index_a],cube_str[index_b]);
            return 0;
        }
    }
    for(int i=0; i<12; i++){
        for(int j=0; j<12; j++){
            if(i != j && c -> ep[i] == c -> ep[j]){
                ESP_LOGE(TAG, "Error: repetition edge.");
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
        ESP_LOGE(TAG, "Flipped edge, ignore.");
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
            ESP_LOGE(TAG, "Error: invalid corner %c%c%c.",
                   cube_str[index_a],cube_str[index_b],cube_str[index_c]);
            return 0;
        }
    }
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(i != j && c -> cp[i] == c -> cp[j]){
                ESP_LOGE(TAG, "Error: repetition corner.");
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
        ESP_LOGE(TAG, "Twisted corner, ignore.");
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
        ESP_LOGE(TAG, "ERROR: parity error.");
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
                    return -1;
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

int verify(const char *from, const char *to, const char *solution)
{
    cube_t cube; 
    // 测试结果的正确性
    cube_from_face_54(&cube, from);
    cube_route_str(&cube, solution);
    // ESP_LOGE(TAG, "totel steps = %d ",count);
    char cube_str[55];
    cube_to_face_54(&cube, cube_str);
    if(strcmp(cube_str, to) == 0){
        ESP_LOGI(TAG, "PASS");
        return 1;
    }else{
        ESP_LOGE(TAG, "FAIL %s", cube_str);
        return 0;
    }
}

void scramble(char *cube_str) {
    // 满足下列任意一个或多个条件时，硬件 RNG 会产生真随机数：
    // RF 子系统已启用，即 Wi-Fi 或蓝牙 已启用。
    // 调用 bootloader_random_enable() 启用了内部熵源 (SAR ADC)，且熵源尚未被 bootloader_random_disable() 禁用。
    // 在 二级引导加载程序 运行时。这是因为默认的 ESP-IDF 引导加载程序启动时会调用 bootloader_random_enable()，并在执行应用程序前调用 bootloader_random_disable()。
    
    // 创建并初始化魔方为复原状态
    cube_t cube;
    cube_from_face_54(&cube, SOLVED_CUBE);

    // 生成至少50步的随机转动
    int steps = 50 + esp_random() % 20; // 50~69步
    for (int i = 0; i < steps; i++) {
        route_t r = (route_t)(esp_random() % 18); // 随机选择0~17的操作
        cube_route(&cube, r);
    }

    // 将打乱后的状态转换为54面字符串
    cube_to_face_54(&cube, cube_str);
}

int reverse_solution(const char *solution_in, char *solution_out) {
    // 复制输入字符串以便分割
    char *input = strdup(solution_in);
    if (input == NULL) {
        return -1;
    }

    char *tokens[25]; // 存储分割后的操作
    int count = 0;

    // 分割输入字符串
    char *token = strtok(input, " ");
    while (token != NULL && count < 25) {
        tokens[count++] = token;
        token = strtok(NULL, " ");
    }

    // 逆序并取逆操作
    char reverse_tokens[25][4]; // 存储逆操作字符串

    int j = 0;
    for (int i = count - 1; i >= 0; i--, j++) {
        const char *op = tokens[i];
        size_t len = strlen(op);

        if (len == 1) {
            // 单字符操作（如 "R"）取逆为 "R'"
            sprintf(reverse_tokens[j], "%c'", op[0]);
        } else if (len == 2) {
            if (op[1] == '\'') {
                // 逆时针操作（如 "R'"）取逆为 "R"
                sprintf(reverse_tokens[j], "%c", op[0]);
            } else if (op[1] == '2') {
                // 180度操作（如 "R2"）取逆为自身
                strcpy(reverse_tokens[j], op);
            } else {
                // 非法操作，直接复制
                strcpy(reverse_tokens[j], op);
            }
        } else {
            // 非法操作，直接复制
            strcpy(reverse_tokens[j], op);
        }
    }

    // 拼接逆操作字符串
    solution_out[0] = '\0';
    for (int i = 0; i < count; i++) {
        strcat(solution_out, reverse_tokens[i]);
        if (i < count - 1) {
            strcat(solution_out, " ");
        }
    }

    // 释放复制的字符串
    free(input);
    return count;
}
/*
int main(void) {
    // 验证测试用例
    verify("BFDBUBFDUFDLRRBRLLRRLLFUUDDLLBFDURDDRLDULFUUBBRURBBFFF", "R' L' F' U B' D' B R B2 D2 F' U2 D' F2 L2 U' B2 R2 D' F2 D");
    
    // 测试逆操作函数
    char solution_out[1000];
    const char *solution_in = "R' L' F' U B' D' B R B2 D2 F' U2 D' F2 L2 U' B2 R2 D' F2 D";
    reverse_solution(solution_in, solution_out);
    printf("Input: %s\n", solution_in);
    printf("Output: %s\n", solution_out);
    // verify("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB", solution_out);
    

    // 生成随机打乱并打印
    char random_cube[55];
    scramble(random_cube);
    printf("Random Cube: %s\n", random_cube);
    return 0;
}
*/