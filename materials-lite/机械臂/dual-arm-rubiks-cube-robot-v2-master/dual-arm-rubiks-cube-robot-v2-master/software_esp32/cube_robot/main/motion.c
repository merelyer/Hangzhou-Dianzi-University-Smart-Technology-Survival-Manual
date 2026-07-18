#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>

#include "esp_system.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "motion.h"
#include "pin_def.h"
#include "scurve.h"
#include "compress.h"

static const char *TAG = "motion";

// ----------------------------------- 配置文件解析 -----------------------------------
// 转速单位转换 RPM -> 编码器计数/控制周期
static const float RPM_TO_SPEED = 16384.0f / 5000.0f / 60.0f;
static const float FINGER_FACTOR = 16384.0f / (2 * 31.416f); // 手指电机每转一圈，手指移动距离31.4mm，两手指间距增加2 * 31.4mm
#define FINGER_DIST2ENC(x) roundf((x - 52.0f) * FINGER_FACTOR)

typedef struct {
    int len;
    int16_t *arm;
}motion_table_1;

typedef struct {
    int len;
    int16_t *finger;
    int16_t *arm;
}motion_table_2;

typedef struct {
    int len;
    int16_t *finger;
    int16_t *arm;
    // int16_t *finger2; (=arm2/2)
    int16_t *arm2;
}motion_table_3;

typedef struct {
    // 零点位置
    int arm4_zero;
    int arm2_zero;
    int offset_finger1_zero;
    int offset_finger3_zero;
    
    // 位置参数（已转换为编码器值）
    int finger_clamp;
    int finger_init;
    
    // 电流参数(取值范围0-7)
    int max_current;
    int clamp_current;
    int single_clamp_current;

    // 速度比例的倒数，数值越大运动越慢
    float speed_factor;

    // 除了回零点，其他所有的运动控制都是确定的片段，采用查表法实现
    // 旋转臂空转控制表
    motion_table_2 table_no_load;
    // 翻面控制表
    motion_table_2 table_flip_90;
    motion_table_2 table_flip_180;
    // 翻面+对侧手指空转(注意这里的90/N90指的是对侧手臂的方向)控制表
    motion_table_2 table_flip_adjust_90;
    motion_table_2 table_flip_adjust_n90;
    motion_table_3 table_flip_adjust_180;
    // 拧魔方控制表
    motion_table_1 table_twist_90;
    motion_table_1 table_twist_180;

} motion_cfg;

// 全局配置实例
static motion_cfg mcfg={0};

// float get_speed_factor(void){
//     return mcfg.speed_factor;
// }
static inline int max(int a, int b) {
    return (a > b) ? a : b;
}
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// 加载表格数据（辅助函数）
int16_t *load_table(int size, FILE *f) {
    // 分配SPIRAM内存（外部内存）
    int16_t *table = heap_caps_malloc(size * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!table) {
        ESP_LOGE(TAG, "内存分配失败（SPIRAM）");
        return NULL;
    }

    int count = 0;
    char line[256];
    while (count < size && fgets(line, sizeof(line), f)) {
        // 移除行尾的换行符（兼容LF、CR、CRLF）
        char *newline = strpbrk(line, "\r\n");
        if (newline) *newline = '\0';
        
        // 跳过注释和空行
        if (line[0] == '#' || line[0] == '\0') continue;
        
        // ESP_LOGI(TAG, "count=%d, line=%s", count, line);
        
        // 移除行尾的逗号
        int len = strlen(line);
        if (len > 0 && line[len-1] == ',') {  // 检查行尾逗号
            line[len-1] = '\0';               // 移除末尾逗号
        }
        
        // 逐行解析逗号分隔的整数
        char *token = strtok(line, ",");
        while (token && count < size) {
            // 去除首尾空格
            char *endptr;
            long val = strtol(token, &endptr, 10);
            if (endptr != token && val >= -32768 && val <= 32767) {
                table[count++] = (int16_t)val;
            }else{
                ESP_LOGE(TAG, "数据格式错误");
                heap_caps_free(table);
                return NULL;
            }
            token = strtok(NULL, ",");
        }
    }

    // 校验数据完整性
    if (count != size) {
        ESP_LOGE(TAG, "表格数据不足：需要%d个值，仅找到%d个", size, count);
        heap_caps_free(table);
        return NULL;
    }
    return table;
}

static int load_table_2(FILE *file, motion_table_2 *table, int table_size)
{
    // 释放旧内存（如果存在）
    heap_caps_free(table->finger);
    table->finger = NULL;
    heap_caps_free(table->arm);
    table->arm = NULL;
    // 加载新数据
    table->finger = load_table(table_size, file);
    if (!table->finger) {
        return -1;
    }
    table->arm = load_table(table_size, file);
    if (!table->arm) {
        heap_caps_free(table->finger);
        table->finger = NULL;
        return -1;
    }
    // 记录数据长度
    table->len = table_size;
    return 0;
}
static int load_table_3(FILE *file, motion_table_3 *table, int table_size)
{
    // 释放旧内存（如果存在）
    heap_caps_free(table->finger);
    table->finger = NULL;
    heap_caps_free(table->arm);
    table->arm = NULL;
    heap_caps_free(table->arm2);
    table->arm2 = NULL;
    // 加载新数据
    table->finger = load_table(table_size, file);
    if (!table->finger) {
        return -1;
    }
    table->arm = load_table(table_size, file);
    if (!table->arm) {
        heap_caps_free(table->finger);
        table->finger = NULL;
        return -1;
    }
    table->arm2 = load_table(table_size, file);
    if (!table->arm2) {
        heap_caps_free(table->finger);
        heap_caps_free(table->arm);
        table->finger = NULL;
        table->arm = NULL;
        return -1;
    }
    // 记录数据长度
    table->len = table_size;
    return 0;
}
static int scale_table_2(motion_table_2 *table, float speed_factor)
{
    // 数值过小，内存使用量会失控，避免发生这个情况
    if(speed_factor < 0.01f){
        speed_factor = 0.01;
    }
    // 计算缩放后的长度
    int new_length = (int)round(table->len / speed_factor);
    // 分配存放缩放后数据的内存
    int16_t* new_data_arm = (int16_t*)heap_caps_malloc(new_length * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!new_data_arm) {
        ESP_LOGE(TAG, "内存分配失败（SPIRAM）");
        return -1;
    }
    int16_t* new_data_finger = (int16_t*)heap_caps_malloc(new_length * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!new_data_finger) {
        ESP_LOGE(TAG, "内存分配失败（SPIRAM）");
        return -1;
    }
    // 进行缩放
    if(scale_table(table->arm, table->len, new_data_arm, new_length) != 0){
        heap_caps_free(new_data_arm);
        heap_caps_free(new_data_finger);
        printf("缩放arm数据失败\n");
        return -1;
    }
    if(scale_table(table->finger, table->len, new_data_finger, new_length) != 0){
        heap_caps_free(new_data_arm);
        heap_caps_free(new_data_finger);
        printf("缩放finger数据失败\n");
        return -1;
    }
    // 释放缩放前的数据，更换为新数据
    heap_caps_free(table->arm);
    table->arm = new_data_arm;
    heap_caps_free(table->finger);
    table->finger = new_data_finger;
    table->len = new_length;
    return 0;
}
static int scale_table_3(motion_table_3 *table, float speed_factor)
{
    // 数值过小，内存使用量会失控，避免发生这个情况
    if(speed_factor < 0.01f){
        speed_factor = 0.01;
    }
    // 计算缩放后的长度
    int new_length = (int)round(table->len / speed_factor);
    // 分配存放缩放后数据的内存
    int16_t* new_data_arm = (int16_t*)heap_caps_malloc(new_length * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!new_data_arm) {
        ESP_LOGE(TAG, "内存分配失败（SPIRAM）");
        return -1;
    }
    int16_t* new_data_arm2 = (int16_t*)heap_caps_malloc(new_length * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!new_data_arm2) {
        ESP_LOGE(TAG, "内存分配失败（SPIRAM）");
        return -1;
    }
    int16_t* new_data_finger = (int16_t*)heap_caps_malloc(new_length * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!new_data_finger) {
        ESP_LOGE(TAG, "内存分配失败（SPIRAM）");
        return -1;
    }
    // 进行缩放
    if(scale_table(table->arm, table->len, new_data_arm, new_length) != 0){
        heap_caps_free(new_data_arm);
        heap_caps_free(new_data_arm2);
        heap_caps_free(new_data_finger);
        printf("缩放arm数据失败\n");
        return -1;
    }
    if(scale_table(table->arm2, table->len, new_data_arm2, new_length) != 0){
        heap_caps_free(new_data_arm);
        heap_caps_free(new_data_arm2);
        heap_caps_free(new_data_finger);
        printf("缩放arm数据失败\n");
        return -1;
    }
    if(scale_table(table->finger, table->len, new_data_finger, new_length) != 0){
        heap_caps_free(new_data_arm);
        heap_caps_free(new_data_arm2);
        heap_caps_free(new_data_finger);
        printf("缩放finger数据失败\n");
        return -1;
    }
    // 释放缩放前的数据，更换为新数据
    heap_caps_free(table->arm);
    table->arm = new_data_arm;
    heap_caps_free(table->arm2);
    table->arm2 = new_data_arm2;
    heap_caps_free(table->finger);
    table->finger = new_data_finger;
    table->len = new_length;
    return 0;
}
static int calc_twist_table(float accel, float jerk, float speed, int target, 
                            int16_t **arm_table, int *table_size)
{
    // 检查参数
    if(!arm_table || !table_size || !(target == 8192 || target == 16384)){
        return -1;
    }

    const float Ts = 1.0f;
    scurve_t s;
    scurve_init(&s, Ts);
    scurve_setPositionOutput(&s, 0);
    scurve_setPositionTarget(&s, target);
    scurve_setVelocityStart(&s, 0);
    scurve_setVelocityStop(&s, 0);
    scurve_setVelocityMax(&s, speed);
    scurve_setAccelerationMax(&s, accel);
    scurve_setDecelerationMax(&s, accel);
    scurve_setJerkMax(&s, jerk);

    if (scurve_startProfile(&s) != CURVE_SUCCESS) {
        ESP_LOGE(TAG, "S曲线规划器参数有误");
        return -1;
    }


    int index = 0;
    int estimated_points = 0;
    while (true) {
        scurve_state_e state = scurve_run(&s);
        if (state == CURVE_BUSY || state == CURVE_ONEND) {
            if (index == 0){
                estimated_points = (int)ceilf(s.param.totalTime / Ts) + 5; // 多分配5个点
                // 分配内存（使用SPIRAM）
                if (*arm_table) {
                    heap_caps_free(*arm_table);
                }
                *arm_table = heap_caps_malloc(estimated_points * sizeof(int16_t), MALLOC_CAP_SPIRAM);
                if (*arm_table == NULL) {
                    ESP_LOGE(TAG, "SPIRAM内存分配失败");
                    return -1;
                }
            }
            if (index >= estimated_points) {
                ESP_LOGE(TAG, "分配的内存过小(%d)", estimated_points);
                return -1;
            }
            
            float pos = scurve_getPositionOutput(&s);
            (*arm_table)[index++] = (int16_t)roundf(pos);
            
        } else {
            break;
        }
    }

    *table_size = index;
    // ESP_LOGI(TAG, "estimated_points=%d, table_size=%d", estimated_points, *table_size);
    return 0;
}

// 加载运动配置（从文件）
int load_motion_cfg(const char *path) 
{
    FILE *file = fopen(path, "r");
    if (!file) {
        ESP_LOGE(TAG, "无法打开配置文件: %s", path);
        return -1;
    }

    // 定义所有配置项的标记
    typedef struct {
        bool arm4_zero;
        bool arm2_zero;
        bool offset_finger1_zero;
        bool offset_finger3_zero;
        bool finger_clamp;
        bool finger_init;
        bool max_current;
        bool clamp_current;
        bool single_clamp_current;
        bool v_twist;
        bool a_twist;
        bool speed_factor;
        bool j_twist;
    } ConfigFlags;

    typedef struct {
        float v;
        float a;
        float j;
    } ConfigTwist;

    ConfigTwist twist = {0};
    ConfigFlags flags = {0};
    char line[256];
    int line_num = 0;
    int error_count = 0;
    
    while (fgets(line, sizeof(line), file)) {
        line_num++;
        // 跳过注释行和空行
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // 处理数据
        int table_size = 0;
        char key[64], value[64];
        if (sscanf(line, "TABLE_NO_LOAD[%d]", &table_size) == 1) {
            if( load_table_2(file, &mcfg.table_no_load, table_size) != 0){
                ESP_LOGE(TAG, "加载NO_LOAD_ARM_TABLE失败");
                error_count++;
            }
        }
        else if (sscanf(line, "TABLE_FLIP_90[%d]", &table_size) == 1) {
            if( load_table_2(file, &mcfg.table_flip_90, table_size) != 0){
                ESP_LOGE(TAG, "加载TABLE_FLIP_90失败");
                error_count++;
            }
        }else if (sscanf(line, "TABLE_FLIP_180[%d]", &table_size) == 1) {
            if( load_table_2(file, &mcfg.table_flip_180, table_size) != 0){
                ESP_LOGE(TAG, "加载TABLE_FLIP_180失败");
                error_count++;
            }
        }else if (sscanf(line, "TABLE_FLIP_ADJUST_90[%d]", &table_size) == 1) {
            if( load_table_2(file, &mcfg.table_flip_adjust_90, table_size) != 0){
                ESP_LOGE(TAG, "加载TABLE_FLIP_ADJUST_90失败");
                error_count++;
            }
        }else if (sscanf(line, "TABLE_FLIP_ADJUST_N90[%d]", &table_size) == 1) {
            if( load_table_2(file, &mcfg.table_flip_adjust_n90, table_size) != 0){
                ESP_LOGE(TAG, "加载TABLE_FLIP_ADJUST_N90失败");
                error_count++;
            }
        }else if (sscanf(line, "TABLE_FLIP_ADJUST_180[%d]", &table_size) == 1) {
            if( load_table_3(file, &mcfg.table_flip_adjust_180, table_size) != 0){
                ESP_LOGE(TAG, "加载TABLE_FLIP_ADJUST_180失败");
                error_count++;
            }
        }else if (sscanf(line, "%63[^=]= %63[^\n]", key, value) == 2) {
            // 去除key中的尾部空格
            char *end = key + strlen(key) - 1;
            while (end > key && isspace((int)*end)) end--;
            *(end + 1) = '\0';
            // 解析并标记找到的配置项
            if (strcmp(key, "ARM4_ZERO") == 0) {
                mcfg.arm4_zero = atoi(value);
                flags.arm4_zero = true;
            }
            else if (strcmp(key, "ARM2_ZERO") == 0) {
                mcfg.arm2_zero = atoi(value);
                flags.arm2_zero = true;
            }
            else if (strcmp(key, "OFFSET_FINGER_1_ZERO") == 0) {
                mcfg.offset_finger1_zero = roundf(atof(value) * FINGER_FACTOR);
                flags.offset_finger1_zero = true;
            }
            else if (strcmp(key, "OFFSET_FINGER_3_ZERO") == 0) {
                mcfg.offset_finger3_zero = roundf(atof(value) * FINGER_FACTOR);
                flags.offset_finger3_zero = true;
            }
            else if (strcmp(key, "FINGER_CLAMP") == 0) {
                mcfg.finger_clamp  = FINGER_DIST2ENC(atof(value));
                flags.finger_clamp = true;
            }
            else if (strcmp(key, "FINGER_INIT") == 0) {
                mcfg.finger_init  = FINGER_DIST2ENC(atof(value));
                flags.finger_init = true;
            }
            else if (strcmp(key, "MAX_CURRENT") == 0) {
                mcfg.max_current = atoi(value);
                flags.max_current = true;
            }
            else if (strcmp(key, "CLAMP_CURRENT") == 0) {
                mcfg.clamp_current = atoi(value);
                flags.clamp_current = true;
            }
            else if (strcmp(key, "SINGLE_CLAMP_CURRENT") == 0) {
                mcfg.single_clamp_current = atoi(value);
                flags.single_clamp_current = true;
            }
            
            else if (strcmp(key, "SPEED_FACTOR") == 0) {
                mcfg.speed_factor = atof(value);
                if(mcfg.speed_factor < 0.01f){
                    mcfg.speed_factor = 0.01f;
                }
                flags.speed_factor = true;
            }
            else if (strcmp(key, "V_TWIST") == 0) {
                twist.v = atof(value) * RPM_TO_SPEED;
                flags.v_twist = true;
            }
            else if (strcmp(key, "A_TWIST") == 0) {
                twist.a = atof(value);
                flags.a_twist = true;
            }
            else if (strcmp(key, "J_TWIST") == 0) {
                twist.j = atof(value);
                flags.j_twist = true;
            } else {
                ESP_LOGW(TAG, "未知配置项 '%s' (行 %d)", key, line_num);
            }
        }
        // 忽略其他的情况，例如完全由空格组成的行
        // else {
        //     ESP_LOGW(TAG, "无效配置行: %s (行 %d)", line, line_num);
        //     error_count++;
        // }
    }
    fclose(file);
    // 检查所有必需的配置项
    #define CHECK_FLAG(flag, name) \
        if (!flag) { \
            ESP_LOGE(TAG, "缺少必需的配置项: %s", name); \
            error_count++; \
        }
    
    CHECK_FLAG(flags.arm4_zero, "ARM4_ZERO");
    CHECK_FLAG(flags.arm2_zero, "ARM2_ZERO");
    CHECK_FLAG(flags.offset_finger1_zero, "OFFSET_FINGER_1_ZERO");
    CHECK_FLAG(flags.offset_finger3_zero, "OFFSET_FINGER_3_ZERO");
    CHECK_FLAG(flags.finger_clamp, "FINGER_CLAMP");
    CHECK_FLAG(flags.finger_init, "FINGER_INIT");
    CHECK_FLAG(flags.single_clamp_current, "SINGLE_CLAMP_CURRENT");
    CHECK_FLAG(flags.max_current, "MAX_CURRENT");
    CHECK_FLAG(flags.speed_factor, "SPEED_FACTOR");
    CHECK_FLAG(flags.clamp_current, "CLAMP_CURRENT");
    CHECK_FLAG(flags.v_twist, "V_TWIST");
    CHECK_FLAG(flags.a_twist, "A_TWIST");
    CHECK_FLAG(flags.j_twist, "J_TWIST");
    CHECK_FLAG(mcfg.table_no_load.arm, "TABLE_NO_LOAD");
    CHECK_FLAG(mcfg.table_flip_90.arm, "TABLE_FLIP_90");
    CHECK_FLAG(mcfg.table_flip_180.arm, "TABLE_FLIP_180");
    CHECK_FLAG(mcfg.table_flip_adjust_90.arm, "TABLE_FLIP_ADJUST_90");
    CHECK_FLAG(mcfg.table_flip_adjust_n90.arm, "TABLE_FLIP_ADJUST_N90");
    CHECK_FLAG(mcfg.table_flip_adjust_180.arm2, "TABLE_FLIP_ADJUST_180");

    #undef CHECK_FLAG
    
    if (error_count > 0) {
        ESP_LOGE(TAG, "配置文件 %s 有 %d 个错误", path, error_count);
        return -1;
    }

    // 根据参数，计算twist表
    float v = twist.v * mcfg.speed_factor;
    float a = twist.a * powf(mcfg.speed_factor, 2);
    float j = twist.j * powf(mcfg.speed_factor, 3);
    if(calc_twist_table(a, j, v, 8192, 
        &mcfg.table_twist_90.arm, &mcfg.table_twist_90.len) != 0){
            ESP_LOGI(TAG, "计算90度twist轨迹表失败");
        return -1;
    }
    if(calc_twist_table(a, j, v, 16384, 
        &mcfg.table_twist_180.arm, &mcfg.table_twist_180.len) != 0){
            ESP_LOGI(TAG, "计算180度twist轨迹表失败");
        return -1;
    }

    
    if(mcfg.speed_factor != 1.0f){
        // 缩放noload表
        if( scale_table_2(&mcfg.table_no_load, mcfg.speed_factor) != 0){
            ESP_LOGE(TAG, "缩放table_no_load失败");
            return -1;
        }
        // 缩放flip表
        if( scale_table_2(&mcfg.table_flip_90, mcfg.speed_factor) != 0){
            ESP_LOGE(TAG, "缩放table_flip_90失败");
            return -1;
        }
        if( scale_table_2(&mcfg.table_flip_180, mcfg.speed_factor) != 0){
            ESP_LOGE(TAG, "缩放table_flip_180失败");
            return -1;
        }
        // 缩放flip+adjust表
        if( scale_table_2(&mcfg.table_flip_adjust_n90, mcfg.speed_factor) != 0){
            ESP_LOGE(TAG, "缩放table_flip_adjust_n90失败");
            return -1;
        }
        if( scale_table_2(&mcfg.table_flip_adjust_90, mcfg.speed_factor) != 0){
            ESP_LOGE(TAG, "缩放table_flip_adjust_90失败");
            return -1;
        }
        if( scale_table_3(&mcfg.table_flip_adjust_180, mcfg.speed_factor) != 0){
            ESP_LOGE(TAG, "缩放table_flip_adjust_180失败");
            return -1;
        }
    }
    
    ESP_LOGI(TAG, "length of table_no_load         = %.1fms", 0.2f * mcfg.table_no_load.len);
    ESP_LOGI(TAG, "length of table_twist_90        = %.1fms", 0.2f * mcfg.table_twist_90.len);
    ESP_LOGI(TAG, "length of table_twist_180       = %.1fms", 0.2f * mcfg.table_twist_180.len);
    ESP_LOGI(TAG, "length of table_flip_90         = %.1fms", 0.2f * mcfg.table_flip_90.len);
    ESP_LOGI(TAG, "length of table_flip_180        = %.1fms", 0.2f * mcfg.table_flip_180.len);
    ESP_LOGI(TAG, "length of table_flip_adjust_n90 = %.1fms", 0.2f * mcfg.table_flip_adjust_n90.len);
    ESP_LOGI(TAG, "length of table_flip_adjust_90  = %.1fms", 0.2f * mcfg.table_flip_adjust_90.len);
    ESP_LOGI(TAG, "length of table_flip_adjust_180 = %.1fms", 0.2f * mcfg.table_flip_adjust_180.len);


    return 0;
}

// ----------------------------------- 485通信逻辑 -----------------------------------
// 运动控制结构体
typedef struct {
    bool init;// = true表示已经初始化
    int32_t finger_zero[2]; // [0]: finger1, [1]: finger3
    int32_t arm_zero[2];    // [0]: arm2, [1]: arm4
    int32_t finger_offset[2]; // [0]: finger1, [1]: finger3
    int32_t arm_offset[2];    // [0]: arm2, [1]: arm4
} MotionCtrl;

// 运动控制结构体，存储电机零点和当前位置信息
MotionCtrl mc = {0};

// 当前电机角度，用于web界面显示
int32_t g_angles[4] = {0};
int32_t *get_g_angle(void){
    return g_angles;
}

// 通信接口配置
#define RS485_TX_BUF_SIZE 129
#define RS485_RX_BUF_SIZE 129
#define RS485_BAUD_RATE   1000000 // 1Mbps
#define RS485_UART_PORT   2  // UART0默认是调试串口，UART1和UART2可用
#define RS485_READ_TICS   (50 / portTICK_PERIOD_MS)
// Timeout threshold for UART = number of symbols (~10 tics) with unchanged state on receive pin
#define RS485_READ_TOUT   (3) // 3.5T * 8 = 28 ticks, TOUT=3 -> ~24..33 ticks

void init_rs485(void)
{
    const int uart_num = RS485_UART_PORT;
    const uart_config_t uart_config = {
        .baud_rate = RS485_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 122,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(uart_num, RS485_TX_BUF_SIZE, RS485_RX_BUF_SIZE, 0, NULL, 0));
    // Configure UART parameters
    ESP_ERROR_CHECK(uart_param_config(uart_num, &uart_config));
    // Set UART pins as per KConfig settings
    ESP_ERROR_CHECK(uart_set_pin(uart_num, RS485_TXD, RS485_RXD, RS485_EN, -1));
    // 启用RX引脚内部上拉电阻（RS485接口芯片发送期间，RX是高阻状态）
    gpio_set_pull_mode(RS485_RXD, GPIO_PULLUP_ONLY);
    // Set RS485 half duplex mode
    ESP_ERROR_CHECK(uart_set_mode(uart_num, UART_MODE_RS485_HALF_DUPLEX));
    // Set read timeout of UART TOUT feature
    ESP_ERROR_CHECK(uart_set_rx_timeout(uart_num, RS485_READ_TOUT));
}

// CRC8计算
static uint8_t crc8(const uint8_t *datagram, int datagramLength)
{
    int i, j;
    uint8_t currentByte;
    uint8_t crc = 0;
    for (i = 0; i < datagramLength; i++)
    {                              // Execute for all bytes of a message
        currentByte = datagram[i]; // Retrieve a byte to be sent from Array
        for (j = 0; j < 8; j++)
        {
            if ((crc >> 7) ^ (currentByte & 0x01)) // update CRC based result of XOR operation
            {
                crc = (crc << 1) ^ 0x07;
            }
            else
            {
                crc = (crc << 1);
            }
            currentByte = currentByte >> 1;
        } // for CRC bit
    }
    return crc;
}

/*  
    构造主机到电机控制器的数据帧
    :param node_id:      电机的ID, 0表示广播
    :param command_type: 指令类型
    :param data:         指令附加数据
    :param data_length:  指令附加的长度
    :param output:       用于存放构造的数据帧
    :return:             构建的数据帧的长度

    ===== 主机→电机控制器数据帧格式 =====
    [字头] + [数据长度] + [指令类型 + ID号] + [数据] + [CRC8]

    [01字头]             uint16_t 固定值0xFFFF（2字节）
    [2 数据长度]          uint8_t len（1字节），指数据包的总长度，含0xFFFF和校验和
    [3 指令类型 + ID号]   低4bit：取值范围1-4，0用于广播
                        高4bit：指令类型
    [n-1 数据]           uint8_t data[250] 最大250字节
    [n CRC8]            uint8_t 校验值（含字头）
    注：每个电机仅处理ID匹配的指令块，其他忽略
*/
static int build_command_frame(int node_id, uint8_t command_type, 
                        const uint8_t *data, int data_length, uint8_t *output) {
    // 检查数据有效性
    int error = 0;
    if (node_id < 0 || node_id > 4) {
        error |= 1;
    }
    if (data_length < 0 || data_length > 250) {
        error |= 2;
    }
    if (command_type > 0x0F) {
        error |= 4;
    }
    if(error != 0){
        ESP_LOGE(TAG, "build_command_frame数据有误，错误代码：%d", error);
        return -1;
    }
    // 计算指令块总长度
    int total_length = 5 + data_length;
    
    // 构建帧
    uint8_t *ptr = output;
    *ptr++ = 0xFF;
    *ptr++ = 0xFF;
    *ptr++ = (uint8_t)total_length;
    *ptr++ = (command_type<<4) | node_id;

    // 复制数据字段(如果存在)
    if (data_length > 0) {
        if(data != NULL){
            memcpy(ptr, data, data_length);
        }
        ptr += data_length;
    }
    
    // 计算并添加CRC
    uint8_t crc = crc8(output, ptr - output);
    *ptr++ = crc;
    
    return (int)(ptr - output);
}

// 接收响应
static int receive_response(uint8_t *buffer, int expected_length) {
    int state = 0;  // 0: 等待帧头第一字节，1: 等待帧头第二字节，2: 接收数据体
    int count = 0;
    while (true) {
        uint8_t byte;
        int n = uart_read_bytes(RS485_UART_PORT, &byte, 1, RS485_READ_TICS);// 最长50ms等待
        if (n != 1) {
            ESP_LOGE(TAG, "读取串口失败");
            return -1;
        }

        // 状态机处理
        switch (state) {
            case 0:  // 等待帧头0xFF
                if (byte == 0xFF) {
                    buffer[0] = byte;
                    state = 1;
                }
                break;
                
            case 1:  // 等待第二个0xFF
                if (byte == 0xFF) {
                    buffer[1] = byte;
                    state = 2;
                    count = 2;
                } else {
                    // 非连续0xFF，重置状态
                    state = (byte == 0xFF) ? 1 : 0;
                }
                break;
                
            case 2:  // 接收数据体
                buffer[count] = byte;
                if (++count >= expected_length) {
                    return count;
                }
                break;
        }
    }
    return 0;
}

// 定义状态响应结构体
// [0] cmd_count  : uint8_t  历史指令计数（0-255循环）
// [1-2] max_error: uint16_t 最大控制误差（小端序）
// [3-4] fifo_status: uint16_t 
//          - bit15：运动状态（1=运行，0=停止）
//          - bit14：堵转（1=堵转，0=正常）
//          - bit0-13：FIFO剩余空间
// [5] temperature: int8_t   温度值（℃）
// [6-9] pos      : int32_t  当前位置（小端序）
// [10-11] voltage: int16_t  电源电压（mV，小端序）
typedef struct {
    uint8_t cmd_count;
    uint16_t pos_error;
    bool run_status;
    bool block;
    uint16_t fifo_count;
    int8_t temperature;
    int32_t position;
    int16_t voltage;
} StatResponse;

// 解析状态响应，响应帧长度应为17字节
static int parse_stat_response(const uint8_t *frame, StatResponse *response) 
{
    // 提取数据部分和CRC（前16字节为数据，最后1字节为CRC）
    uint8_t calculated_crc = crc8(frame, 16);
    if (calculated_crc != frame[16]) {
        ESP_LOGE(TAG, "CRC校验失败: 计算值%02X, 接收值%02X", calculated_crc, frame[16]);
        return -1;
    }
    
    if (frame[0] != 0xFF || frame[1] != 0xFF) {
        ESP_LOGE(TAG, "帧头必须为0xFF 0xFF");
        return -1;
    }
    
    if (frame[2] != 17) {
        ESP_LOGE(TAG, "长度标识应为17，实际为%d", frame[2]);
        return -1;
    }
    
    if (frame[3] != 0xFF) {
        ESP_LOGE(TAG, "从机标志位不为0xFF, 实际为%02X", frame[3]);
        return -1;
    }
    
    response->cmd_count = frame[4];
    response->pos_error = (frame[6] << 8) | frame[5];
    uint16_t fifo_status = (frame[8] << 8) | frame[7];
    response->run_status = (fifo_status & 0x8000) != 0;
    response->block = (fifo_status & 0x4000) != 0;
    response->fifo_count = fifo_status & 0x3FFF;
    response->temperature = (int8_t)frame[9];
    response->position = (int32_t)((frame[13] << 24) | (frame[12] << 16) | 
                                  (frame[11] << 8) | frame[10]);
    response->voltage = (int16_t)((frame[15] << 8) | frame[14]);
    
    return 0;
}

static int parse_other_response(const uint8_t *frame) 
{
    // 检查CRC
    uint8_t calculated_crc = crc8(frame, 4);
    if (calculated_crc != frame[4]) {
        ESP_LOGE(TAG, "CRC校验失败: 计算值%02X, 接收值%02X", calculated_crc, frame[4]);
        return -1;
    }
    
    // 检查标志位
    if (frame[3] != 0xFF) {
        ESP_LOGE(TAG, "从机标志位不为0xFF, 实际为%02X", frame[3]);
        return -1;
    }
    
    return 0;
}

typedef enum{
    RS485_CMD_STAT = 0,
    RS485_CMD_ENABLE = 1,
    RS485_CMD_DISABLE = 2,
    RS485_CMD_RESET = 3,
    RS485_CMD_CLEAR_FIFO = 4,
    RS485_CMD_SYNC = 5,
    RS485_CMD_FIFO = 6,
}rs485_cmd;

// 处理不带有附加数据，也没有附加返回数据的指令
static int simple_cmd(int node_id, uint8_t command_type)
{
    // 构造指令
    const int FRAME_LENGTH = 5;
    uint8_t frame[FRAME_LENGTH];
    if (build_command_frame(node_id, command_type, NULL, 0, frame) != FRAME_LENGTH) {
        return -1;
    }

    // 发送数据
    int bytes_written = uart_write_bytes(RS485_UART_PORT, frame, FRAME_LENGTH);
    if (bytes_written != FRAME_LENGTH) {
        ESP_LOGE(TAG, "simple_cmd发送命令失败");
    }

    // 接收响应
    const int RESPONSE_FRAME_LENGTH = 5;
    uint8_t response[RESPONSE_FRAME_LENGTH];
    int response_length = receive_response(response, sizeof(response));
    if (response_length < RESPONSE_FRAME_LENGTH) {
        ESP_LOGE(TAG, "simple_cmd接收响应失败");
        return -1;
    }
    
    // 检查响应
    if (parse_other_response(response) != 0) {
        return -1;
    }
    return 0;
}
// 发送同步控制命令到下位机（移除电流设置）
static int cmd_sync(int node_id_mask, bool stop_when_block)
{
    // P[4] bit 7 : 堵转时是否暂停，==1暂停
    // P[4] bit 6 : 预留，忽略
    // P[4] bit 5 : 预留，忽略
    // P[4] bit 4 : 控制电机4
    // P[4] bit 3 : 控制电机3
    // P[4] bit 2 : 控制电机2
    // P[4] bit 1 : 控制电机1
    // P[4] bit 0 : 预留，忽略
    uint8_t p4;
    p4 = node_id_mask & 0b00011110;
    if(stop_when_block){
        p4 |= 0x80;
    }

    // 构造指令
    const int FRAME_LENGTH = 5+1;
    uint8_t frame[FRAME_LENGTH];
    if (build_command_frame(0, RS485_CMD_SYNC, &p4, 1, frame) != FRAME_LENGTH) {
        return -1;
    }

    // 发送数据
    int bytes_written = uart_write_bytes(RS485_UART_PORT, frame, FRAME_LENGTH);
    if (bytes_written != FRAME_LENGTH) {
        ESP_LOGE(TAG, "cmd_sync发送命令失败");
    }
    return 0;
}
static int cmd_enable(int node_id){
    return simple_cmd(node_id, RS485_CMD_ENABLE);
}
static int cmd_disable(int node_id){
    return simple_cmd(node_id, RS485_CMD_DISABLE);
}
static int cmd_reset(int node_id){
    return simple_cmd(node_id, RS485_CMD_RESET);
}
static int cmd_clear_fifo(int node_id){
    return simple_cmd(node_id, RS485_CMD_CLEAR_FIFO);
}

// 查询状态
static int cmd_stat(int node_id, StatResponse *response) 
{
    // 查询指令禁止广播
    if(node_id == 0){
        return -1;
    }
    // 构造指令
    const int FRAME_LENGTH = 5;
    uint8_t frame[FRAME_LENGTH];
    if (build_command_frame(node_id, RS485_CMD_STAT, NULL, 0, frame) != FRAME_LENGTH) {
        return -1;
    }

    // 发送数据
    int bytes_written = uart_write_bytes(RS485_UART_PORT, frame, FRAME_LENGTH);
    if (bytes_written != FRAME_LENGTH) {
            ESP_LOGE(TAG, "cmd_stat发送命令失败");
    }

    // 接收响应
    const int RESPONSE_FRAME_LENGTH = 17;
    uint8_t response_frame[RESPONSE_FRAME_LENGTH];
    int response_length = receive_response(response_frame, sizeof(response_frame));
    if (response_length < RESPONSE_FRAME_LENGTH) {
        ESP_LOGE(TAG, "cmd_stat接收响应失败");
        return -1;
    }
    
    // 检查响应
    if (parse_stat_response(response_frame, response) != 0) {
        return -1;
    }

    // 更新电机角度信息，用于web界面显示
    g_angles[node_id - 1] = response->position;
    return 0;
}

// 获取当前位置
static int cmd_get_pos(int node_id, int32_t *pos) {
    StatResponse response;
    
    if (cmd_stat(node_id, &response) == 0) {
        *pos = response.position;
        return 0;
    }else{
        return -1;
    }
}

// 压缩并且发送运动路径 RS485_CMD_FIFO
// current_data和fixed_current二选一，当current_data==NULL时自动选择fixed_current
static int cmd_fifo(int node_id, int32_t *raw_date, uint8_t *current_data, int fixed_current, int length) 
{
    if(length > 256 || length <= 0){
        return -1;
    }
    const int MAX_FRAME_LENGTH = 128; // 至少104 + 5 + 额外的电流控制信息
    uint8_t frame[MAX_FRAME_LENGTH]; 
    int frame_length = MAX_FRAME_LENGTH - 5; // 可用输出缓冲大小
    int ret = compress_data(raw_date, current_data, fixed_current, length, &frame[4], &frame_length);
    if(ret != 0 || frame_length == 0){
        ESP_LOGE(TAG, "压缩数据失败，错误代码%d", ret);
    }
    // 构造指令
    // 通过向build_command_frame的data参数传递NULL，可以跳过复制数据的环节，使用预先放置的数据
    if (build_command_frame(node_id, RS485_CMD_FIFO, NULL, frame_length, frame) != frame_length + 5) {
        return -1;
    }

    // 发送数据
    int bytes_written = uart_write_bytes(RS485_UART_PORT, frame, frame_length + 5);
    if (bytes_written != frame_length + 5) {
        ESP_LOGE(TAG, "cmd_fifo发送命令失败");
    }

    // 接收响应
    const int RESPONSE_FRAME_LENGTH = 5;
    uint8_t response[RESPONSE_FRAME_LENGTH];
    int response_length = receive_response(response, sizeof(response));
    if (response_length < RESPONSE_FRAME_LENGTH) {
        ESP_LOGE(TAG, "cmd_fifo接收响应失败");
        return -1;
    }
    
    // 检查响应
    if (parse_other_response(response) != 0) {
        return -1;
    }
    return 0;
}

// ----------------------------------- 回零点 -----------------------------------
// 计算旋转臂最短路径目标位置函数
static int32_t calculate_arm_target(int32_t current_pos, int32_t target_zero, int32_t full_steps)
{
    // 1. 映射当前位置到单圈范围 [0, full_steps)
    int32_t current_normalized = current_pos % full_steps;
    if (current_normalized < 0) current_normalized += full_steps;
    
    // 2. 映射目标零点到单圈范围
    int32_t target_normalized = target_zero % full_steps;
    if (target_normalized < 0) target_normalized += full_steps;
    
    // 3. 计算两个可能方向的差值
    int32_t diff_forward = target_normalized - current_normalized;
    int32_t diff_backward = (target_normalized > current_normalized) ? 
        (target_normalized - full_steps - current_normalized) : 
        (full_steps - current_normalized + target_normalized);
    
    // 4. 选择最短路径
    int32_t min_diff = (abs(diff_forward) < abs(diff_backward)) ? 
        diff_forward : diff_backward;
    
    // 5. 计算最终目标位置
    return current_pos + min_diff;
}

// 辅助函数：等待FIFO有足够空间、堵转、耗尽数据自然停止
#define WAIT_BLOCK         2
#define WAIT_MOTION_FINISH 3
static int wait_for_fifo_space_or_block(int node_id, int required_space, 
                                        int32_t *pos, int32_t *error, bool exit_on_block) {
    StatResponse response;
    int64_t start_time = esp_timer_get_time();
    if(error) *error = 0;
    // 256个数据，对应时间是51.2ms，此处超时设置为100ms
    while (esp_timer_get_time() - start_time < 100*1000) {
        // 获取状态
        if (cmd_stat(node_id, &response) != 0) {
            return -1;
        }
        // 记录当前位置
        if(pos){
            *pos = response.position;
        }
        // 记录最大误差
        if(error){
            if(response.pos_error > *error){
                *error = response.pos_error;
            }
            // ESP_LOGI(TAG, "node=%d, error=%ld", node_id, *error);
        }
        // 是否堵转?
        if(exit_on_block && response.block){
            ESP_LOGI(TAG, "%d号电机堵转", node_id);
            return WAIT_BLOCK; 
        }
        // 是否耗尽数据停止？
        if(response.run_status == false){
            return WAIT_MOTION_FINISH;
        }
        // FIFO中是否有充足空间？
        if(required_space > 0){
            // FIFO剩余空间 = FIFO总容量 - 当前数据量
            // FIFO总容量为512 (2 * BLOCK_SIZE)
            int free_space = 512 - response.fifo_count;
            if (free_space >= required_space) {
                return 0; // 有足够空间
            }
        }
        
        vTaskDelay(1);
    }
    ESP_LOGE(TAG, "等待FIFO空间超时");
    return -1;
}
// 辅助函数：发送数据块并处理同步,完成后返回当前位置信息
static int send_data_block(int node_id, int32_t *data, int length, bool *first_block, 
                           bool stop_when_block, int current, int32_t *pos) {
    if (*first_block){
        // 如果是第一个块，需要清空FIFO
        if (cmd_clear_fifo(node_id) != 0) {
            ESP_LOGE(TAG, "清空FIFO失败");
            return -1;
        }
    } else {
        // 如果不是第一个块，需要等待FIFO有足够空间
        int wait_result = wait_for_fifo_space_or_block(node_id, length, pos, NULL, stop_when_block);
        if (wait_result != 0) {
            return wait_result;
        }
    }
    // 发送数据
    if (cmd_fifo(node_id, data, NULL, current, length) != 0) {
        ESP_LOGE(TAG, "发送FIFO数据失败");
        return -1;
    }
    
    // 第一次发送后立即同步
    if (*first_block) {
        if (cmd_sync(1<<node_id, stop_when_block) != 0) {
            ESP_LOGE(TAG, "发送同步命令失败");
            return -1;
        }
        *first_block = false;
    }
    
    return 0;
}
// 
static int send_data_two_block(int node_id_1, int32_t *data_1, int node_id_2, int32_t *data_2, 
                               int length, bool *first_block, 
                               bool stop_when_block, int current) {
    if (*first_block){
        // 如果是第一个块，需要清空FIFO
        if (cmd_clear_fifo(node_id_1) != 0 || cmd_clear_fifo(node_id_2) != 0) {
            ESP_LOGE(TAG, "清空FIFO失败");
            return -1;
        }
    } else {
        // 如果不是第一个块，需要等待FIFO有足够空间
        int wait_result_1 = wait_for_fifo_space_or_block(node_id_1, length, NULL, NULL, stop_when_block);
        int wait_result_2 = wait_for_fifo_space_or_block(node_id_2, length, NULL, NULL, stop_when_block);
        if (wait_result_1 != 0 || wait_result_2 != 0) {
            return -1;
        }
    }
    
    // 发送数据
    int cmd_fifo_result_1 = cmd_fifo(node_id_1, data_1, NULL, current, length);
    int cmd_fifo_result_2 = cmd_fifo(node_id_2, data_2, NULL, current, length);
    if (cmd_fifo_result_1 != 0 || cmd_fifo_result_2 != 0) {
        ESP_LOGE(TAG, "发送FIFO数据失败");
        return -1;
    }
    
    // 第一次发送后立即同步
    if (*first_block) {
        int node_id_mask = (1<<node_id_1) | (1<<node_id_2);
        if (cmd_sync(node_id_mask, stop_when_block) != 0) {
            ESP_LOGE(TAG, "发送同步命令失败");
            return -1;
        }
        *first_block = false;
    }
    
    return 0;
}

static int move_one_finger(int node_id, int current, int32_t start, int32_t stop,
    float accel, float jerk, float speed, int32_t *final_pos)
{
    if(node_id != 1 && node_id != 3){
        return -1;
    }

    // 初始化S曲线规划器
    scurve_t s;
    scurve_init(&s, 1.0f);
    scurve_setPositionOutput(&s, 0);
    scurve_setPositionTarget(&s, stop - start);
    scurve_setVelocityStart(&s, 0);
    scurve_setVelocityStop(&s, 0);
    scurve_setVelocityMax(&s, speed);
    scurve_setAccelerationMax(&s, accel);
    scurve_setDecelerationMax(&s, accel);
    scurve_setJerkMax(&s, jerk);

    if (scurve_startProfile(&s) != CURVE_SUCCESS) {
        ESP_LOGE(TAG, "S曲线规划器参数有误");
        return -1;
    }

    const int BLOCK_SIZE = 256;
    int32_t buffer[BLOCK_SIZE];
    int index = 0;
    bool first_block = true;  // 标记是否为第一个数据块

    while(true){
        scurve_state_e state = scurve_run(&s);
        if (state == CURVE_BUSY || state == CURVE_ONEND) {
            int32_t pos = start + (int32_t)roundf(scurve_getPositionOutput(&s));
            buffer[index++] = pos;
            
            // 缓冲区满时发送数据
            if (index >= BLOCK_SIZE) {
                int result = send_data_block(node_id, buffer, BLOCK_SIZE, &first_block, true, current, final_pos);
                if (result != 0) {
                    return result; // 堵转或者错误时返回
                }
                index = 0;
            }
        } else {
            break;
        }
    }
    
    // 处理剩余数据
    if (index > 0) {
        int result = send_data_block(node_id, buffer, index, &first_block, true, current, final_pos);
        if (result != 0) {
            return result;
        }
    }
    // 等待自然停止、或者堵转停止
    return wait_for_fifo_space_or_block(node_id, -1, final_pos,  NULL, true);
}

static int move_arm_and_finger(int finger_node_id, int current, 
    int32_t start_arm, int32_t start_finger, int32_t arm_offset,
    float accel, float jerk, float speed)
{
    if(finger_node_id != 1 && finger_node_id != 3){
        return -1;
    }
    int arm_node_id = finger_node_id + 1;

    // 初始化S曲线规划器
    scurve_t s;
    scurve_init(&s, 1.0f);
    scurve_setPositionOutput(&s, 0);
    scurve_setPositionTarget(&s, arm_offset);
    scurve_setVelocityStart(&s, 0);
    scurve_setVelocityStop(&s, 0);
    scurve_setVelocityMax(&s, speed);
    scurve_setAccelerationMax(&s, accel);
    scurve_setDecelerationMax(&s, accel);
    scurve_setJerkMax(&s, jerk);

    if (scurve_startProfile(&s) != CURVE_SUCCESS) {
        ESP_LOGE(TAG, "S曲线规划器参数有误");
        return -1;
    }

    const int BLOCK_SIZE = 256;
    int32_t buffer_arm[BLOCK_SIZE];
    int32_t buffer_finger[BLOCK_SIZE];
    int index = 0;
    bool first_block = true;  // 标记是否为第一个数据块

    while(true){
        scurve_state_e state = scurve_run(&s);
        if (state == CURVE_BUSY || state == CURVE_ONEND) {
            int32_t offset = roundf(scurve_getPositionOutput(&s));
            // 减速比为2:1
            int32_t pos_arm = start_arm + offset;
            int32_t pos_finger = start_finger + offset / 2;
            buffer_arm[index] = pos_arm;
            buffer_finger[index] = pos_finger;
            index++;
            // 缓冲区满时发送数据
            if (index >= BLOCK_SIZE) {
                if(send_data_two_block(arm_node_id, buffer_arm, finger_node_id, buffer_finger,
                                    BLOCK_SIZE, &first_block, true, current) != 0){
                    return -1;
                }
                index = 0;
            }
        } else {
            break;
        }
    }
    
    // 处理剩余数据
    if (index > 0) {
        if(send_data_two_block(arm_node_id, buffer_arm, finger_node_id, buffer_finger,
                           index, &first_block, true, current) != 0){
            return -1;
        }
    }
    // 等待自然停止、或者堵转停止
    int arm_result = wait_for_fifo_space_or_block(arm_node_id, -1, NULL, NULL, true);
    int finger_result = wait_for_fifo_space_or_block(arm_node_id, -1, NULL, NULL, true);
    if(arm_result == WAIT_MOTION_FINISH && finger_result == WAIT_MOTION_FINISH){
        return 0;
    }else{
        return -1;
    }
}

void enable_all(void)
{
    for(int i = 1; i <= 4; i++){
        if (cmd_enable(i) != 0){
            ESP_LOGE(TAG, "使能%d号电机失败", i);
        }
    }
}

void disable_all(void)
{
    for(int i = 1; i <= 4; i++){
        if (cmd_disable(i) != 0){
            ESP_LOGE(TAG, "禁用%d号电机失败", i);
        }
    }
}

int cmd_zero(void) 
{
    // 无限位回零电流，取值范围0-7，
    // 对应的百分比为100% * (1 + current_low) / 8
    const int current_low = 0; // 12.5%
    const int current_high = mcfg.max_current;

    const float accel = 0.02f;
    const float jerk = 0.02f;
    const float speed_slow = 50 * RPM_TO_SPEED;
    const float speed_fast = 150 * RPM_TO_SPEED;
    // 禁用全部电机
    disable_all();
    // ----------------- 手指1回零 -----------------
    // 同时使能手指1和旋转臂2
    for(int i = 1; i <= 2; i++){
        if (cmd_enable(i) != 0){
            return -1;
        }
    }
    int32_t finger1_pos_old, current_finger1_pos;
    if(cmd_get_pos(1, &finger1_pos_old) != 0){
        return -1;
    }
    if(move_one_finger(1, current_low, finger1_pos_old, finger1_pos_old-10000, 
                      accel, jerk, speed_slow, &current_finger1_pos) != WAIT_BLOCK){
        ESP_LOGE(TAG, "手指1未检测到堵转事件");
        return -1;
    }

    int32_t motion_distance = current_finger1_pos - finger1_pos_old;
    ESP_LOGI(TAG, "手指1回零移动距离: %ld, 当前位置: %ld", motion_distance, current_finger1_pos);
    
    if (abs(motion_distance) > 9500) {
        ESP_LOGE(TAG, "手指1移动距离过长，回零失败");
        return -1;
    }

    // 补偿偏移
    finger1_pos_old = current_finger1_pos;
    current_finger1_pos += mcfg.offset_finger1_zero;
    if(move_one_finger(1, current_high, finger1_pos_old, current_finger1_pos, 
                       accel, jerk, speed_slow, NULL) != WAIT_MOTION_FINISH){
        ESP_LOGE(TAG, "手指1补偿偏移失败");
        return -1;
    }

    // 获取旋转臂2当前位置
    int32_t arm2_old;
    if(cmd_get_pos(2, &arm2_old) != 0){
        return -1;
    }
    
    // 计算旋转臂2目标位置（考虑多圈）
    int32_t arm2_zero_pos = calculate_arm_target(arm2_old, mcfg.arm2_zero, 16384);
    int32_t arm2_offset = arm2_zero_pos - arm2_old;
    int32_t finger1_target = current_finger1_pos + arm2_offset / 2;
    if(arm2_offset != 0){
        if(move_arm_and_finger(1, current_high, 
                               arm2_old, current_finger1_pos, arm2_offset,
                               accel, jerk, speed_fast) != 0){
            ESP_LOGE(TAG, "旋转臂2回零点失败");
            return -1;
        }
    }
    
    // ----------------- 手指2回零 -----------------
    // 同时使能手指3和旋转臂4
    for(int i = 3; i <= 4; i++){
        if (cmd_enable(i) != 0){
            return -1;
        }
    }
    int32_t finger3_pos_old, current_finger3_pos;
    if(cmd_get_pos(3, &finger3_pos_old) != 0){
        return -1;
    }
    if(move_one_finger(3, current_low, finger3_pos_old, finger3_pos_old-10000, 
                      accel, jerk, speed_slow, &current_finger3_pos) != WAIT_BLOCK){
        ESP_LOGE(TAG, "手指3未检测到堵转事件");
        return -1;
    }

    motion_distance = current_finger3_pos - finger3_pos_old;
    ESP_LOGI(TAG, "手指3回零移动距离: %ld, 当前位置: %ld", motion_distance, current_finger3_pos);
    
    if (abs(motion_distance) > 9500) {
        ESP_LOGE(TAG, "手指3移动距离过长，回零失败");
        return -1;
    }

    // 补偿偏移
    finger3_pos_old = current_finger3_pos;
    current_finger3_pos += mcfg.offset_finger3_zero;
    if(move_one_finger(3, current_high, finger3_pos_old, current_finger3_pos, 
                       accel, jerk, speed_slow, NULL) != WAIT_MOTION_FINISH){
        ESP_LOGE(TAG, "手指3补偿偏移失败");
        return -1;
    }

    // 获取旋转臂4当前位置
    int32_t arm4_old;
    if(cmd_get_pos(4, &arm4_old) != 0){
        return -1;
    }
    
    // 计算旋转臂4目标位置（考虑多圈）
    int32_t arm4_zero_pos = calculate_arm_target(arm4_old, mcfg.arm4_zero, 16384);
    int32_t arm4_offset = arm4_zero_pos - arm4_old;
    int32_t finger3_target = current_finger3_pos + arm4_offset / 2;
    if(arm4_offset != 0){
        if(move_arm_and_finger(3, current_high, 
                               arm4_old, current_finger3_pos, arm4_offset,
                               accel, jerk, speed_fast) != 0){
            ESP_LOGE(TAG, "旋转臂4回零点失败");
            return -1;
        }
    }
  
    ESP_LOGI(TAG, "回零点成功:");
    ESP_LOGI(TAG, "  手指1位置: %ld", finger1_target);
    ESP_LOGI(TAG, "  旋转臂2位置: %ld", arm2_zero_pos);
    ESP_LOGI(TAG, "  手指3位置: %ld", finger3_target);
    ESP_LOGI(TAG, "  旋转臂4位置: %ld", arm4_zero_pos);

    // 初始化运动控制
    mc.finger_zero[0]   = finger1_target;
    mc.arm_zero[0]      = arm2_zero_pos;
    mc.finger_zero[1]   = finger3_target;
    mc.arm_zero[1]      = arm4_zero_pos;
    mc.finger_offset[0] = 0;
    mc.finger_offset[1] = 0;
    mc.arm_offset[0]    = 0;
    mc.arm_offset[1]    = 0;
    mc.init = true;

    return 0;
}

// ----------------------------------- 夹紧、松开魔方 -----------------------------------
// 移动两个手指
static int move_two_finger_raw(int32_t target, int *error, int current) 
{
    if (!mc.init) {
        ESP_LOGE(TAG, "需要先调用cmd_zero");
        return -1;
    }
    if(mc.finger_offset[0] - mc.arm_offset[0] / 2 == target && 
       mc.finger_offset[1] - mc.arm_offset[1] / 2 == target ){
        // 已经在指定位置了，无需处理
        *error = 512; // 随便报一个误差，防止后续代码报错
        ESP_LOGI(TAG, "两个手指已经处于指定位置");
        return 0;
    }
    int32_t finger1_now = mc.finger_zero[0] + mc.finger_offset[0];
    int32_t finger3_now = mc.finger_zero[1] + mc.finger_offset[1];

    mc.finger_offset[0] = target + mc.arm_offset[0] / 2;
    mc.finger_offset[1] = target + mc.arm_offset[1] / 2;
    
    int32_t finger1_target = mc.finger_zero[0] + mc.finger_offset[0];
    int32_t finger3_target = mc.finger_zero[1] + mc.finger_offset[1];
    
    // 这个函数不会频繁调用，对性能影响很小
    // 初始化S曲线规划器，处于简化代码考虑，使用固定参数
    const float jerk = 0.1f;
    const float v_finger = 200 * RPM_TO_SPEED;
    const float a_finger = 1.0f;
    scurve_t s;
    scurve_init(&s, 1.0f);
    scurve_setPositionOutput(&s, 0);
    scurve_setPositionTarget(&s, finger1_target - finger1_now);
    scurve_setVelocityStart(&s, 0);
    scurve_setVelocityStop(&s, 0);
    scurve_setVelocityMax(&s, v_finger);
    scurve_setAccelerationMax(&s, a_finger);
    scurve_setDecelerationMax(&s, a_finger);
    scurve_setJerkMax(&s, jerk);

    if (scurve_startProfile(&s) != CURVE_SUCCESS) {
        ESP_LOGE(TAG, "S曲线规划器参数有误");
        return -1;
    }

    const int BLOCK_SIZE = 256;
    int32_t buffer_finger1[BLOCK_SIZE];
    int32_t buffer_finger3[BLOCK_SIZE];
    int index = 0;
    bool first_block = true;  // 标记是否为第一个数据块

    while(true){
        scurve_state_e state = scurve_run(&s);
        if (state == CURVE_BUSY || state == CURVE_ONEND) {
            int32_t finger_offset = roundf(scurve_getPositionOutput(&s));
            buffer_finger1[index] = finger1_now + finger_offset;
            buffer_finger3[index] = finger3_now + finger_offset;
            index++;
            // 缓冲区满时发送数据
            if (index >= BLOCK_SIZE) {
                if(send_data_two_block(1, buffer_finger1, 3, buffer_finger3,
                                    BLOCK_SIZE, &first_block, false, current) != 0){
                    return -1;
                }
                index = 0;
            }
        } else {
            break;
        }
    }
    
    // 处理剩余数据
    if (index > 0) {
        if(send_data_two_block(1, buffer_finger1, 3, buffer_finger3,
                               index, &first_block, false, current) != 0){
            return -1;
        }
    }
    // 等待自然停止、或者堵转停止
    int32_t finger1_real = 0, finger3_real = 0;
    int finger1_result = wait_for_fifo_space_or_block(1, -1, &finger1_real, NULL, false);
    int finger3_result = wait_for_fifo_space_or_block(3, -1, &finger3_real, NULL, false);
    int finger1_error = abs(finger1_target - finger1_real);
    int finger3_error = abs(finger3_target - finger3_real);
    // ESP_LOGI(TAG, "手指1位置误差: %d, 手指3位置误差: %d", finger1_error, finger3_error);
    if(error){
        *error = min(finger1_error, finger3_error);
    }
    if(finger1_result == WAIT_MOTION_FINISH && finger3_result == WAIT_MOTION_FINISH){
        return 0;
    }else{
        return -1;
    }
}
// 手指初始化位置
int two_finger_init(void) 
{   
    return move_two_finger_raw(mcfg.finger_init, NULL, mcfg.max_current);
}

// 手指夹紧
int two_finger_clamp(void) 
{
    int error;
    int ret = move_two_finger_raw(mcfg.finger_clamp, &error, mcfg.clamp_current);
    // 通过测量误差的值，检测有没有夹住魔方
    if(ret == 0 && error > (int)(0.5f * FINGER_FACTOR)){
        return 0;
    }else{
        // TODO 添加魔方夹紧检测，返回-1
        ESP_LOGE(TAG, "夹紧魔方失败, 误差%.1fmm, 需要大于0.5mm",  error/FINGER_FACTOR);
        return -1;
    }
}

// ----------------------------------- 运动轨迹控制代码 -----------------------------------
// 同时控制4个电机
// 全部电机配置为堵转后电流减半，防止损坏部件
// 每发完一个block，控制程序都会检查上一个block的控制误差，超标后会触发堵转保护
// TODO检查误差是否超标
static int send_data_four_block(int32_t data[4][256], int length, bool *first_block, 
                                int arm_current, uint8_t finger_current_1[256], uint8_t finger_current_3[256], 
                                int32_t error[4]) 
{
    if (*first_block){
        // 如果是第一个块，需要清空FIFO
        for(int i=1; i<=4; i++){
            if (cmd_clear_fifo(i) != 0) {
                ESP_LOGE(TAG, "清空电机控制板%d的FIFO失败", i);
                return -1;
            }
        }
    } else {
        // 如果不是第一个块，需要等待FIFO有足够空间
        for(int i=4; i>=1; i--){
            int wait_result = wait_for_fifo_space_or_block(i, length, NULL, &error[i-1], false);
            if (wait_result != 0) {
                ESP_LOGE(TAG, "等待电机控制板%d的FIFO有足够空间时出错，错误代码%d", i, wait_result);
                return -1;
            }
        }
    }
    // 发送数据
    int result[4];
    result[0] = cmd_fifo(1, data[0], finger_current_1, 0, length);
    result[1] = cmd_fifo(2, data[1], NULL, arm_current, length);
    result[2] = cmd_fifo(3, data[2], finger_current_3, 0, length);
    result[3] = cmd_fifo(4, data[3], NULL, arm_current, length);
    for(int i=0; i<4; i++){
        if (result[i] != 0) {
            ESP_LOGE(TAG, "发送电机控制板%d的FIFO数据失败", i+1);
            return -1;
        }
    }

    // 第一次发送后立即同步
    if (*first_block) {
        const int node_id_mask = (1<<1) | (1<<2) | (1<<3) | (1<<4);
        if (cmd_sync(node_id_mask, false) != 0) {
            ESP_LOGE(TAG, "发送同步命令失败");
            return -1;
        }
        *first_block = false;
    }
    
    return 0;
}

#define RIGHT    0
#define LEFT     1

#define DIR_N90   0
#define DIR_90    1
#define DIR_180   2

#define TYPE_ARM_BACK      0
#define TYPE_TWIST         1
#define TYPE_FLIP          2
#define TYPE_FLIP_ADJUST   3

#define RIGHT_FINGER_1 0 // pos_offset[0] 右手指，1号
#define RIGHT_ARM_2    1 // pos_offset[1] 右手臂，2号
#define LEFT_FINGER_3  2 // pos_offset[2] 左手指，3号
#define LEFT_ARM_4     3 // pos_offset[3] 左手臂，4号

typedef union {
    uint8_t cmd;
    struct{
        uint8_t type       : 2;
        uint8_t left_right : 1;
        uint8_t dir        : 2;
        uint8_t rsv        : 3;
    };
} motion_cmd;
static_assert(sizeof(motion_cmd) == 1);

// 从控制板加载位置数据
// 返回 0：成功，后面还有数据
// 返回 1：成功，最后一个数据
// 返回其他：失败
static int get_pos_from_table(int32_t pos_offset[4], uint8_t *finger_1_current, uint8_t *finger_3_current, int i, motion_cmd cmd)
{
    int status = -1;
    if(cmd.type == TYPE_ARM_BACK){
        // 单侧手臂空转90°，手指避开魔方
        // .type = TYPE_ARM_BACK, .left_right = LEFT/RIGHT
        int len = mcfg.table_no_load.len;
        int32_t pos_arm = mcfg.table_no_load.arm[i];
        int32_t pos_finger = mcfg.table_no_load.finger[i];
        // 从数组中取出相对开始位置的偏移量
        if(cmd.left_right == RIGHT){
            pos_offset[RIGHT_FINGER_1] = pos_finger + mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = pos_arm    + mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = mc.arm_offset[LEFT];
        }else{
            pos_offset[RIGHT_FINGER_1] = mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = pos_finger + mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = pos_arm    + mc.arm_offset[LEFT];
        }
        // 判断是否处于夹紧状态，如果是，减小电流
        int finger_1_offset = pos_offset[RIGHT_FINGER_1] - pos_offset[RIGHT_ARM_2] / 2;
        *finger_1_current = finger_1_offset > mcfg.finger_init ? mcfg.max_current : mcfg.clamp_current;
        int finger_3_offset = pos_offset[LEFT_FINGER_3] - pos_offset[LEFT_ARM_4] / 2;
        *finger_3_current = finger_3_offset > mcfg.finger_init ? mcfg.max_current : mcfg.clamp_current;
        if(i < len - 1){
            // 增大对侧(对侧，其他分支有本侧的，注意区分)手指的驱动电流，减小魔方“低头”的现象
            uint8_t curr = mcfg.single_clamp_current;
            if(cmd.left_right == RIGHT){
                if(*finger_3_current < curr) *finger_3_current = curr;
            }else if(cmd.left_right == LEFT){
                if(*finger_1_current < curr) *finger_1_current = curr;
            }
            status = 0;
        }else{
            // 在处理最后一个数据时，还原正常的驱动电流，使用上一个步骤设置的，而不进行覆盖操作
            status = 1;
        }
    }else if(cmd.type == TYPE_TWIST){
        // 拧魔方，旋转魔方的其中一层
        //  .type = TYPE_TWIST, .left_right = LEFT/RIGHT, .cw_ccw = CCW90/CW90/CW180
        int len;
        int32_t pos;  
        switch (cmd.dir)
        {
        case DIR_N90:
            pos = -mcfg.table_twist_90.arm[i];
            len = mcfg.table_twist_90.len;
            break;
        case DIR_90:
            pos = mcfg.table_twist_90.arm[i];
            len = mcfg.table_twist_90.len;
            break;
        case DIR_180:
            pos = mcfg.table_twist_180.arm[i];
            len = mcfg.table_twist_180.len;
            break;
        default:
            return -1;
        }
        if(cmd.left_right == RIGHT){
            pos_offset[RIGHT_FINGER_1] = pos/2 + mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = pos   + mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = mc.arm_offset[LEFT];
        }else{
            pos_offset[RIGHT_FINGER_1] = mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = pos/2 + mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = pos   + mc.arm_offset[LEFT];
        }
        // 整个过程中都是夹紧的状态
        *finger_1_current = mcfg.clamp_current;
        *finger_3_current = mcfg.clamp_current;
        if(i < len - 1){
            status = 0;
        }else{
            status = 1;
        }
    }else if(cmd.type == TYPE_FLIP){
        // 翻转魔方，同时旋转魔方的三层
        //  .type = TYPE_FLIP, .left_right = LEFT/RIGHT, .cw_ccw = CCW90/CW90/CW180
        int len;
        int32_t arm, finger;  
        switch (cmd.dir)
        {
        case DIR_N90:
            arm = -mcfg.table_flip_90.arm[i];
            finger = mcfg.table_flip_90.finger[i];
            len = mcfg.table_flip_90.len;
            break;
        case DIR_90:
            arm = mcfg.table_flip_90.arm[i];
            finger = mcfg.table_flip_90.finger[i];
            len = mcfg.table_flip_90.len;
            break;
        case DIR_180:
            arm = mcfg.table_flip_180.arm[i];
            finger = mcfg.table_flip_180.finger[i];
            len = mcfg.table_flip_180.len;
            break;
        default:
            return -1;
        }
        if(cmd.left_right == RIGHT){
            pos_offset[RIGHT_FINGER_1] = arm/2  + mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = arm    + mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = finger + mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = mc.arm_offset[LEFT];
        }else{
            pos_offset[RIGHT_FINGER_1] = finger + mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = arm/2 + mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = arm   + mc.arm_offset[LEFT];
        }
        // 判断是否处于夹紧状态，如果是，减小电流
        int finger_1_offset = pos_offset[RIGHT_FINGER_1] - pos_offset[RIGHT_ARM_2] / 2;
        *finger_1_current = finger_1_offset > mcfg.finger_init ? mcfg.max_current : mcfg.clamp_current;
        int finger_3_offset = pos_offset[LEFT_FINGER_3] - pos_offset[LEFT_ARM_4] / 2;
        *finger_3_current = finger_3_offset > mcfg.finger_init ? mcfg.max_current : mcfg.clamp_current;
        // 增大本侧(不是对侧)手指的驱动电流，减小魔方“低头”的现象
        if(i < len - 1){
            uint8_t curr = mcfg.single_clamp_current;
            if(cmd.left_right == LEFT){
                if(*finger_3_current < curr) *finger_3_current = curr;
            }else if(cmd.left_right == RIGHT){
                if(*finger_1_current < curr) *finger_1_current = curr;
            }
            status = 0;
        }else{
            // 在处理最后一个数据时，还原正常的驱动电流，使用上一个步骤设置的，而不进行覆盖操作
            status = 1;
        }
    }else if(cmd.type == TYPE_FLIP_ADJUST){
        // 翻转魔方，同时旋转魔方的三层
        // 与此同时，对侧手臂旋转90°
        //  .type = TYPE_FLIP, .left_right = LEFT/RIGHT, .cw_ccw = CCW90/CW90/CW180
        int len;
        int32_t arm_flip, arm_no_load, finger_no_load;
       
        switch (cmd.dir)
        {        
        case DIR_N90:
            // 注意cmd.dir是翻转的手臂的方向，table_flip_adjust_n90/90是对侧空转手臂的方向，正好相反
            // DIR_N90对应的是table_flip_adjust_90
            // DIR_90对应的是table_flip_adjust_n90
            arm_flip = - mcfg.table_flip_adjust_90.arm[i];
            arm_no_load = mcfg.table_flip_adjust_90.arm[i];
            finger_no_load = mcfg.table_flip_adjust_90.finger[i];
            len = mcfg.table_flip_adjust_90.len;
            break;
        case DIR_90:
            // 左右旋转臂方向相反，角度一致，因此arm_flip带有负号
            arm_flip = - mcfg.table_flip_adjust_n90.arm[i];
            arm_no_load = mcfg.table_flip_adjust_n90.arm[i];
            finger_no_load = mcfg.table_flip_adjust_n90.finger[i];
            len = mcfg.table_flip_adjust_n90.len;
            break;
        case DIR_180:
            // 180°
            arm_flip = mcfg.table_flip_adjust_180.arm2[i];
            arm_no_load = mcfg.table_flip_adjust_180.arm[i];
            finger_no_load = mcfg.table_flip_adjust_180.finger[i];
            len = mcfg.table_flip_adjust_180.len;
            break;
        default:
            return -1;
        }
        if(cmd.left_right == RIGHT){
            pos_offset[RIGHT_FINGER_1] = arm_flip/2  + mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = arm_flip    + mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = finger_no_load + mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = arm_no_load + mc.arm_offset[LEFT];
        }else{
            pos_offset[RIGHT_FINGER_1] = finger_no_load + mc.finger_offset[RIGHT];
            pos_offset[RIGHT_ARM_2]    = arm_no_load + mc.arm_offset[RIGHT];
            pos_offset[LEFT_FINGER_3]  = arm_flip/2 + mc.finger_offset[LEFT];
            pos_offset[LEFT_ARM_4]     = arm_flip   + mc.arm_offset[LEFT];
        }
        // 判断是否处于夹紧状态，如果是，减小电流
        int finger_1_offset = pos_offset[RIGHT_FINGER_1] - pos_offset[RIGHT_ARM_2] / 2;
        *finger_1_current = finger_1_offset > mcfg.finger_init ? mcfg.max_current : mcfg.clamp_current;
        int finger_3_offset = pos_offset[LEFT_FINGER_3] - pos_offset[LEFT_ARM_4] / 2;
        *finger_3_current = finger_3_offset > mcfg.finger_init ? mcfg.max_current : mcfg.clamp_current;
        // 增大本侧(不是对侧)手指的驱动电流，减小魔方“低头”的现象
        if(i < len - 1){
            uint8_t curr = mcfg.single_clamp_current;
            if(cmd.left_right == LEFT){
                if(*finger_3_current < curr) *finger_3_current = curr;
            }else if(cmd.left_right == RIGHT){
                if(*finger_1_current < curr) *finger_1_current = curr;
            }
            status = 0;
        }else{
            // 在处理最后一个数据时，还原正常的驱动电流，使用上一个步骤设置的，而不进行覆盖操作
            status = 1;
        }
    }
    
    return status;
}

static int check_arm_finger_error(int32_t max_error[4], int32_t error[4]) 
{
    for(int i=0; i<4; i++){
        if(error[i] > max_error[i]){
            max_error[i] = error[i];
        }
    }
    const int32_t MAX_FINGER_ERROR = 10 * FINGER_FACTOR;      // 10mm
    const int32_t MAX_ARM_ERROR = 16384.0f / 180.0f * 30.0f;  // 30°
    if(error[0] > MAX_FINGER_ERROR || error[2] > MAX_FINGER_ERROR){
        ESP_LOGE(TAG, "手指电机堵转，1号电机%ld，3号电机%ld", error[0], error[2]);
        return -1;
    }
    if(error[1] > MAX_ARM_ERROR || error[3] > MAX_ARM_ERROR){
        ESP_LOGE(TAG, "手臂电机堵转，2号电机%ld，4号电机%ld", error[1], error[3]);
        return -1;
    }

    return 0;
}

static void print_max_error(int32_t max_error[4]){
    ESP_LOGI(TAG, "误差统计：右手指=%.1fmm, 右旋转臂=%.1f°, 左手指=%.1fmm, 左旋转臂=%.1f°", 
        max_error[0] / FINGER_FACTOR, 
        max_error[1] / (16384.0f / 180.0f), 
        max_error[2] / FINGER_FACTOR, 
        max_error[3] / (16384.0f / 180.0f)
    );
}
/*
action示例
旋转一层：L*T R*T L-T L+T R-T R+T 
翻面：L1 R*F L0 R1 L*F R0 L1 R+F L0 L1 R-F L0 R1 L+F R0 R1 L-F R0 
翻面+调整对侧手臂：
L2 L+N L0 
R2 R+N L+F R0 
L2 L+N R+F L0 
R2 R+N L-F R0 
L2 L+N R-F L0 
L2 L+N R*F L0 
R2 R+N R0 
R2 R+N L*F R0
*/
static int translate_motions_str(const char *actions, motion_cmd cmd_list[128], int *estimate_time_cost, int *cmd_count){
    *estimate_time_cost = 0;
    *cmd_count = 0;
    // 定义动作映射表（按token数量从多到少排序）
    typedef struct {
        const char *words[4];// token
        int word_count;      // token长度
        motion_cmd cmd;      // 对应指令
        int *p_time_cost;    // 指令耗时 
    } action_map;
    static const action_map map[] = {
        // 4个token的动作
        {{"L2", "L+N", "R-F", "L0"}, 4, { .type = TYPE_FLIP_ADJUST, .left_right = RIGHT, .dir = DIR_N90 }, &mcfg.table_flip_adjust_90.len},
        {{"L2", "L+N", "R+F", "L0"}, 4, { .type = TYPE_FLIP_ADJUST, .left_right = RIGHT, .dir = DIR_90 }, &mcfg.table_flip_adjust_n90.len},
        {{"L2", "L+N", "R*F", "L0"}, 4, { .type = TYPE_FLIP_ADJUST, .left_right = RIGHT, .dir = DIR_180 }, &mcfg.table_flip_adjust_180.len},
        {{"R2", "R+N", "L-F", "R0"}, 4, { .type = TYPE_FLIP_ADJUST, .left_right = LEFT, .dir = DIR_N90 }, &mcfg.table_flip_adjust_90.len},
        {{"R2", "R+N", "L+F", "R0"}, 4, { .type = TYPE_FLIP_ADJUST, .left_right = LEFT, .dir = DIR_90 }, &mcfg.table_flip_adjust_n90.len},
        {{"R2", "R+N", "L*F", "R0"}, 4, { .type = TYPE_FLIP_ADJUST, .left_right = LEFT, .dir = DIR_180 }, &mcfg.table_flip_adjust_180.len},
        // 3个token的动作
        {{"L2", "L+N", "L0"}, 3, { .type = TYPE_ARM_BACK, .left_right = LEFT }, &mcfg.table_no_load.len},
        {{"R2", "R+N", "R0"}, 3, { .type = TYPE_ARM_BACK, .left_right = RIGHT }, &mcfg.table_no_load.len},
        {{"L1", "R+F", "L0"}, 3, { .type = TYPE_FLIP, .left_right = RIGHT, .dir = DIR_90 }, &mcfg.table_flip_90.len},
        {{"L1", "R-F", "L0"}, 3, { .type = TYPE_FLIP, .left_right = RIGHT, .dir = DIR_N90 }, &mcfg.table_flip_90.len},
        {{"L1", "R*F", "L0"}, 3, { .type = TYPE_FLIP, .left_right = RIGHT, .dir = DIR_180 }, &mcfg.table_flip_180.len},
        {{"R1", "L+F", "R0"}, 3, { .type = TYPE_FLIP, .left_right = LEFT, .dir = DIR_90 }, &mcfg.table_flip_90.len},
        {{"R1", "L-F", "R0"}, 3, { .type = TYPE_FLIP, .left_right = LEFT, .dir = DIR_N90 }, &mcfg.table_flip_90.len},
        {{"R1", "L*F", "R0"}, 3, { .type = TYPE_FLIP, .left_right = LEFT, .dir = DIR_180 }, &mcfg.table_flip_180.len},
        // 1个token的动作
        {{"R-T"}, 1, { .type = TYPE_TWIST, .left_right = RIGHT, .dir = DIR_N90 }, &mcfg.table_twist_90.len},
        {{"R+T"}, 1, { .type = TYPE_TWIST, .left_right = RIGHT, .dir = DIR_90 }, &mcfg.table_twist_90.len},
        {{"R*T"}, 1, { .type = TYPE_TWIST, .left_right = RIGHT, .dir = DIR_180 }, &mcfg.table_twist_180.len},
        {{"L-T"}, 1, { .type = TYPE_TWIST, .left_right = LEFT, .dir = DIR_N90 }, &mcfg.table_twist_90.len},
        {{"L+T"}, 1, { .type = TYPE_TWIST, .left_right = LEFT, .dir = DIR_90 }, &mcfg.table_twist_90.len},
        {{"L*T"}, 1, { .type = TYPE_TWIST, .left_right = LEFT, .dir = DIR_180 }, &mcfg.table_twist_180.len}
    };
    const int map_size = sizeof(map) / sizeof(map[0]);

    // 直接处理输入字符串（避免内存复制）
    const char *p = actions;
    while (*p != '\0' && *cmd_count < 127) {
        // 跳过空格
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p == '\0') break;

        // 记录当前token起始位置
        const char *start = p;
        
        // 找到当前token结束位置
        while (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\0') p++;
        int token_len = p - start;

        // 尝试匹配各种长度的动作模式
        int matched = 0;
        for (int i = 0; i < map_size; i++) {
            const action_map *m = &map[i];
            
            // 检查剩余长度是否足够
            const char *check_ptr = start;
            int match_ok = 1;
            
            // 检查当前token是否匹配模式中的第一个token
            int first_token_len = strlen(m->words[0]);
            if (token_len != first_token_len || 
                memcmp(start, m->words[0], token_len) != 0) {
                continue;
            }
            
            // 检查后续token是否匹配
            for (int j = 1; j < m->word_count; j++) {
                // 移动到下一个token
                while (*check_ptr != '\0' && !(*check_ptr == ' ' || *check_ptr == '\t' || *check_ptr == '\n')) {
                    check_ptr++;
                }
                while (*check_ptr == ' ' || *check_ptr == '\t' || *check_ptr == '\n') check_ptr++;
                
                // 检查是否结束
                if (*check_ptr == '\0') {
                    match_ok = 0;
                    break;
                }
                
                // 获取下一个token
                const char *next_start = check_ptr;
                while (*check_ptr != ' ' && *check_ptr != '\t' && *check_ptr != '\n' && *check_ptr != '\0') check_ptr++;
                int next_token_len = check_ptr - next_start;
                
                // 比较token
                int cmp_len = strlen(m->words[j]);
                if (next_token_len != cmp_len || 
                    memcmp(next_start, m->words[j], next_token_len) != 0) {
                    match_ok = 0;
                    break;
                }
            }
            
            // 如果匹配成功
            if (match_ok) {
                cmd_list[(*cmd_count)++] = m->cmd;
                *estimate_time_cost += *(m->p_time_cost);
                if (m->word_count > 1) {
                    p = check_ptr;  // 对于多token动作，更新指针到最后一个token之后
                }
                matched = 1;
                break;
            }
        }
        
        // 如果没有匹配任何模式
        if (!matched) {
            // 提取当前token用于错误信息
            char token_buf[32];
            int copy_len = token_len < 31 ? token_len : 31;
            memcpy(token_buf, start, copy_len);
            token_buf[copy_len] = '\0';
            
            ESP_LOGI(TAG, "未知动作: %s", token_buf);
            return -1;
        }
    }
    return 0;
}
int get_time_motions_str(const char *actions){
    int cmd_count, estimate_time_cost;
    motion_cmd cmd_list[128];
    if(translate_motions_str(actions, cmd_list, &estimate_time_cost, &cmd_count) != 0){
        return -1;
    }else{
        // ESP_LOGI(TAG, "%s -> %d", actions, estimate_time_cost);
        return (int)roundf(estimate_time_cost * 0.2f);
    }
}

int motions(const char *actions) 
{
    int64_t start_time = esp_timer_get_time();
    int time_in_second = 1;
    if (!mc.init) {
        ESP_LOGE(TAG, "需要先调用cmd_zero");
        return -1;
    }
    motion_cmd cmd_list[128];
    int cmd_count, estimate_time_cost;
    if(translate_motions_str(actions, cmd_list, &estimate_time_cost, &cmd_count) != 0){
        ESP_LOGE(TAG, "解析动作序列失败");
    }
    // ESP_LOGI(TAG, "预估耗时%.3fs", estimate_time_cost / 5000.0f);
    // for(int i=0; i<cmd_count; i++){
    //     ESP_LOGI(TAG, "index=%d: type=%d, lr=%d, dir=%d", i, cmd_list[i].type, cmd_list[i].left_right, cmd_list[i].dir);
    // }
    // 块大小
    const int BLOCK_SIZE = 256;
    // 四个电机的位置控制（严格同步）
    int32_t data[4][BLOCK_SIZE];
    // 电流控制
    uint8_t finger_1_current[BLOCK_SIZE];
    uint8_t finger_3_current[BLOCK_SIZE];
    // 运动控制表索引
    int table_idx = 0;
    // 数据库索引
    int idx = 0;
    // 标记是否为第一个数据块
    bool first_block = true;  

    int32_t pos_offset[4] = {0};
    int32_t error[4] = {0};
    int32_t max_error[4] = {0};

    for(int i=0; i<cmd_count; ){
        motion_cmd cmd = cmd_list[i];
        int status = get_pos_from_table(pos_offset, &finger_1_current[idx], &finger_3_current[idx], table_idx, cmd);
        if(status < 0){
            ESP_LOGE(TAG, "未知的运动控制指令0x%x", cmd.cmd);
            return -1;
        }
        data[RIGHT_FINGER_1][idx] = mc.finger_zero[RIGHT] + pos_offset[RIGHT_FINGER_1];
        data[RIGHT_ARM_2][idx] = mc.arm_zero[RIGHT]    + pos_offset[RIGHT_ARM_2];
        data[LEFT_FINGER_3][idx] = mc.finger_zero[LEFT] + pos_offset[LEFT_FINGER_3];
        data[LEFT_ARM_4][idx] = mc.arm_zero[LEFT]    + pos_offset[LEFT_ARM_4];
        idx++;

        // 缓冲区满时发送数据
        if (idx >= BLOCK_SIZE) {
            if(send_data_four_block(data, BLOCK_SIZE, &first_block, 
                                    mcfg.max_current, finger_1_current, finger_3_current,
                                    error) != 0){
                return -1;
            }
            if(check_arm_finger_error(max_error, error) != 0){
                return -1;
            }
            idx = 0;
        }
        // 每隔秒汇报一次当前阶段的最大误差
        if(esp_timer_get_time() - start_time > time_in_second * 1000000){
            time_in_second++;
            print_max_error(max_error);
            memset(max_error, 0, sizeof(max_error));
        }
        // 如果是最后一个数据点，加载下一组数据
        if(status == 1){
            // 更新全局偏移量数据
            mc.finger_offset[RIGHT] = pos_offset[RIGHT_FINGER_1];
            mc.arm_offset[RIGHT]    = pos_offset[RIGHT_ARM_2];
            mc.finger_offset[LEFT]  = pos_offset[LEFT_FINGER_3];
            mc.arm_offset[LEFT]     = pos_offset[LEFT_ARM_4];
            table_idx = 0;
            // 加载下一组数据
            i ++;
        }else{
            table_idx++;
        }
    }

    // 处理剩余数据
    if (idx > 0) {
        if(send_data_four_block(data, idx, &first_block, mcfg.max_current,
                                finger_1_current, finger_3_current, error) != 0){
            return -1;
        }
        if(check_arm_finger_error(max_error, error) != 0){
            return -1;
        }
    }
    // 等待自然停止
    for(int i=1; i<=4; i++){
        int result = wait_for_fifo_space_or_block(i, -1, NULL, &error[i-1], false);
        if(result != WAIT_MOTION_FINISH){
            return -1;
        }
    }
    if(check_arm_finger_error(max_error, error) != 0){
        return -1;
    }
    print_max_error(max_error);
    ESP_LOGI(TAG, "motions()实际耗时%.3fs", (esp_timer_get_time() - start_time) / 1000000.0f);
    return 0;
}

void print_pos(void)
{
    for (int i = 1; i <= 4; i++) {
        int32_t pos;
        cmd_get_pos(i, &pos);
        // 即使没有打印信息，也会更新g_angles[4]
        // ESP_LOGI(TAG, "电机 %d 位置: %ld", i, pos);
    }
}

// 检查通信稳定性
int motion_self_test(void) 
{
    int avg_voltage = 0, avg_temperature = 0;
    int error_count = 0;
    for(int count=0; count<16; count++){
        for (int i = 1; i <= 4; i++) {
            StatResponse resp;
            if (cmd_stat(i, &resp) == 0) {
                avg_voltage += resp.voltage;
                avg_temperature += resp.temperature;
            }else{
                error_count ++;
                ESP_LOGE(TAG, "%d号电机通信不稳定", i);
            }
        }
        // 出现不稳定的情况不要立即退出，4个电机都检查一遍再退出
        if(error_count != 0){
            break;
        }
    }
    if(error_count != 0){
        return -1;
    }else{
        avg_voltage /= 64;
        avg_temperature /= 64;
        ESP_LOGI(TAG, "完成电机通信自检，平均电压%.3fV, 平均温度%d℃", avg_voltage/1000.0f, avg_temperature);

        return 0;
    }
}

// 复位全部电机
int motion_reset_all(void)
{
    int error_count = 0;
    for (int i = 1; i <= 4; i++) {
        if(cmd_reset(i) != 0){
            error_count ++;
            ESP_LOGE(TAG, "复位%d号电机控制板失败，控制板不支持这一功能", i);
        }
    }
    vTaskDelay(pdMS_TO_TICKS(100));
    for (int i = 1; i <= 4; i++) {
        StatResponse resp;
        
        if (cmd_stat(i, &resp) == 0) {
            if(resp.cmd_count != 1){
                error_count ++;
                ESP_LOGE(TAG, "复位%d号电机控制板失败，指令计数没有清零", i);
            }
        }else{
            error_count ++;
            ESP_LOGE(TAG, "复位%d号电机控制板失败，复位后控制板连接中断", i);
        }
    }
    if(error_count == 0){
        ESP_LOGI(TAG, "复位全部电机控制板成功");
    }
    return error_count;
}