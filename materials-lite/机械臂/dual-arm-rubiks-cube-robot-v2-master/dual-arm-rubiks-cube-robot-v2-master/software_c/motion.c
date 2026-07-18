// gcc motion.c -o motion -Wall -lserialport -I /opt/homebrew/include -L /opt/homebrew/lib
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <libserialport.h>

// 最大电机数量，用于分配内存
const int MAX_MOTOR_COUNT = 4;
// 日志配置
#define LOG_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)

// 通信接口配置
const char DEFAULT_SERIAL_PORT[] = "/dev/cu.usbserial-120";
const int BAUD_RATE = 1000000;  // 1Mbps
const float FINGER_FACTOR = 16384.0f / (2 * 31.416f); // 手指电机每转一圈，手指移动距离31.4mm，两手指间距增加2 * 31.4mm

// 运动控制配置 -- 零点
#define ROUND(x) ((int)(x + 0.5f))
const int ARM4_ZERO     = 15095;                             // 左侧旋转臂零点位置
const int ARM2_ZERO     = 803;                               // 右侧旋转臂零点位置
const int OFFSET_FINGER_1_ZERO = ROUND(1.0f * FINGER_FACTOR); // 调整右手指的偏移量
const int OFFSET_FINGER_3_ZERO = ROUND(0.4f * FINGER_FACTOR); // 调整左手指的偏移量

// 运动控制配置 -- 位置
#define FINGER_DIST2ENC(x) ROUND((x - 52.0f) * FINGER_FACTOR) // 零点位于距离内侧限位点0.5mm的位置，此时两手指间距52mm
const int FINGER_CLAMP  = FINGER_DIST2ENC(54.0f);    // 手指锁紧，这是闭环控制的，可填写比实际需求小一点的数字
const int FINGER_INIT   = FINGER_DIST2ENC(56.2f);    // 手指初始位置，用于放置魔方，52+2*2 = 56mm，与魔方边长保持一致(有的魔方偏大，可以改大点)
const int FINGER_FLIP_WAIT = FINGER_DIST2ENC(57.0f); // 翻转魔方时，手指移动到该位置，就同步开启对侧旋转板的旋转运动(需要比FINGER_INIT大一点，避免出现等待失效的BUG)
const int FINGER_FLIP   = FINGER_DIST2ENC(78.0f);    // 翻转魔方时，手指移动的最远位置，最小74mm，留4mm余量
const int FINGER_MAX    = FINGER_DIST2ENC(81.0f);    // 56*√2 = 79.2mm， 魔方是圆角的，实测值是77mm，留4mm余量，取81mm，机械结构最大支持到84mm

// 运动控制配置 -- 速度、加速度、电流
const int MAX_CURRENT   = 100;    // 最大电流
const int CLAMP_CURRENT = 40;     // 夹持魔方时的电流百分比

// 速度比例，调整为1.0为标准还原速度，0.2为1/5速度
// 不要使用过小的值，底层是用int处理的，可能会产生很大的误差
const float SPEED_FACTOR  = 1.0f;
const float SPEED_FACTOR2 = SPEED_FACTOR*SPEED_FACTOR;

const int V_FLIP  = ROUND(200 * SPEED_FACTOR);  // 翻转魔方时的速度
const int A_FLIP  = ROUND(80 * SPEED_FACTOR2);  // 翻转魔方时的加速度，不宜过大，否则可能只旋转了一层，而不是三层一起旋转

const int V_TWIST = ROUND(350 * SPEED_FACTOR);  // 拧魔方时的速度，如果魔方的润滑比较好，可用调大一点
const int A_TWIST = ROUND(300 * SPEED_FACTOR2); // 拧魔方时的加速度，在停转不抖动的前提下，尽可能大

const int V_NO_LOAD = ROUND(600 * SPEED_FACTOR); // 旋转臂空转最大速度
const int A_NO_LOAD = ROUND(300 * SPEED_FACTOR2);// 旋转臂空转最大加速度，在停转不抖动的前提下，尽可能大

const int V_FINGER = ROUND(1500 * SPEED_FACTOR); //手指移动速度，这个惯性小，也不会变形，可用很快
const int A_FINGER = ROUND(1600 * SPEED_FACTOR2);

// 利用verify_arm_finger_linkage.py计算
const int V_NO_LOAD_20_70_DEG = ROUND(846 * SPEED_FACTOR);
const int FINGER_NO_LOAD_START_ARM = FINGER_DIST2ENC(63.98f);

// 运动控制 -- 超时等待(单位s)
const float ARM_MOTION_TIME_OUT = 0.5f;
const float ARM_TRAP_TIME_OUT = 5.0f;

// 方向定义
const bool LEFT = true;
const bool RIGHT = false;
const bool CW = true;
const bool CCW = false;

// 定义梯形运动参数结构体
typedef struct {
    int32_t x1;        // 目标位置
    int16_t v1;        // 终止速度
    int16_t vmax;      // 最大速度
    int16_t a;         // 加速度
    uint8_t current;   // 电流百分比
} TrapParam;

// 运动控制结构体
typedef struct {
    bool init;// = true表示已经初始化
    int32_t finger_zero[2]; // [0]: finger1, [1]: finger3
    int32_t arm_zero[2];    // [0]: arm2, [1]: arm4
    int32_t finger_offset[2]; // [0]: finger1, [1]: finger3
    int32_t arm_offset[2];    // [0]: arm2, [1]: arm4
} MotionCtrl;

// 串口句柄
struct sp_port *serial_port = NULL;
// 运动控制结构体，存储电机零点和当前位置信息
MotionCtrl mc = {0};


// 获取当前时间(毫秒)
static double get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
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

/* 构造主机到电机控制器的数据帧
:param motor_count: 电机数量（1-8）
:param ids: 每个电机的ID列表，长度等于motor_count
:param command_types: 每个电机的指令类型列表
:param data_list: 每个电机的数据列表，元素为bytes类型
:return: 完整的字节数据帧
*/
static int build_command_frame(uint8_t motor_count, const uint8_t *ids, 
                        const uint8_t *command_types, 
                        const uint8_t **data_list, const int *data_lengths,
                        uint8_t *output, int output_size) {
    if (motor_count < 1 || motor_count > 8) {
        LOG_ERROR("电机数量必须在1-8之间");
        return -1;
    }
    
    // 计算指令块总长度
    int instruction_length = 0;
    for (int i = 0; i < motor_count; i++) {
        instruction_length += 1 + 1 + data_lengths[i]; // ID + 命令类型 + 数据
    }
    
    // 计算总长度: 2(FF FF) + 1(长度) + 1(电机数量) + 指令块长度 + 1(CRC)
    int total_length = 2 + 1 + 1 + instruction_length + 1;
    if (total_length > output_size) {
        LOG_ERROR("输出缓冲区太小");
        return -1;
    }
    // 构建帧
    uint8_t *ptr = output;
    *ptr++ = 0xFF;
    *ptr++ = 0xFF;
    *ptr++ = (uint8_t)total_length;
    *ptr++ = motor_count;
    
    // 添加指令块
    for (int i = 0; i < motor_count; i++) {
        *ptr++ = ids[i];
        *ptr++ = command_types[i];
        memcpy(ptr, data_list[i], data_lengths[i]);
        ptr += data_lengths[i];
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
        int n = sp_blocking_read(serial_port, &byte, 1, 50);  // 最长50ms等待
        if (n != 1) {
            LOG_ERROR("读取串口失败");
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

// 解析状态响应
static int parse_stat_response(const uint8_t *frame, size_t length, 
                        uint16_t *cmd_count, uint8_t *trap_status, 
                        int8_t *temperature, int32_t *pos, int16_t *voltage) {
    if (length != 15) {
        LOG_ERROR("状态响应长度应为15字节");
        return -1;
    }
    
    // 检查CRC
    uint8_t calculated_crc = crc8(frame, 14);
    if (calculated_crc != frame[14]) {
        LOG_ERROR("CRC校验失败: 计算值%02X, 接收值%02X", calculated_crc, frame[14]);
        return -1;
    }
    
    // 检查标志位
    if (frame[3] != 0) {
        LOG_ERROR("从机数据标志位不为零: %d", frame[3]);
        return -1;
    }
    
    // 解析字段
    *cmd_count = (frame[5] << 8) | frame[4];
    *trap_status = frame[6];
    *temperature = (int8_t)frame[7];
    *pos = (frame[11] << 24) | (frame[10] << 16) | (frame[9] << 8) | frame[8];
    *voltage = (frame[13] << 8) | frame[12];
    
    return 0;
}
static int parse_other_response(const uint8_t *frame, size_t length) {
    if (length != 5) {
        LOG_ERROR("状态响应长度应为5字节");
        return -1;
    }
    
    // 检查CRC
    uint8_t calculated_crc = crc8(frame, 4);
    if (calculated_crc != frame[4]) {
        LOG_ERROR("CRC校验失败: 计算值%02X, 接收值%02X", calculated_crc, frame[4]);
        return -1;
    }
    
    // 检查标志位
    if (frame[3] != 0) {
        LOG_ERROR("从机数据标志位不为零: %d", frame[3]);
        return -1;
    }
    
    return 0;
}
// 发送使能命令(最多MAX_MOTOR_COUNT个电机)
// ff ff 11 04 01 01 00 02 01 00 03 01 00 04 01 00 c9
static int cmd_enable(const uint8_t *id_list, int count, uint8_t enable) {
    uint8_t enable_data = enable;
    const uint8_t *data_list[MAX_MOTOR_COUNT];
    int data_lengths[MAX_MOTOR_COUNT];
    uint8_t command_types[MAX_MOTOR_COUNT];
    
    for (int i = 0; i < count; i++) {
        data_list[i] = &enable_data;
        data_lengths[i] = 1;
        command_types[i] = 0x01; // 使能命令
    }
    
    uint8_t frame[5 + 3*MAX_MOTOR_COUNT];
    int frame_length = build_command_frame((uint8_t)count, id_list, command_types, 
                                         (const uint8_t **)data_list, data_lengths, 
                                         frame, sizeof(frame));
    
    if (frame_length < 0) {
        return -1;
    }

    // 发送数据
    int bytes_written = sp_blocking_write(serial_port, frame, frame_length, 0);
    if (bytes_written != frame_length) {
        LOG_ERROR("发送使能命令失败，写入 %d/%d 字节", bytes_written, frame_length);
    }
    
    // 接收响应
    uint8_t response[5];
    int response_length = receive_response(response, sizeof(response));
    if (response_length < 5) {
        LOG_ERROR("接收响应失败");
        return -1;
    }
    
    // 检查响应
    if (parse_other_response(response, response_length) != 0) {
        LOG_ERROR("响应标志位错误: %d", response[3]);
        return -1;
    }
    
    return 0;
}

// 查询状态
static int cmd_stat(uint8_t id, uint16_t *cmd_count, uint8_t *trap_status, 
             int8_t *temperature, int32_t *pos, int16_t *voltage) {
    uint8_t ids[] = {id};
    uint8_t command_types[] = {0x00}; // 查询命令
    const uint8_t *data_list[] = {NULL};
    int data_lengths[] = {0};
    
    uint8_t frame[5 + 3*MAX_MOTOR_COUNT];
    int frame_length = build_command_frame(1, ids, command_types, 
                                         data_list, data_lengths, 
                                         frame, sizeof(frame));
    if (frame_length < 0) {
        return -1;
    }
    
    // 发送数据
    int bytes_written = sp_blocking_write(serial_port, frame, frame_length, 0);
    if (bytes_written != frame_length) {
        LOG_ERROR("发送查询命令失败，写入 %d/%d 字节", bytes_written, frame_length);
        return -1;
    }
    
    // 接收响应
    uint8_t response[15];
    int response_length = receive_response(response, sizeof(response));
    if (response_length < 15) {
        LOG_ERROR("接收响应失败");
        return -1;
    }
    
    return parse_stat_response(response, response_length, 
                              cmd_count, trap_status, temperature, pos, voltage);
}

// 获取当前位置
int32_t cmd_get_pos(uint8_t id) {
    uint16_t cmd_count;
    uint8_t trap_status;
    int8_t temperature;
    int32_t pos;
    int16_t voltage;
    
    if (cmd_stat(id, &cmd_count, &trap_status, &temperature, &pos, &voltage) == 0) {
        // LOG_INFO("cmd_count=%d trap_status=%d temperature=%d pos=%d voltage=%d\n",
        //     cmd_count, trap_status, temperature, pos, voltage);
        return pos;
    }
    return 0;
}

// 等待运动完成
int32_t cmd_wait_motion(uint8_t id) {
    LOG_DEBUG("等待%d号控制板完成运动控制", id);
    double start_time = get_current_time_ms();
    double current_time = start_time;
    uint8_t trap_status;
    
    do {
        uint16_t cmd_count;
        int8_t temperature;
        int32_t pos;
        int16_t voltage;
        
        if (cmd_stat(id, &cmd_count, &trap_status, &temperature, &pos, &voltage) != 0) {
            return 0;
        }
        current_time = get_current_time_ms();
        if (trap_status == 0) {
            LOG_DEBUG("耗时: %.2fms", current_time - start_time);
            return pos;
        }

    } while (current_time - start_time < ARM_TRAP_TIME_OUT * 1000);
    
    LOG_ERROR("等待运动超时");
    return 0;
}

// 发送梯形运动命令
static int cmd_trap(const uint8_t *id_list, int count, bool zero, const TrapParam *trap_list) 
{
    // 准备数据
    const uint8_t *data_list[MAX_MOTOR_COUNT];
    int data_lengths[MAX_MOTOR_COUNT];
    uint8_t command_types[MAX_MOTOR_COUNT];
    uint8_t motor_data[MAX_MOTOR_COUNT][11]; // 每个电机的数据缓冲区
    
    for (int i = 0; i < count; i++) {
        // 打包数据: x1(4), v1(2), vmax(2), a(2), current(1)
        uint8_t *ptr = motor_data[i];
        *ptr++ = (trap_list[i].x1) & 0xFF;
        *ptr++ = (trap_list[i].x1 >> 8) & 0xFF;
        *ptr++ = (trap_list[i].x1 >> 16) & 0xFF;
        *ptr++ = (trap_list[i].x1 >> 24) & 0xFF;
        
        *ptr++ = trap_list[i].v1 & 0xFF;
        *ptr++ = (trap_list[i].v1 >> 8) & 0xFF;
        
        *ptr++ = trap_list[i].vmax & 0xFF;
        *ptr++ = (trap_list[i].vmax >> 8) & 0xFF;
        
        *ptr++ = trap_list[i].a & 0xFF;
        *ptr++ = (trap_list[i].a >> 8) & 0xFF;
        
        *ptr++ = trap_list[i].current;
        
        data_list[i] = motor_data[i];
        data_lengths[i] = 11;
        command_types[i] = zero ? 0x03 : 0x02; // 回零或绝对位置指令
    }
    
    // 构建帧
    uint8_t frame[5 + 3*MAX_MOTOR_COUNT + MAX_MOTOR_COUNT*11];
    int frame_length = build_command_frame((uint8_t)count, id_list, command_types, 
                                          data_list, data_lengths, 
                                          frame, sizeof(frame));
    if (frame_length < 0) {
        return -1;
    }
    
    // 发送数据
    int bytes_written = sp_blocking_write(serial_port, frame, frame_length, 0);
    if (bytes_written != frame_length) {
        LOG_ERROR("发送梯形命令失败，写入 %d/%d 字节", bytes_written, frame_length);
        return -1;
    }
    
    // 接收响应
    uint8_t response[5];
    int response_length = receive_response(response, sizeof(response));
    if (response_length < 5) {
        LOG_ERROR("接收梯形响应失败");
        return -1;
    }
    
    return parse_other_response(response, response_length);
}

static int cmd_zero(void) 
{
    if (!serial_port) {
        LOG_ERROR("串口未打开");
        return -1;
    }
    // 配置参数
    const uint8_t current = 15; // 电流15%
    const int16_t accel = 20;
    const int16_t speed_slow = 30;
    const int16_t speed_fast = 100;
    
    // 禁用全部电机
    if (cmd_enable((uint8_t[]){1, 2, 3, 4}, 4, 0) != 0) {
        LOG_ERROR("禁用电机失败");
        return -1;
    }
    
    // ----------------- 手指1回零 -----------------
    // 同时使能手指1和旋转臂2
    if (cmd_enable((uint8_t[]){1, 2}, 2, 1) != 0) {
        LOG_ERROR("使能手指1和旋转臂2失败");
        return -1;
    }
    
    int32_t finger1_pos_old = cmd_get_pos(1);    
    // 创建运动参数 - 回零运动
    TrapParam finger1_zero_param = {
        .x1 = finger1_pos_old - 10000,
        .v1 = 0,
        .vmax = speed_slow,
        .a = accel,
        .current = current
    };
    
    // 发送梯形运动命令 - 只控制手指1回零
    if (cmd_trap((uint8_t[]){1}, 1, true, &finger1_zero_param) != 0) {
        LOG_ERROR("手指1回零运动失败");
        return -1;
    }
    
    int32_t current_finger1_pos = cmd_wait_motion(1);
    int32_t motion_distance = current_finger1_pos - finger1_pos_old;
    LOG_INFO("手指1回零移动距离: %d, 当前位置: %d", motion_distance, current_finger1_pos);
    
    if (abs(motion_distance) > 9500) {
        LOG_ERROR("手指1移动距离过长，回零失败");
        return -1;
    }
    
    // 补偿偏移
    current_finger1_pos += OFFSET_FINGER_1_ZERO;
    TrapParam finger1_adj_param = {
        .x1 = current_finger1_pos,
        .v1 = 0,
        .vmax = speed_slow,
        .a = accel,
        .current = MAX_CURRENT
    };
    
    // 只控制手指1调整位置
    if (cmd_trap((uint8_t[]){1}, 1, false, &finger1_adj_param) != 0) {
        LOG_ERROR("手指1位置调整失败");
        return -1;
    }
    cmd_wait_motion(1);
    
    // 获取旋转臂2当前位置
    int32_t arm2_old = cmd_get_pos(2);
    
    // 计算旋转臂2目标位置（考虑多圈）
    int32_t arm2_zero_pos = ARM2_ZERO;
    int32_t arm2_zero_cycles = (int32_t)roundf((float)ARM2_ZERO / 16384.0f);
    int32_t arm2_current_cycles = (int32_t)roundf((float)arm2_old / 16384.0f);
    arm2_zero_pos += (arm2_current_cycles - arm2_zero_cycles) * 16384;
    
    // 计算手指1补偿量
    int32_t finger1_target = current_finger1_pos + (arm2_zero_pos - arm2_old) / 2;
    
    // 准备多轴运动 - 同时控制手指1和旋转臂2
    TrapParam multi_params[2] = {
        { // 手指1
            .x1 = finger1_target,
            .v1 = 0,
            .vmax = speed_fast,
            .a = accel,
            .current = MAX_CURRENT
        },
        { // 旋转臂2
            .x1 = arm2_zero_pos,
            .v1 = 0,
            .vmax = speed_fast * 2,  // 两倍速度
            .a = accel * 2,          // 两倍加速度
            .current = MAX_CURRENT
        }
    };
    
    // 同时控制两个电机
    if (cmd_trap((uint8_t[]){1, 2}, 2, false, multi_params) != 0) {
        LOG_ERROR("多轴运动失败");
        return -1;
    }
    cmd_wait_motion(2);  // 等待两个电机都完成
    
    // ----------------- 手指3回零 -----------------
    // 同时使能手指3和旋转臂4
    if (cmd_enable((uint8_t[]){3, 4}, 2, 1) != 0) {
        LOG_ERROR("使能手指3和旋转臂4失败");
        return -1;
    }
    
    int32_t finger3_pos_old = cmd_get_pos(3);
    // 只控制手指3回零
    TrapParam finger3_zero_param = {
        .x1 = finger3_pos_old - 10000,
        .v1 = 0,
        .vmax = speed_slow,
        .a = accel,
        .current = current
    };
    
    if (cmd_trap((uint8_t[]){3}, 1, true, &finger3_zero_param) != 0) {
        LOG_ERROR("手指3回零失败");
        return -1;
    }
    
    int32_t current_finger3_pos = cmd_wait_motion(3);
    motion_distance = current_finger3_pos - finger3_pos_old;
    LOG_INFO("手指3回零移动距离: %d, 当前位置: %d", motion_distance, current_finger3_pos);
    
    if (abs(motion_distance) > 9500) {
        LOG_ERROR("手指3移动距离过长，回零失败");
        return -1;
    }
    
    // 补偿偏移
    current_finger3_pos += OFFSET_FINGER_3_ZERO;
    TrapParam finger3_adj_param = {
        .x1 = current_finger3_pos,
        .v1 = 0,
        .vmax = speed_slow,
        .a = accel,
        .current = MAX_CURRENT
    };
    
    // 只控制手指3调整位置
    if (cmd_trap((uint8_t[]){3}, 1, false, &finger3_adj_param) != 0) {
        LOG_ERROR("手指3位置调整失败");
        return -1;
    }
    cmd_wait_motion(3);
    
    // 获取旋转臂4当前位置
    int32_t arm4_old = cmd_get_pos(4);
    
    // 计算旋转臂4目标位置（考虑多圈）
    int32_t arm4_zero_pos = ARM4_ZERO;
    int32_t arm4_zero_cycles = (int32_t)roundf((float)ARM4_ZERO / 16384.0f);
    int32_t arm4_current_cycles = (int32_t)roundf((float)arm4_old / 16384.0f);
    arm4_zero_pos += (arm4_current_cycles - arm4_zero_cycles) * 16384;
    
    // 计算手指3补偿量
    int32_t finger3_target = current_finger3_pos + (arm4_zero_pos - arm4_old) / 2;
    
    // 准备多轴运动 - 同时控制手指3和旋转臂4
    TrapParam multi_params2[2] = {
        { // 手指3
            .x1 = finger3_target,
            .v1 = 0,
            .vmax = speed_fast,
            .a = accel,
            .current = MAX_CURRENT
        },
        { // 旋转臂4
            .x1 = arm4_zero_pos,
            .v1 = 0,
            .vmax = speed_fast * 2,  // 两倍速度
            .a = accel * 2,          // 两倍加速度
            .current = MAX_CURRENT
        }
    };
    
    // 同时控制两个电机
    if (cmd_trap((uint8_t[]){3, 4}, 2, false, multi_params2) != 0) {
        LOG_ERROR("多轴运动失败");
        return -1;
    }
    cmd_wait_motion(4);  // 等待两个电机都完成
    
    
    LOG_INFO("回零点成功:");
    LOG_INFO("  手指1位置: %d", finger1_target);
    LOG_INFO("  旋转臂2位置: %d", arm2_zero_pos);
    LOG_INFO("  手指3位置: %d", finger3_target);
    LOG_INFO("  旋转臂4位置: %d", arm4_zero_pos);

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

// 移动两个手指
int move_two_finger_raw(int32_t target, uint8_t current) {
    if (!serial_port) {
        LOG_ERROR("串口未打开");
        return -1;
    }
    if (!mc.init) {
        LOG_ERROR("需要先调用cmd_zero");
        return -1;
    }
    mc.finger_offset[0] = target;
    mc.finger_offset[1] = target;
    
    int32_t finger1 = mc.finger_zero[0] + mc.arm_offset[0] / 2 + mc.finger_offset[0];
    int32_t finger3 = mc.finger_zero[1] + mc.arm_offset[1] / 2 + mc.finger_offset[1];
    
    // 准备多轴运动 - 同时控制手指1和手指3
    TrapParam multi_params2[2] = {
        { // 手指1
            .x1 = finger1,
            .v1 = 0,
            .vmax = V_FINGER,
            .a = A_FINGER,
            .current = current
        },
        { // 手指3
            .x1 = finger3,
            .v1 = 0,
            .vmax = V_FINGER,
            .a = A_FINGER,
            .current = current
        }
    };
    
    // 同时控制两个电机
    if (cmd_trap((uint8_t[]){1, 3}, 2, false, multi_params2) != 0) {
        return -1;
    }
    cmd_wait_motion(3);
    return 0;
}

// 手指初始化位置
int two_finger_init(void) {
    return move_two_finger_raw(FINGER_INIT, MAX_CURRENT);
}

// 手指夹紧
int two_finger_clamp(void) {
    return move_two_finger_raw(FINGER_CLAMP, CLAMP_CURRENT);
}

// 检查手臂位置是否在允许误差范围内
static void check_arm_pos(uint8_t id, int32_t real, int32_t expect) {
    double start_time = get_current_time_ms();
    const int MAX_ERROR = round(8192 * (2.0/90.0)); // 最多允许2°误差
    int32_t error = abs(real - expect);
    LOG_DEBUG("手臂角度误差 %d", error);
    
    if (error > MAX_ERROR) {
        LOG_DEBUG("手臂角度超差，等待误差合格。当前误差%d，最大允许%d", error, MAX_ERROR);
        while (true) {
            int32_t current_pos = cmd_get_pos(id);
            error = abs(current_pos - expect);
            double current_time = get_current_time_ms();
            
            if (error <= MAX_ERROR) {
                LOG_DEBUG("耗时: %.2fms", current_time - start_time);
                break;
            }
            
            if (difftime(current_time, start_time) > ARM_MOTION_TIME_OUT) {
                LOG_ERROR("运动控制超时，可能是出现了电机堵转问题");
                uint8_t ids[] = {1, 2, 3, 4};
                cmd_enable(ids, 4, 0);
                LOG_ERROR("关闭全部电机");
                exit(1); // 严重错误，退出程序
            }
        }
    }
}
// 等待电机位置超过指定点
static void wait_motion_by_pos(uint8_t id, int32_t now, int32_t ret, int32_t target) {
    LOG_DEBUG("等待电机 %d 超过指定位置", id);
    LOG_DEBUG("%d --> %d(在此处返回) --> %d", now, ret, target);
    double start_time = get_current_time_ms();
    
    while (true) {
        int32_t pos = cmd_get_pos(id);
        double current_time = get_current_time_ms();
        
        if (current_time - start_time > ARM_MOTION_TIME_OUT * 1000) {
            LOG_ERROR("运动控制超时");
            uint8_t ids[] = {1, 2, 3, 4};
            cmd_enable(ids, 4, 0);
            LOG_ERROR("关闭全部电机");
            exit(1);
        }
        
        if (target > now && pos > ret) break;
        if (target < now && pos < ret) break;
    }
    
    LOG_DEBUG("耗时: %.2fms", get_current_time_ms() - start_time);
}
// 旋转机械臂（带手指联动）
static void move_arm(int angle, bool left, uint8_t finger_current, int speed, int accel, int wait) {
    uint8_t id_list[2];
    int index;
    
    if (left) {
        id_list[0] = 3; // 手指
        id_list[1] = 4; // 手臂
        index = 1;
    } else {
        id_list[0] = 1; // 手指
        id_list[1] = 2; // 手臂
        index = 0;
    }
    
    // 手臂旋转电机目前位置
    int32_t arm_now = mc.arm_zero[index] + mc.arm_offset[index];
    // 计算函数返回时的电机位置
    int32_t arm_ret = mc.arm_zero[index] + mc.arm_offset[index] + 8192 * (wait / 90.0);
    // 更新手臂位置
    mc.arm_offset[index] += 8192 * (angle / 90);
    // 手臂目标位置
    int32_t arm_target = mc.arm_zero[index] + mc.arm_offset[index];
    // 手指目标位置
    int32_t finger_target = mc.finger_zero[index] + mc.arm_offset[index] / 2 + mc.finger_offset[index];

    TrapParam params[2] = {
        { // 手指电机
            .x1 = finger_target,
            .v1 = 0,
            .vmax = speed,
            .a = accel,
            .current = finger_current
        },
        { // 手臂电机
            .x1 = arm_target,
            .v1 = 0,
            .vmax = 2 * speed,
            .a = 2 * accel,
            .current = MAX_CURRENT
        }
    };
    
    cmd_trap(id_list, 2, false, params);
    
    if (wait == 0) {
        int32_t real_pos = cmd_wait_motion(id_list[1]);
        check_arm_pos(id_list[1], real_pos, arm_target);
    } else {
        wait_motion_by_pos(id_list[1], arm_now, arm_ret, arm_target);
    }
}

// 单独移动手臂（不带手指联动）
static void move_arm_without_finger(int angle, bool left, int speed, int accel) {
    uint8_t id;
    int index;
    
    if (left) {
        id = 4;
        index = 1;
    } else {
        id = 2;
        index = 0;
    }
    
    // 更新手臂位置
    mc.arm_offset[index] += 8192 * (angle / 90);
    // 手臂目标位置
    int32_t arm_target = mc.arm_zero[index] + mc.arm_offset[index];
    
    TrapParam param = {
        .x1 = arm_target,
        .v1 = 0,
        .vmax = 2 * speed,
        .a = 2 * accel,
        .current = MAX_CURRENT
    };
    
    cmd_trap(&id, 1, false, &param);
    int32_t real_pos = cmd_wait_motion(id);
    check_arm_pos(id, real_pos, arm_target);
}

// 移动单根手指
static void move_single_finger_raw(bool left, int32_t pos, int speed, int accel, uint8_t current, int wait) {
    uint8_t id;
    int index;
    
    if (left) {
        id = 3;
        index = 1;
    } else {
        id = 1;
        index = 0;
    }
    
    if (mc.finger_offset[index] == pos) {
        LOG_ERROR("手指已经处于该位置");
        return;
    }
    
    // 手指电机当前位置
    int32_t finger_now = mc.finger_zero[index] + mc.arm_offset[index] / 2 + mc.finger_offset[index];
    // 计算返回时的位置
    int32_t finger_ret = mc.finger_zero[index] + mc.arm_offset[index] / 2 + wait;
    // 更新手指位置
    mc.finger_offset[index] = pos;
    // 手指目标位置
    int32_t finger_target = mc.finger_zero[index] + mc.arm_offset[index] / 2 + mc.finger_offset[index];
    
    TrapParam param = {
        .x1 = finger_target,
        .v1 = 0,
        .vmax = speed,
        .a = accel,
        .current = current
    };
    
    cmd_trap(&id, 1, false, &param);
    
    if (wait == 0) {
        cmd_wait_motion(id);
    } else {
        wait_motion_by_pos(id, finger_now, finger_ret, finger_target);
    }
}

// 手指锁紧
static void move_finger_lock(bool left) {
    move_single_finger_raw(left, FINGER_CLAMP, V_FINGER, A_FINGER, CLAMP_CURRENT, 0);
}

// 手指初始化位置
static void move_finger_init(bool left) {
    move_single_finger_raw(left, FINGER_INIT, V_FINGER, A_FINGER, MAX_CURRENT, 0);
}

// 手指翻转到指定位置
static void move_finger_flip(bool left, int wait) {
    move_single_finger_raw(left, FINGER_FLIP, V_FINGER, A_FINGER, MAX_CURRENT, wait);
}

// 手臂空转90度
static void arm_90_no_load(bool left, bool no_finger_return) {
    const int angle = 90;
    move_finger_init(left);
    
    uint8_t id;
    int index;
    
    if (left) {
        id = 3;
        index = 1;
    } else {
        id = 1;
        index = 0;
    }
    
    if (mc.finger_offset[index] != FINGER_INIT) {
        LOG_ERROR("finger_offset 不等于 FINGER_INIT");
    }
    
    // 手指当前位置
    int32_t finger_now = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_INIT;
    // 计算返回位置
    int32_t finger_ret = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_NO_LOAD_START_ARM;
    // 计算角度对应的位置
    int32_t arm_90_deg = round(8192.0 / 2);
    int32_t arm_20_deg = round(8192.0 * (20.0/90.0) / 2);
    int32_t arm_70_deg = round(8192.0 * (70.0/90.0) / 2);
    
    int32_t finger_target_stage1, finger_target_stage2, finger_target_stage3;
    
    if (!no_finger_return) {
        finger_target_stage1 = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_MAX + arm_20_deg;
        finger_target_stage2 = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_MAX + arm_70_deg;
        finger_target_stage3 = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_INIT + arm_90_deg;
        
        TrapParam param1 = {
            .x1 = finger_target_stage1,
            .v1 = V_NO_LOAD_20_70_DEG / 2,
            .vmax = V_FINGER,
            .a = A_FINGER,
            .current = MAX_CURRENT
        };
        cmd_trap(&id, 1, false, &param1);
        
        TrapParam param2 = {
            .x1 = finger_target_stage2,
            .v1 = V_NO_LOAD_20_70_DEG / 2,
            .vmax = V_NO_LOAD,
            .a = A_NO_LOAD,
            .current = MAX_CURRENT
        };
        cmd_trap(&id, 1, false, &param2);
        
        TrapParam param3 = {
            .x1 = finger_target_stage3,
            .v1 = 0,
            .vmax = V_FINGER,
            .a = A_FINGER,
            .current = MAX_CURRENT
        };
        cmd_trap(&id, 1, false, &param3);
    } else {
        finger_target_stage1 = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_MAX + arm_20_deg;
        finger_target_stage2 = mc.finger_zero[index] + mc.arm_offset[index] / 2 + FINGER_MAX + arm_90_deg;
        mc.finger_offset[index] = FINGER_MAX;
        
        TrapParam param1 = {
            .x1 = finger_target_stage1,
            .v1 = V_NO_LOAD_20_70_DEG / 2,
            .vmax = V_FINGER,
            .a = A_FINGER,
            .current = MAX_CURRENT
        };
        cmd_trap(&id, 1, false, &param1);
        
        TrapParam param2 = {
            .x1 = finger_target_stage2,
            .v1 = 0,
            .vmax = V_NO_LOAD,
            .a = A_NO_LOAD,
            .current = MAX_CURRENT
        };
        cmd_trap(&id, 1, false, &param2);
    }
    
    // 等待手指到达指定位置
    wait_motion_by_pos(id, finger_now, finger_ret, finger_target_stage1);
    // 空转90度
    move_arm_without_finger(angle, left, V_NO_LOAD, A_NO_LOAD);
    
    if (!no_finger_return) {
        cmd_wait_motion(id);
        move_finger_lock(left);
    }
}
// 修改后的motions函数
static int motions(const char *actions) {
    if (!serial_port) {
        LOG_ERROR("串口未打开");
        return -1;
    }
    if (!mc.init) {
        LOG_ERROR("需要先调用cmd_zero");
        return -1;
    }
    
    const char *p = actions;
    // int i = 0;
    
    while (*p != '\0') {
        double start_time = get_current_time_ms();
        // i++;
        
        // 跳过空格
        while (*p == ' ') p++;
        if (*p == '\0') break;
        
        // 读取当前指令（两个字符）
        char action[4] = {p[0], p[1], '\0', '\0'};
        p += 2;
        
        // 处理当前指令
        bool left;
        if (action[0] == 'L') {
            left = LEFT;
        } else if (action[0] == 'R') {
            left = RIGHT;
        } else {
            LOG_ERROR("未知指令: %s", action);
            return -1;
        }
        
        // 根据指令类型处理
        if (action[1] == '0') {
            move_finger_init(left);
            move_finger_lock(left);
        } 
        else if (action[1] == '1') {
            move_finger_flip(left, FINGER_FLIP_WAIT);
        } 
        else if (action[1] == '2') {
            // 跳过空格，预读下一条指令
            while (*p == ' ') p++;
            if (*p == '\0') {
                LOG_ERROR("%s不能位于序列末尾", action);
                return -1;
            }
            
            char next_action[3] = {p[0], p[1], '\0'};
            
            // 检查下一条指令是否为"+N"格式
            if (next_action[0] != action[0] || next_action[1] != '+' || p[2] != 'N') {
                LOG_ERROR("%s的下一条指令必须为%c+N, 实际为%s", action, action[0], next_action);
                return -1;
            }
            
            // 继续读取+N后的部分
            p += 3; // 跳过下一条指令（三个字符：R+N）
            
            // 跳过空格，尝试读取下下条指令
            while (*p == ' ') p++;
            
            bool close_finger = false;
            if (*p != '\0') {
                // 检查下下条指令是否为"X0"格式
                char next_next_action[3] = {p[0], p[1], '\0'};
                
                // 如果是指令末尾的夹紧操作
                if (next_next_action[0] == action[0] && next_next_action[1] == '0') {
                    close_finger = true;
                    p += 2; // 跳过后面的夹紧指令
                }
            }
            
            arm_90_no_load(left, !close_finger);
        } 
        else if (action[1] == '+' || action[1] == '-' || action[1] == '*') {
            // 检查是否完整动作指令
            action[2] = p[0];
            if (action[2] == 'T' || action[2] == 'F') {
                // 如果动作指令被分割，移动指针
                if (action[1] != '+' && action[1] != '-' && action[1] != '*') {
                    LOG_ERROR("无效指令格式: %c%c%c", action[0], action[1], p[0]);
                    return -1;
                }
                p++; // 跳过最后一个字符
                
                const char *op = &action[1];
                if (strncmp(op, "*T", 2) == 0) {
                    move_arm(-180, left, CLAMP_CURRENT, V_TWIST, A_TWIST, -170);
                } 
                else if (strncmp(op, "+T", 2) == 0) {
                    move_arm(90, left, CLAMP_CURRENT, V_TWIST, A_TWIST, 80);
                } 
                else if (strncmp(op, "-T", 2) == 0) {
                    move_arm(-90, left, CLAMP_CURRENT, V_TWIST, A_TWIST, -80);
                } 
                else if (strncmp(op, "*F", 2) == 0) {
                    move_arm(-180, left, CLAMP_CURRENT, V_FLIP, A_FLIP, 0);
                } 
                else if (strncmp(op, "+F", 2) == 0) {
                    move_arm(90, left, CLAMP_CURRENT, V_FLIP, A_FLIP, 0);
                } 
                else if (strncmp(op, "-F", 2) == 0) {
                    move_arm(-90, left, CLAMP_CURRENT, V_FLIP, A_FLIP, 0);
                } 
                else {
                    LOG_ERROR("未知指令: %s", action);
                    return -1;
                }
            } 
            else {
                LOG_ERROR("无效动作指令: %s", action);
                return -1;
            }
        } 
        else {
            LOG_ERROR("未知指令: %s", action);
            return -1;
        }
        
        double elapsed = get_current_time_ms() - start_time;
        LOG_INFO("处理指令 %s, 耗时: %.1fms", action, elapsed);
    }
    return 0;
}

// 主函数 - 交互测试
int main(int argc, char *argv[]) {
    const char *serial_port_name = (argc > 1) ? argv[1] : DEFAULT_SERIAL_PORT;
    
    // 打开串口
    if (sp_get_port_by_name(serial_port_name, &serial_port) != SP_OK) {
        LOG_ERROR("找不到串口设备: %s", serial_port_name);
        return -1;
    }
    
    if (sp_open(serial_port, SP_MODE_READ_WRITE) != SP_OK) {
        LOG_ERROR("无法打开串口: %s", serial_port_name);
        sp_free_port(serial_port);
        return -1;
    }
    
    // 配置串口
    sp_set_baudrate(serial_port, BAUD_RATE);
    sp_set_bits(serial_port, 8);
    sp_set_parity(serial_port, SP_PARITY_NONE);
    sp_set_stopbits(serial_port, 1);
    sp_set_flowcontrol(serial_port, SP_FLOWCONTROL_NONE);
    
    LOG_INFO("串口 %s 已打开，波特率 %d", serial_port_name, BAUD_RATE);
    
    int running = 1;
    
    while (running) {
        printf("\n请选择测试项目:\n");
        printf("[1]: 使能全部电机\n");
        printf("[2]: 禁用全部电机\n");
        printf("[3]: 查询电机角度\n");
        printf("[4]: 回零点\n");
        printf("[5]: 松开魔方\n");
        printf("[6]: 夹紧魔方\n");
        printf("[7]: 预留\n");
        printf("[8]: 测试旋转臂动作\n");
        printf("[9]: 打乱魔方再还原\n");
        printf("[q]: 退出程序\n");
        
        printf("> ");
        char choice[10];
        fgets(choice, sizeof(choice), stdin);
        
        switch (choice[0]) {
            case '1': {
                // 使能指令数据帧: ffff110401010102010103010104010177
                // 接收到响应帧: ffff0500e2
                uint8_t ids[] = {1, 2, 3, 4};
                if (cmd_enable(ids, 4, 1) == 0) {
                    LOG_INFO("使能成功");
                } else {
                    LOG_ERROR("使能失败");
                }
                break;
            }
            case '2': {
                // 使能指令数据帧: ffff1104010100020100030100040100c9
                // 接收到响应帧: ffff0500e2
                uint8_t ids[] = {1, 2, 3, 4};
                if (cmd_enable(ids, 4, 0) == 0) {
                    LOG_INFO("禁用成功");
                } else {
                    LOG_ERROR("禁用失败");
                }
                break;
            }
            case '3': {
                for (int i = 1; i <= 4; i++) {
                    int32_t pos = cmd_get_pos(i);
                    LOG_INFO("电机 %d 位置: %d", i, pos);
                }
                break;
            }
            case '4': {
                if (cmd_zero() == 0) {
                    two_finger_init();
                    LOG_INFO("回零点成功");
                } else {
                    LOG_ERROR("回零点失败");
                }
                break;
            }
            case '5':
                two_finger_init();
                LOG_INFO("魔方已松开");
                break;
            case '6':
                two_finger_clamp();
                LOG_INFO("魔方已夹紧");
                break;
            case '8':
                {
                    const char action_1[] = "R1 L*F R0 L1 R+F L0 R2 R+N R0";
                    const char action_2[] = "L1 R*F L0 R1 L+F R0 L2 L+N L0";
                    two_finger_clamp();
                    motions(action_1);
                    motions(action_2);
                    two_finger_init();
                } 
                break;
            case '9':
            {
                const char action_1[] = "R+T R2 R+N L-F R0 L+T L1 R-F L0 R+T L+T R1 L+F R0 R*T L+T L2 L+N R+F L0 R+T L-T R1 L*F R0 L2 L+N L0 R-T R2 R+N L+F R0 L2 L+N L0 R*T R1 L+F R0 L2 L+N L0 R*T L*T L1 R-F L0 R2 R+N R0 L-T L2 L+N L0 R*T R1 L+F R0 L-T R*T R1 L+F R0 L2 L+N L0 R*T R1 L*F R0 R*T L*T R*T R1 L+F R0 L2 L+N L0 R*T";
                two_finger_clamp();
                double start_time = get_current_time_ms();
                motions(action_1);
                double elapsed = get_current_time_ms() - start_time;
                LOG_INFO("总计耗时: %.2fs", elapsed/1000);
                two_finger_init();
            } 
                break;
            case 'q':
                running = 0;
                break;
            default:
                printf("无效选项\n");
        }
    }
    
    // 清理
    sp_close(serial_port);
    sp_free_port(serial_port);
    LOG_INFO("程序已退出");
    
    return 0;
}


