#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "compress.h"

// 压缩函数
// 输出数据长度 = 8 + (length - 2) * 3 / 8 + 电流变化的此时 * 6 / 8，向上取整
// 当length = 512时，电流变化10次时，需要207字节的输出数据缓冲
// 当length = 256时，电流变化10次时，需要111字节的输出数据缓冲

// 增加电流调整指令功能（已完成）
// 当current数组发生变化时，应当在对应的input中的数据之前，发送一个电流调节指令
// 第一个数据，视为电流发生变化
// current数组的大小和input一致
// 电流值的取值范围为0-7，可以使用3bit数据表示
// 电流调节指令的发送方式，为mapped_accel=7后面跟随3bit的电流数据

// 通过compressed_size传入compressed的最大大小，如果超过需要返回错误代码，如果不超过，通过compressed_size输出实际大小
// current == NULL时，使用指定的固定电流值fixed_current
int compress_data(const int32_t* input, const uint8_t* current, int fixed_current, uint16_t length, 
        uint8_t* compressed, int* compressed_size)
{
    if (!input || !compressed || length == 0) {
        return -1;
    }

    // 获取输出缓冲大小
    if(!compressed_size || *compressed_size <= 8){
        return -4;
    }
    int ouput_buffer_size = *compressed_size;

    // 写入数据长度
    memcpy(compressed, &length, 2);
    int byte_index = 2;

    // 写入第一个值
    memcpy(compressed + byte_index, &input[0], 4);
    byte_index += 4;
    
    // 计算并写入第一次差分
    int32_t first_diff = 0;
    if(length > 1){
        first_diff = input[1] - input[0];
    }
    if (first_diff < -32768 || first_diff > 32767) {
        return -2;
    }
    int16_t first_diff_16bit = first_diff;
    memcpy(compressed + byte_index, &first_diff_16bit, 2);
    byte_index += 2;
    
    // 准备位流写入
    uint32_t bit_buffer = 0;
    int bit_count = 0;
    
    int first_current = current ? current[0] : fixed_current;
    // printf("current change to %d\n", first_current);
    // 添加第一个点的电流指令（视为变化）
    bit_buffer = (bit_buffer << 3) | 0x7;  // 特殊标记111
    bit_buffer = (bit_buffer << 3) | (first_current & 0x7);  // 电流值
    bit_count += 6;
    
    // 处理后续的二次差分
    for (uint32_t i = 2; i < length; i++) {
        // 检查电流变化
        if (current && current[i] != current[i-1]) {
            // printf("current change %d -> %d\n", current[i-1], current[i]);
            bit_buffer = (bit_buffer << 3) | 0x7;  // 特殊标记111
            bit_buffer = (bit_buffer << 3) | (current[i] & 0x7);  // 电流值
            bit_count += 6;
        }
        
        int32_t prev_diff = input[i-1] - input[i-2];
        int32_t curr_diff = input[i] - input[i-1];
        int32_t accel = curr_diff - prev_diff;
        
        // 严格检查加速度值是否在-3到3范围内
        if (accel < -3 || accel > 3) {
            // for(int i=0; i<length; i++){
            //     printf("%ld\n", input[i]);
            // }
            return -3;
        }
        
        // 将加速度值映射到0-6范围
        uint8_t mapped_accel = accel + 3;
        
        // 将3位加速度值添加到位缓冲区
        bit_buffer = (bit_buffer << 3) | mapped_accel;
        bit_count += 3;
        
        // 如果位缓冲区中有足够的位形成一个完整的字节，写入压缩数据
        while (bit_count >= 8) {
            if (byte_index >= ouput_buffer_size){
                return -4;
            }
            compressed[byte_index++] = (bit_buffer >> (bit_count - 8)) & 0xFF;
            bit_count -= 8;
            bit_buffer &= (1 << bit_count) - 1;  // 清除已写入的位
        }
    }
    
    // 处理剩余的位
    if (bit_count > 0) {
        if (byte_index >= ouput_buffer_size){
            return -4;
        }
        compressed[byte_index++] = (bit_buffer << (8 - bit_count)) & 0xFF;
    }
    
    *compressed_size = byte_index;
    return 0;
}

// 平滑函数，确保任一点的加速度绝对值不超过max_diff2 = 3
static int smooth_data(int16_t* data, int n, int16_t max_diff2, int max_iterations) {
    if (n <= 2) return 0;  // 数据太短无需平滑

    int status = -1;
    bool changed;

    for (int iter = 0; iter < max_iterations; iter++) {
        changed = false;
        int i = 1;
        while (i < n - 1) {
            // 跳过无效索引（首点0，中间点从1开始）
            if (i < 1) {
                i++;
                continue;
            }
            // 动态计算二次差分（使用int32_t避免溢出）
            int32_t d2 = (int32_t)data[i+1] - 2*(int32_t)data[i] + (int32_t)data[i-1];
            if (abs(d2) > max_diff2) {
                // 计算目标值（四舍五入）
                int16_t target = ((int32_t)data[i-1] + (int32_t)data[i+1] + 1) / 2;
                int16_t adjustment = target - data[i];
                // 限制调整幅度为[-1, 1]
                if (adjustment > 1) adjustment = 1;
                else if (adjustment < -1) adjustment = -1;
                
                data[i] += adjustment;
                changed = true;
                
                // 回退索引以重新检查受影响的点
                if (i > 1) {
                    i -= 2;  // 影响前两个点，回退2步
                } else {
                    i = 0;   // 已到起点，回退到0（下一步i++后变为1）
                }
            }
            i++;  // 移动到下一个点
        }

        // 检查整个数组是否满足条件
        bool satisfied = true;
        for (int j = 0; j < n - 2; j++) {
            int32_t d2_val = (int32_t)data[j+2] - 2*(int32_t)data[j+1] + (int32_t)data[j];
            if (abs(d2_val) > max_diff2) {
                satisfied = false;
                break;
            }
        }
        if (satisfied) {
            status = 0;
            break;
        }

        if (!changed) break;  // 提前终止
    }
    return status;
}

// 缩放函数,用于将查找表缩放到所需的长度（包含平滑）
int scale_table(const int16_t* data, int n, int16_t* new_data, int new_length) 
{
    // 处理特殊情况
    if (new_length == 1) {
        new_data[0] = data[0];
        return 0;
    }
    
    // 线性插值
    for (int j = 0; j < new_length; j++) {
        float x = j * (n - 1.0f) / (new_length - 1);
        int i0 = (int)x;
        float t = x - i0;
        
        if (i0 >= n - 1) {
            new_data[j] = data[n - 1];
        } else {
            float interpolated = data[i0] + t * (data[i0+1] - data[i0]);
            new_data[j] = (int16_t)roundf(interpolated);
        }
    }
    
    // 原地平滑处理
    if (smooth_data(new_data, new_length, 3, 100) != 0) {
        return -1;
    }
    
    return 0;
}