#pragma once
// 压缩函数
// 输出数据长度 = 8 + (length - 2) * 3 / 8 + 电流变化的此时 * 6 / 8，向上取整
int compress_data(const int32_t* input, const uint8_t* current, int fixed_current, uint16_t length, 
    uint8_t* compressed, int* compressed_size);

// 缩放函数,用于将查找表缩放到所需的长度（包含平滑）
int scale_table(const int16_t* data, int n, int16_t* new_data, int new_length);