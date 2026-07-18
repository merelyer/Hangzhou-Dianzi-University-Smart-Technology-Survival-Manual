#pragma once

/**
 * @brief 初始化RS485通信接口
 * 
 * 配置UART参数（1M波特率，8数据位，无校验），设置半双工模式，
 * 启用内部上拉电阻，安装UART驱动并设置超时参数。
 */
void init_rs485(void);

/**
 * @brief 使能所有电机
 * 
 * 向电机ID 1-4发送使能指令(0x01)，开启电机闭环控制。
 * 成功时记录日志，失败时记录错误。
 */
void enable_all(void);

/**
 * @brief 禁用所有电机
 * 
 * 向电机ID 1-4发送禁用指令(0x00)，关闭电机闭环控制。
 * 成功时记录日志，失败时记录错误。
 */
void disable_all(void);

/**
 * @brief 打印电机当前位置
 * 
 * 查询并记录ID 1-4电机的实时编码器位置，
 * 通过串口日志输出位置数据。
 */
void print_pos(void);

/**
 * @brief 执行电机回零校准流程
 * 
 * 包含两阶段操作：
 * 1. 手指1+旋转臂2联合校准
 * 2. 手指3+旋转臂4联合校准
 * 校准后初始化运动控制结构体(mc)，设置各轴零点偏移量。
 * 
 * @return int 成功返回0，失败返回-1（含错误日志）
 */
int cmd_zero(void);

/**
 * @brief 执行魔方还原动作序列
 * 
 * 解析动作指令字符串（如"L0 R1 R*T"）并执行：
 * - L0/R0: 单侧手指初始化+夹紧
 * - L1/R1: 单侧手指翻转预备
 * - L2/R2: 空转90°（需后接+N指令）
 * - *T/+T/-T: 带手指联动的旋转动作（180°/90°/-90°）
 * - *F/+F/-F: 魔方翻转动作（180°/90°/-90°）
 * 
 * @param actions 动作指令字符串（空格分隔）
 * @return int 成功返回0，语法错误/执行失败返回-1（含错误日志）
 */
int motions(const char *actions);

/**
 * @brief 双指初始化位置
 * 
 * 将两侧手指同步移动到预设初始位置(FINGER_INIT)，
 * 用于放置魔方（间距≈56mm）。
 * 
 * @return int 成功返回0，失败返回-1
 */
int two_finger_init(void);

/**
 * @brief 双指夹紧位置
 * 
 * 将两侧手指同步移动到预设夹紧位置(FINGER_CLAMP)，
 * 用于固定魔方（间距≈54mm）。
 * 
 * @return int 成功返回0，失败返回-1
 */
int two_finger_clamp(void);

/**
 * @brief 加载配置文件
 * @param path 文件路径
 * @return int 成功返回0，失败返回-1
 */
int load_motion_cfg(const char *path);

int motion_self_test(void);
int motion_reset_all(void);
int get_time_motions_str(const char *actions);
int32_t *get_g_angle(void);