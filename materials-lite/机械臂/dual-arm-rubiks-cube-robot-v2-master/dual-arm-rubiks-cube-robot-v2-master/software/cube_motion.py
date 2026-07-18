import serial
import struct
import time
import logging
import sys
from serial.tools import list_ports

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 通信接口配置
DEFAULT_SERIAL_PORT = '/dev/cu.usbserial-120'
BAUD_RATE = 1000000  # 1Mbps

FINGER_FACTOR = 16384.0 / (2 * 31.416)# 手指电机每转一圈，手指移动距离31.4mm，两手指间距增加2 * 31.4mm

# 运动控制配置 -- 零点
ARM4_ZERO     = 15095  # 左侧旋转臂零点位置
ARM2_ZERO     = 803  # 右侧旋转臂零点位置
OFFSET_FINGER_1_ZERO = round(1.0 * FINGER_FACTOR) # 调整右手指的偏移量
OFFSET_FINGER_3_ZERO = round(0.4 * FINGER_FACTOR) # 调整左手指的偏移量

# TODO 检查一下机械结构，为什么会有偏差？最初比较小，中间拆装过一次变大了。

# 运动控制配置 -- 位置
def finger_dist2enc(x): 
    FINGER_ZERO   = 52.0                  # 零点位于距离内侧限位点0.5mm的位置，此时两手指间距52mm
    return round((x - FINGER_ZERO) * FINGER_FACTOR)
FINGER_CLAMP  = finger_dist2enc(54.0) # 手指锁紧，这是闭环控制的，可填写比实际需求小一点的数字
FINGER_INIT   = finger_dist2enc(56.2) # 手指初始位置，用于放置魔方，52+2*2 = 56mm，与魔方边长保持一致(有的魔方偏大，可以改大点)
FINGER_FLIP_WAIT = finger_dist2enc(57) # 翻转魔方时，手指移动到该位置，就同步开启对侧旋转板的旋转运动(需要比FINGER_INIT大一点，避免出现等待失效的BUG)
FINGER_FLIP   = finger_dist2enc(78.0) # 翻转魔方时，手指移动的最远位置，最小74mm，留4mm余量
FINGER_MAX    = finger_dist2enc(81.0) # 56*√2 = 79.2mm， 魔方是圆角的，实测值是77mm，留4mm余量，取81mm，机械结构最大支持到84mm

# 运动控制配置 -- 速度、加速度、电流
MAX_CURRENT   = 100    # 最大电流
CLAMP_CURRENT = 40     # 夹持魔方时的电流百分比

# 速度比例，调整为1.0为标准还原速度，0.2为1/5速度
# 不要使用过小的值，底层是用int处理的，可能会产生很大的误差
SPEED_FACTOR  = 1.0
SPEED_FACTOR2 = SPEED_FACTOR*SPEED_FACTOR

V_FLIP  = round(200 * SPEED_FACTOR)  # 翻转魔方时的速度
A_FLIP  = round(80 * SPEED_FACTOR2)  # 翻转魔方时的加速度，不宜过大，否则可能只旋转了一层，而不是三层一起旋转

V_TWIST = round(350 * SPEED_FACTOR)  # 拧魔方时的速度，如果魔方的润滑比较好，可用调大一点
A_TWIST = round(300 * SPEED_FACTOR2) # 拧魔方时的加速度，在停转不抖动的前提下，尽可能大

V_NO_LOAD = round(600 * SPEED_FACTOR) # 旋转臂空转最大速度
A_NO_LOAD = round(300 * SPEED_FACTOR2)# 旋转臂空转最大加速度，在停转不抖动的前提下，尽可能大

V_FINGER = round(1500 * SPEED_FACTOR) #手指移动速度，这个惯性小，也不会变形，可用很快
A_FINGER = round(1600 * SPEED_FACTOR2)

# 利用verify_arm_finger_linkage.py计算
V_NO_LOAD_20_70_DEG = round(846 * SPEED_FACTOR)
FINGER_NO_LOAD_START_ARM = finger_dist2enc(63.98)

# 运动控制 -- 超时等待(单位s)
ARM_MOTION_TIME_OUT = 0.5

# ------------------------------- 以下是485通信代码 -------------------------------
def crc8(datagram):
    """计算CRC8校验值，多项式0x07，初始值0x00"""
    crc = 0
    for byte in datagram:
        current_byte = byte
        for _ in range(8):
            bit = (crc >> 7) ^ (current_byte & 0x01)
            if bit:
                crc = (crc << 1) ^ 0x07
            else:
                crc = (crc << 1)
            crc &= 0xFF  # 保持8位
            current_byte >>= 1
    return crc

def build_command_frame(motor_count, ids, command_types, data_list):
    """
    构造主机到电机控制器的数据帧
    :param motor_count: 电机数量（1-8）
    :param ids: 每个电机的ID列表，长度等于motor_count
    :param command_types: 每个电机的指令类型列表
    :param data_list: 每个电机的数据列表，元素为bytes类型
    :return: 完整的字节数据帧
    """
    if motor_count < 1 or motor_count > 8:
        raise ValueError("电机数量必须在1-8之间")
    if len(ids) != motor_count or len(command_types) != motor_count or len(data_list) != motor_count:
        raise ValueError("参数长度与电机数量不匹配")
    
    # 构造所有指令块
    instruction_blocks = []
    for i in range(motor_count):
        id_byte = bytes([ids[i]])
        cmd_byte = bytes([command_types[i]])
        data_bytes = data_list[i]
        block = id_byte + cmd_byte + data_bytes
        instruction_blocks.append(block)
    
    # 合并指令块
    instruction_data = b''.join(instruction_blocks)
    
    # 计算数据长度字段: 2(FF FF) + 1（自身） + 1（电机数量） + 指令块总长度 + 1（CRC）
    data_length = 2 + 1 + 1 + len(instruction_data) + 1
    if data_length > 128:
        raise ValueError("数据长度超过最大限制128字节")
        
    # 构造完整数据帧
    frame = b'\xff\xff'  # 字头
    frame += bytes([data_length])
    frame += bytes([motor_count])
    frame += instruction_data
    crc_value = crc8(frame)
    frame += bytes([crc_value])
    
    return frame

def receive_response(ser, expected_length=15, timeout=1):
    """接收响应数据，并校验结构"""
    start_time = time.time()
    buffer = bytearray()
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            buffer += ser.read(ser.in_waiting)
            # 查找字头0xFF 0xFF
            while len(buffer) >= 2:
                pos = buffer.find(b'\xff\xff')
                if pos == -1:
                    # 没有找到字头，保留最后一个字节继续查找
                    buffer = buffer[-1:] if buffer else bytearray()
                    break
                else:
                    # 找到字头，截取后续数据
                    buffer = buffer[pos:]
                    if len(buffer) < expected_length:
                        # 数据不足，继续等待
                        break
                    else:
                        # 提取完整帧
                        frame = buffer[:expected_length]
                        buffer = buffer[expected_length:]
                        return frame
        time.sleep(0.001)
    return None

def parse_stat_response(frame):
    """解析查询指令的响应数据"""
    if len(frame) != 15:
        raise ValueError("响应帧长度必须为15字节")
    # 提取数据部分和CRC
    data_part = frame[0:14]
    received_crc = frame[14]
    # 计算CRC
    calculated_crc = crc8(data_part)
    if calculated_crc != received_crc:
        raise ValueError(f"CRC校验失败: 计算值{calculated_crc:02X}, 接收值{received_crc:02X}")
    # 解析数据字段
    flag = data_part[3]
    if flag != 0:
        raise ValueError(f"标志位不为零, flag={flag}")
    cmd_count = struct.unpack('<H', data_part[4:6])[0]
    trap_status = data_part[6]
    temperature = struct.unpack('b', data_part[7:8])[0]
    pos = struct.unpack('<i', data_part[8:12])[0]
    voltage = struct.unpack('<h', data_part[12:14])[0]
    return [cmd_count, trap_status, temperature, pos, voltage]

def parse_other_response(response_frame):
    if len(response_frame) != 5:
        raise ValueError("响应帧长度必须为5字节")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("接收到响应帧: %s", response_frame.hex())
    # 计算CRC
    calculated_crc = crc8(response_frame[0:4])
    received_crc = response_frame[4]
    if calculated_crc != received_crc:
        raise ValueError(f"CRC校验失败: 计算值{calculated_crc:02X}, 接收值{received_crc:02X}")
    # 解析数据字段
    flag = response_frame[3]
    if flag != 0:
        raise ValueError(f"标志位不为零, flag={flag}")
    return True

# 使能、禁用编号为id_list的电机，可同步控制多个
# 用法举例，禁用编号为1,2的电机: cmd_enable(ser,[1,2],0)
def cmd_enable(ser, id_list, en):
    # 构造使能指令（0x01）
    data = bytes([en])  # 启用动力
    motor_count = len(id_list) 
    enable_frame = build_command_frame(
        motor_count, 
        id_list, 
        [0x01] * motor_count, 
        [data] * motor_count)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("使能指令数据帧: %s", enable_frame.hex())
    ser.write(enable_frame)

    # 接收响应
    response_frame = receive_response(ser, 5)
    return parse_other_response(response_frame)

def cmd_stat(ser, id):
    # 构造查询指令（0x00）
    motor_count = 1
    ids = [id]
    command_types = [0x00]
    data_list = [bytes()]  # 无数据部分
    stat_frame = build_command_frame(motor_count, ids, command_types, data_list)
    #logger.debug(f"电机编号: {id}, 查询指令数据帧: {stat_frame.hex()}")
    ser.write(stat_frame)

    # 接收并解析响应
    response_frame = receive_response(ser, 15)
    if response_frame:
        #logger.debug(f"接收到响应帧: {response_frame.hex()}")
        parsed_data = parse_stat_response(response_frame)
        # logger.debug(f"[count, trap, temp, pos, voltage]={parsed_data}")
        return parsed_data
    else:
        logger.debug("未接收到响应")
        return None

def cmd_wait_motion(ser, id):
    logger.debug("等待%d号控制板完成运动控制", id)
    start_time = time.time()
    while(True):
        resp = cmd_stat(ser, id)
        pos = resp[3]
        if resp[1] == 0:
            break
    logger.debug("耗时: %.2fms", 1000*(time.time() - start_time))
    return pos

def cmd_get_pos(ser, id):
    resp = cmd_stat(ser, id)
    # logger.debug("获取%d号控制板当前位置%d", id, resp[3])
    return resp[3]

def cmd_trap(ser, id_list, zero, trap_list):
    count = len(id_list)
    cmd_type = 0x03 if zero else 0x02
    data_bytes = [None]*count
    for i in range (count):
        data_bytes[i] = struct.pack('<i', trap_list[i][0])  # int32 x1
        data_bytes[i] += struct.pack('<h', trap_list[i][1]) # int16 v1
        data_bytes[i] += struct.pack('<h', trap_list[i][2]) # int16 vmax
        data_bytes[i] += struct.pack('<h', trap_list[i][3]) # int16 a
        data_bytes[i] += bytes([trap_list[i][4]])           # uint8 max_current
    trap_frame = build_command_frame(count, id_list, [cmd_type]*count, data_bytes)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("梯形运动指令数据帧: %s", trap_frame.hex())
    ser.write(trap_frame)

    # 接收响应
    response_frame = receive_response(ser, 5)
    return parse_other_response(response_frame)

# ------------------------------- 以下是回零点代码 -------------------------------
# 同步带版本和齿轮版本的区别
# 1 手指的控制方向是反的
# 2 齿轮分度圆周长不同
# 3 同步带版本旋转臂带有1:2减速，齿轮版本是1:1的
def cmd_zero(ser):
    # 根据测试情况修改
    current = 15 #电流15%
    accel = 20
    speed_slow = 30
    speed_fast = 100
    # 禁用全部电机4
    cmd_enable(ser, [1,2,3,4], 0)
    # 手指1回零点
    cmd_enable(ser, [1, 2], True)
    finger1_pos_old = cmd_get_pos(ser, 1)
    # 位置，终止速度，最大速度30RPM，加速度，电流current
    cmd_trap(ser, [1], True, [[finger1_pos_old - 10000, 0, speed_slow, accel, current]]) 
    finger1_pos = cmd_wait_motion(ser, 1)
    motion_distance = finger1_pos - finger1_pos_old
    logger.info(f"手指1回零点过程移动距离: {motion_distance}, 目前位置: {finger1_pos}")
    # 手指行程17mm，20齿0.5模齿轮分度圆约31.4mm，17/31.4*16384 = 8870
    if (abs(motion_distance) > 9500):
        logger.error("移动距离过长，回零点失败")
        return None
    finger1_pos += OFFSET_FINGER_1_ZERO
    cmd_trap(ser, [1], False, [[finger1_pos, 0, speed_slow, accel, MAX_CURRENT]]) # 回退0.5mm
    cmd_wait_motion(ser, 1)
    # 旋转臂2回零点
    arm2_pos = cmd_get_pos(ser, 2)
    arm2_zero = ARM2_ZERO + ( - round(ARM2_ZERO / 16384) + round(arm2_pos / 16384)) * 16384
    finger1_pos += (arm2_zero - arm2_pos) // 2
    # 旋转臂带有1:2减速，因此需要双倍速运行
    cmd_trap(ser, [1, 2], False, [[finger1_pos, 0, speed_fast, accel, MAX_CURRENT], 
                                  [arm2_zero, 0, speed_fast*2, accel*2, MAX_CURRENT]])
    cmd_wait_motion(ser, 2)
    arm2_pos = arm2_zero

    # 手指3回零点
    cmd_enable(ser, [3, 4], True)
    finger3_pos_old = cmd_get_pos(ser, 3)
    # 位置，终止速度，最大速度30RPM，加速度，电流current
    cmd_trap(ser, [3], True, [[finger3_pos_old - 10000, 0, speed_slow, accel, current]]) 
    finger3_pos = cmd_wait_motion(ser, 3)
    motion_distance = finger3_pos - finger3_pos_old
    logger.info(f"手指3回零点过程移动距离: {motion_distance}, 目前位置: {finger3_pos}")
    if (abs(motion_distance) > 9500):
        logger.error("移动距离过长，回零点失败")
        return None
    finger3_pos += OFFSET_FINGER_3_ZERO
    cmd_trap(ser, [3], False, [[finger3_pos, 0, speed_slow, accel, MAX_CURRENT]]) # 回退0.5mm
    cmd_wait_motion(ser, 3)
    # 旋转臂4回零点
    arm4_pos = cmd_get_pos(ser, 4)
    arm4_zero = ARM4_ZERO + ( - round(ARM4_ZERO / 16384) + round(arm4_pos / 16384)) * 16384
    finger3_pos += (arm4_zero - arm4_pos) // 2
    cmd_trap(ser, [3, 4], False, [[finger3_pos, 0, speed_fast, accel, MAX_CURRENT], 
                                  [arm4_zero, 0, speed_fast*2, accel*2, MAX_CURRENT]])
    cmd_wait_motion(ser, 4)
    arm4_pos = arm4_zero
    logger.info(f"finger1_pos={finger1_pos}")
    logger.info(f"arm2_pos={arm2_pos}")
    logger.info(f"finger3_pos={finger3_pos}")
    logger.info(f"arm4_pos={arm4_pos}")
    
    return (finger1_pos, arm2_pos, finger3_pos, arm4_pos)

# ------------------------------- 以下是运动控制代码 -------------------------------
LEFT          = True
RIGHT         = False
CW            = True
CCW           = False

class MotionCtrl:
    def __init__(self, ser, finger1_pos, arm2_pos, finger3_pos, arm4_pos):
        self.finger_zero = [finger1_pos, finger3_pos]
        self.arm_zero = [arm2_pos, arm4_pos]

        self.finger_offset = [0, 0]
        self.arm_offset = [0, 0]

        self.ser = ser
        pass

    def check_arm_pos(self, real, expect, id):
        start_time = time.time()
        MAX_ERROR = round(8192 * (2/90)) # 最多允许2°误差
        error = abs(real - expect)
        logger.debug(f"手臂角度误差{error}")
        if error > MAX_ERROR:
            logger.debug(f"手臂角度超差，等待误差合格。当前误差{error}，最大允许{MAX_ERROR}。")
            while True:
                error = abs(cmd_get_pos(self.ser, id) - expect)
                end_time = time.time()
                if error <= MAX_ERROR:
                    logger.debug("耗时: %.2fms", 1000*(end_time - start_time))
                    break
                if end_time - start_time > ARM_MOTION_TIME_OUT:
                    logger.error(f"运动控制超时，可能是出现了电机堵转问题")
                    cmd_enable(self.ser, [1,2,3,4], 0)
                    logger.error(f"关闭全部电机")
                    raise ValueError('运动控制超时')                    

    
    def move_two_finger_raw(self, target, current):
        self.finger_offset[0] = target
        self.finger_offset[1] = target
        # 手臂目标位置 = arm_zero + arm_offset
        # 手指目标位置 = finger_zero + arm_offset / 2 + finger_offset
        finger1 = self.finger_zero[0] + self.arm_offset[0] // 2 + self.finger_offset[0]
        finger3 = self.finger_zero[1] + self.arm_offset[1] // 2 + self.finger_offset[1]
        cmd_trap(self.ser, [1, 3], False, 
                 [[finger1, 0, V_FINGER, A_FINGER, current], 
                  [finger3, 0, V_FINGER, A_FINGER, current]])
        cmd_wait_motion(self.ser, 3)

    def two_finger_init(self):
        self.move_two_finger_raw(FINGER_INIT, MAX_CURRENT)

    def two_finger_clamp(self):
        self.move_two_finger_raw(FINGER_CLAMP, CLAMP_CURRENT)
    
    # def two_finger_max_raw(self):
    #     self.move_two_finger(FINGER_MAX, MAX_CURRENT)
    
    # 旋转机械臂，当wait达到设定角度后返回，如果wait=0，则等待整个控制过程结束再返回
    # angle 只能是90度的整数倍，且不能为0
    def move_arm(self, angle ,left, finger_current, speed, accel, wait = 0):
        if left:
            id_list = [3, 4]
            index = 1
        else:
            id_list = [1, 2]
            index = 0
        # 手臂旋转电机目前位置
        arm_now = self.arm_zero[index] + self.arm_offset[index]
        # 计算函数返回时的电机位置
        arm_ret = self.arm_zero[index] + self.arm_offset[index] + 8192 * (wait / 90.0)
        # 更新手臂位置
        self.arm_offset[index] += 8192 * (angle // 90)
        # 手臂目标位置 = arm_zero + arm_offset
        arm_target = self.arm_zero[index] + self.arm_offset[index]
        # 存在齿轮，所以需要和手臂电机旋转方向相反，转速绝对值相同，才能保证相对静止
        # 手指目标位置 = finger_zero + arm_offset / 2 + finger_offset
        # 计算手指电机的目标位置
        finger_target = self.finger_zero[index] + self.arm_offset[index] // 2 + self.finger_offset[index]

        cmd_trap(self.ser, id_list, False, 
                 [[finger_target, 0, speed, accel, finger_current], 
                  [arm_target, 0, 2*speed, 2*accel, MAX_CURRENT]])
        if wait == 0:
            real_pos = cmd_wait_motion(self.ser, id_list[1])
            self.check_arm_pos(real_pos, arm_target, id_list[1])
        else:
            self.wait_motion_by_pos(id_list[1], arm_now, arm_ret, arm_target)

    def move_arm_without_finger(self, angle ,left, speed, accel):
        if left:
            id_list = [4]
            index = 1
        else:
            id_list = [2]
            index = 0
        # 更新手臂位置
        self.arm_offset[index] += 8192 * (angle // 90)
        # 手臂目标位置 = arm_zero + arm_offset
        arm_target = self.arm_zero[index] + self.arm_offset[index]
        cmd_trap(self.ser, id_list, False, 
                 [[arm_target, 0, 2*speed, 2*accel, MAX_CURRENT]])
        # 等待动作完成
        real_pos = cmd_wait_motion(self.ser, id_list[0])
        self.check_arm_pos(real_pos, arm_target, id_list[0])

    def wait_motion_by_pos(self, id, now, ret, target):
        logger.debug("等待%d号电机超过指定位置", id)
        logger.debug("%d --> %d(在此处返回) --> %d", now, ret, target)
        start_time = time.time()
        while(True):
            pos = cmd_get_pos(self.ser, id)
            end_time = time.time()
            if end_time - start_time > ARM_MOTION_TIME_OUT:
                logger.error(f"运动控制超时，可能是出现了电机堵转问题")
                cmd_enable(self.ser, [1,2,3,4], 0)
                logger.error(f"关闭全部电机")
                raise ValueError('运动控制超时')
            if target > now and pos > ret:
                break
            if target < now and pos < ret:
                break

        logger.debug("耗时: %.2fms", 1000*(end_time - start_time))

    # 手指伸缩，当wait达到设定位置后返回，如果wait=0，则等待整个控制过程结束再返回
    def move_single_finger_raw(self, left, pos, speed, accel, current, wait = 0):
        if left:
            id_list = [3]
            index = 1
        else:
            id_list = [1]
            index = 0
        
        if self.finger_offset[index] == pos:
            logger.error("手指已经处于该位置")
            return

        # 手指电机目前位置
        finger_now = self.finger_zero[index] + self.arm_offset[index] // 2 + self.finger_offset[index]
        # 计算函数返回时的电机位置
        finger_ret = self.finger_zero[index] + self.arm_offset[index] // 2 + wait
        # 计算手指电机的目标位置
        self.finger_offset[index] = pos
        finger_target = self.finger_zero[index] + self.arm_offset[index] // 2 + self.finger_offset[index]
        cmd_trap(self.ser, id_list, False, [[finger_target, 0, speed, accel, current]])
        if wait == 0:
            cmd_wait_motion(self.ser, id_list[0])
        else:
            self.wait_motion_by_pos(id_list[0], finger_now, finger_ret, finger_target)
    def move_finger_lock(self, left):
        self.move_single_finger_raw(left, FINGER_CLAMP, V_FINGER, A_FINGER, CLAMP_CURRENT, 0)

    def move_finger_init(self, left):
        self.move_single_finger_raw(left, FINGER_INIT, V_FINGER, A_FINGER, MAX_CURRENT, 0)
        
    def move_finger_flip(self, left, wait = 0):
        self.move_single_finger_raw(left, FINGER_FLIP, V_FINGER, A_FINGER, MAX_CURRENT, wait)

    def arm_90_no_load(self, left, no_finger_return=False):
        # 只支持固定的方向
        angle = 90
        self.move_finger_init(left)
        # 本侧手指松开（这里还可以优化，可用尝试边松开手指，边空转90度）
        if left:
            id_list = [3]
            index = 1
        else:
            id_list = [1]
            index = 0
        if self.finger_offset[index] != FINGER_INIT:
            logger.warning("finger_offset[index] != FINGER_INIT")
        # 手指电机目前位置
        finger_now    = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_INIT
        # 计算函数返回时的电机位置
        finger_ret    = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_NO_LOAD_START_ARM
        # 计算手指电机的目标位置
        arm_90_deg = round(8192 / 2)
        arm_20_deg = round(8192 * (20/90) / 2)
        arm_70_deg = round(8192 * (70/90) / 2)

        if no_finger_return == False:
            finger_target_stage_1 = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_MAX
            finger_target_stage_1 += arm_20_deg
            finger_target_stage_2 = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_MAX
            finger_target_stage_2 += arm_70_deg
            finger_target_stage_3 = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_INIT
            finger_target_stage_3 += arm_90_deg
            
            cmd_trap(self.ser, id_list, False, 
                    [[finger_target_stage_1, V_NO_LOAD_20_70_DEG//2, V_FINGER, A_FINGER, MAX_CURRENT]])
            cmd_trap(self.ser, id_list, False, 
                    [[finger_target_stage_2, V_NO_LOAD_20_70_DEG//2, V_NO_LOAD, A_NO_LOAD, MAX_CURRENT]])
            cmd_trap(self.ser, id_list, False, 
                    [[finger_target_stage_3, 0                     , V_FINGER, A_FINGER, MAX_CURRENT]])
        else:
            finger_target_stage_1 = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_MAX
            finger_target_stage_1 += arm_20_deg
            finger_target_stage_2 = self.finger_zero[index] + self.arm_offset[index] // 2 + FINGER_MAX
            finger_target_stage_2 += arm_90_deg
            self.finger_offset[index] = FINGER_MAX

            cmd_trap(self.ser, id_list, False, 
                    [[finger_target_stage_1, V_NO_LOAD_20_70_DEG//2, V_FINGER, A_FINGER, MAX_CURRENT]])
            cmd_trap(self.ser, id_list, False, 
                    [[finger_target_stage_2, 0                     , V_NO_LOAD, A_NO_LOAD, MAX_CURRENT]])
        # 等待手指到达指定位置
        self.wait_motion_by_pos(id_list[0], finger_now, finger_ret, finger_target_stage_1)
        # 空转90度
        self.move_arm_without_finger(angle, left, V_NO_LOAD, A_NO_LOAD)
        if no_finger_return == False:
            # 等待手指归位
            cmd_wait_motion(self.ser, id_list[0])
            # 手指锁紧
            self.move_finger_lock(left)

    def motions(self, actions):
        i = 0 
        while i < len(actions):
            start_time = time.time()
            action = actions[i]
            i += 1
            # action[0]取值范围L、R
            if action[0] == 'L':
                left = LEFT
            elif action[0] == 'R':
                left = RIGHT
            else:
                logger.error(f"未知指令 {action}")
                return
            # action[1]取值范围0、1、2、+、-、*
            if action[1] == '0':
                # 夹爪夹紧操作，如果去掉move_finger_init，也可以还原，但是稳定性稍差，时间能减少10ms左右
                self.move_finger_init(left)
                self.move_finger_lock(left)
            elif action[1] == '1':
                # 夹爪张开，下一步一定是翻面操作，松到与魔方表面齐平就可以下一步了
                self.move_finger_flip(left, FINGER_FLIP_WAIT)
            elif action[1] == '2':
                # 夹爪张开最大角度，这个涉及同侧电机联动问题，2-3条指令合并处理
                # 加载后面的两条指令
                if i < len(actions):
                    action_arm = actions[i] # R+N
                else:
                    logger.error(f"{action}不能位于序列末尾")
                    return
                if i+1 < len(actions):
                    action_finger = actions[i+1]
                else:
                    action_finger = 'XXX'

                if action_arm[1:] == '+N':
                    if action_finger[1:] == '0':
                        self.arm_90_no_load(left, False) # 完成后手指夹紧
                        i += 2
                    else:
                        self.arm_90_no_load(left, True)  # 完成后手指松开
                        i += 1
                else:
                    logger.error(f"{action}的下一条指令必须为R+N, 实际为{action_arm}")
                    return

            elif action[1] in ('+', '-', '*'):
                op = action[1:]
                if  op == '*T':
                    # 转动-180°，等到旋转-170°时，开始手指归位操作
                    self.move_arm(-180, left, CLAMP_CURRENT, V_TWIST, A_TWIST, -170)
                elif op == '+T':
                    # 转90°，等到旋转80°时，开始手指归位操作
                    self.move_arm(90, left, CLAMP_CURRENT, V_TWIST, A_TWIST, 80)
                elif op == '-T':
                    # 转-90°，等到旋转-80°时，开始手指归位操作
                    self.move_arm(-90, left, CLAMP_CURRENT, V_TWIST, A_TWIST, -80)
                elif op == '*F':
                    # 旋转-180°
                    self.move_arm(-180, left, CLAMP_CURRENT, V_FLIP, A_FLIP)
                elif op == '+F':
                    # 转90°
                    self.move_arm(90, left, CLAMP_CURRENT, V_FLIP, A_FLIP)
                elif op == '-F':
                    # 转-90°
                    self.move_arm(-90, left, CLAMP_CURRENT, V_FLIP, A_FLIP)
                else:
                    logger.error(f"未知指令 {action}")
                    return
            else:
                logger.error(f"未知指令 {action}")
                return
            logger.info(f"序号{i}，处理指令{action}，耗时{1000 * (time.time() - start_time):.1f}ms")

            

# ------------------------------- 以下是测试程序 -------------------------------
def list_serial_ports():
    """列出可用串口"""
    ports = list_ports.comports()
    if not ports:
        print("未检测到可用串口设备")
        return
    print("可用串口设备:")
    for port in ports:
        print(f"  {port.device} - {port.description}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        serial_port = sys.argv[1]
    else:
        serial_port = DEFAULT_SERIAL_PORT  # 默认值

    mc = None
    try:
        with serial.Serial(serial_port, baudrate = BAUD_RATE, bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser0:
            print("\n串口连接成功，进入交互模式")
            while True:
                print("\n请选择测试项目:")
                print("[1]: 使能全部电机")
                print("[2]: 禁用全部电机")
                print("[3]: 查询电机角度")
                print("[4]: 回零点(需要先回零点才能执行其他的！)")
                print("[5]: 松开魔方")
                print("[6]: 夹紧魔方")
                print("[7]: 预留")
                print("[8]: 测试旋转臂动作")
                print("[9]: 打乱魔方再还原")
                print("[q]: 退出程序")
                
                choice = input("请输入选项(1/2/q/...) >> ").strip().lower()
                
                if choice in ['exit', 'quit', 'q']:
                    print("退出程序...")
                    break
                elif choice == '1':
                    success = cmd_enable(ser0, [1,2,3,4], 1)
                    print("执行结果:", "成功" if success else "失败")
                elif choice == '2':
                    success = cmd_enable(ser0, [1,2,3,4], 0)
                    print("执行结果:", "成功" if success else "失败")
                elif choice == '3':
                    exit = False
                    pos = [None] * 4
                    for id in (1,2,3,4):
                        resp = cmd_stat(ser0, id)
                        if resp == None:
                            print(f"未收到控制器回复，控制器编号={id}")
                            exit = True
                        else:
                            pos[id-1] = resp[3]
                    print(f"当前电机角度: {pos}")
                elif choice == '4':
                    zero = cmd_zero(ser0)
                    mc = MotionCtrl(ser0, zero[0], zero[1], zero[2], zero[3])
                    mc.two_finger_init()
                elif choice == '5':
                    mc.two_finger_init()
                elif choice == '6':
                    mc.two_finger_clamp()
                elif choice == '7':
                    pass
                elif choice == '8':
                    mc.two_finger_clamp()
                    mc.motions(['R1', 'L*F', 'R0',   'L1', 'R+F', 'L0',   'R2', 'R+N', 'R0'])
                    mc.motions(['L1', 'R*F', 'L0',   'R1', 'L+F', 'R0',   'L2', 'L+N', 'L0'])
                    mc.two_finger_init()
                elif choice == '9':
                    mc.two_finger_clamp()
                    scramble_string_a = "L1 R-F L0 R2 R+N R0 L+T R1 L+F R0 R-T R2 R+N R0 L*T L1 R-F L0 R2 R+N R0 L-T L2 L+N R*F L0 L-T L2 L+N L0 R*T L1 R*F L0 L*T R*T L1 R+F L0 R2 R+N R0 L*T R*T R1 L-F R0 L2 L+N L0 R-T R2 R+N R0 L*T L1 R-F L0 R2 R+N R0 L*T R-T R2 R+N R0 L-T R1 L+F R0 R-T R2 R+N L+F R0 L2 L+N L0 R*T L-T R1 L-F R0 R-T L1 R*F L0 R2 R+N R0 L+T L2 L+N L0 R+T R2 R+N R0"
                    time_start = time.time()
                    mc.motions(scramble_string_a.split(' '))
                    logger.info(f"time = {time.time() - time_start:.2f}s")
                    mc.two_finger_init()
                else:
                    print("无效选项，请重新输入")


    except serial.SerialException as e:
        logger.error(f"打开串口 {serial_port} 失败: {str(e)}")
        list_serial_ports()
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")