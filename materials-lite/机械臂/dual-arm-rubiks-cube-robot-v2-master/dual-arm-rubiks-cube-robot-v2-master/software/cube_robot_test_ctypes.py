#!/usr/bin/python3
# 有个问题，如果在vscode下直接运行这个程序，有可能会无法正常拖动圆圈，将实时预览窗口切换最小化再切回就能恢复
# 这个可能是输入焦点不在实时预览窗口引起的
# 在终端下运行没有这个问题（偶发的，没解决思路，也不影响使用）

import cube_motion
# import cube_color_detect
import serial
import sys
import time
import logging
import cv2
import threading
import queue
import os
import json
import traceback
#import cube_optimizer
import ctypes
import platform
import numpy as np

# 配置文件路径
POINTS_CONFIG_FILE = "points_config.json"
# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 根据平台加载动态库，如果失败，启用备用的python版本程序
system = platform.system()
c_lib_name = "../software_c/libcube"
if system == "Windows":
    c_lib_name += ".dll"
elif system == "Linux":
    c_lib_name += ".so"
elif system == "Darwin":
    c_lib_name += ".dylib"
else:
    raise OSError("Unsupported platform")
# ------------------------魔方求解函数的接口-------------------------------
try:
    lib = ctypes.cdll.LoadLibrary(c_lib_name)
    # 设置函数原型
    lib.solve_motion.argtypes = [
        ctypes.c_char_p,  # facelets
        ctypes.c_char_p,  # initial_state
        ctypes.c_char_p   # output_sequence
    ]
    lib.solve_motion.restype = ctypes.c_int
except Exception as e:
    logger.warning(f"加载{c_lib_name}失败：{str(e)}")

def solve_cube(facelets):
    start_time_solve = time.time()
    # 创建足够大的缓冲区接收结果
    output_buf = ctypes.create_string_buffer(1024)
    
    # 调用C函数
    initial_state = "RU"
    result = lib.solve_motion(
        facelets.encode('utf-8'),
        initial_state.encode('utf-8'),
        output_buf
    )
    
    if result < 0:
        logger.error(f"不可解的魔方 - {result}")
        motion_sequence = None
    else:
        motion_sequence = output_buf.value.decode('utf-8').strip().split(' ')
    logger.info(f"魔方求解耗时: {(time.time() - start_time_solve)*1000:.2f}ms")
    return motion_sequence

# ------------------------颜色识别函数的接口-------------------------------
# 定义Point结构体
class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int16),
                ("y", ctypes.c_int16)]

# 设置color_detect_img函数原型
lib.color_detect_img.argtypes = [
    ctypes.c_int,              # width
    ctypes.c_int,              # height
    ctypes.POINTER(ctypes.c_uint8),  # img1 data pointer
    ctypes.POINTER(ctypes.c_uint8),  # img2 data pointer
    ctypes.POINTER(ctypes.c_uint8),  # img3 data pointer
    ctypes.POINTER(Point),           # point_xy array
    ctypes.c_char_p                  # cube_str output buffer
]
lib.color_detect_img.restype = ctypes.c_int

def detect_cube_color(images, points):
    """
    调用C函数检测魔方颜色
    
    :param images: 包含三张图像(BGR格式)的列表，每张图像为numpy数组
    :param points: 6个点的坐标列表，每个点为[x, y]
    :return: 54个字符的魔方状态字符串
    """
    start_time_detect = time.time()
    # 检查输入参数
    if len(images) != 3:
        raise ValueError("需要3张图像")
    
    if len(points) != 6:
        raise ValueError("需要6个点")
    
    # 获取图像尺寸
    height, width, _ = images[0].shape
    
    # 创建Point数组
    points_arr = (Point * 6)()
    for i, pt in enumerate(points):
        points_arr[i].x = pt[0]
        points_arr[i].y = pt[1]
    
    # 将图像数据转换为连续的一维数组(确保连续内存)
    img_data = []
    for img in images:
        if img.shape[:2] != (height, width):
            raise ValueError("所有图像必须具有相同的尺寸")
        if img.dtype != np.uint8:
            raise ValueError("图像数据类型必须为uint8")
        img_data.append(img.astype(np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    
    # 创建输出缓冲区(55字节，用于存放54字符+结束符)
    cube_str_buf = ctypes.create_string_buffer(55)
    
    # 调用C函数
    result = lib.color_detect_img(
        width, height,
        img_data[0],
        img_data[1],
        img_data[2],
        points_arr,
        cube_str_buf
    )
    
    if result != 0:
        logger.error(f"颜色检测失败，错误代码: {result}")
        return None
    
    logger.info(f"魔方颜色识别耗时: {(time.time() - start_time_detect)*1000:.2f}ms")
    # 返回前54个字符(实际魔方状态)
    return cube_str_buf.value[:54].decode('utf-8')

# 设置avg_color_detect_img函数原型
lib.avg_color_detect_img.argtypes = [
    ctypes.c_int,              # width
    ctypes.c_int,              # height
    ctypes.POINTER(ctypes.c_uint8),  # bgr_image data pointer
    ctypes.POINTER(Point),           # point_xy array (6 points)
    ctypes.POINTER(ctypes.c_int)     # output: color_hs (18*2的数组，按顺序存放18个色块的H和S)
]
lib.avg_color_detect_img.restype = ctypes.c_int

def get_avg_color_hs(frame, points):
    """
    调用C函数获取18个色块的平均色相和饱和度
    
    :param frame: 一帧图像(BGR格式)
    :param points: 6个点的坐标列表，每个点为[x, y]
    :return: 包含18个色块[色相, 饱和度]的列表，每个元素是包含两个整数的列表
    """
    # 检查输入参数
    if len(points) != 6:
        raise ValueError("需要6个点")
    
    # 获取图像尺寸
    height, width = frame.shape[:2]
    
    # 创建Point数组
    points_arr = (Point * 6)()
    for i, pt in enumerate(points):
        points_arr[i].x = pt[0]
        points_arr[i].y = pt[1]
    
    # 创建输出缓冲区 (18个色块 * 2个值)
    color_hs_buf = (ctypes.c_int * (18 * 2))()
    
    # 获取图像数据指针
    bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    # 调用C函数
    result = lib.avg_color_detect_img(
        width, height,
        bgr_ptr,
        points_arr,
        color_hs_buf
    )
    
    if result != 0:
        logger.error(f"平均颜色检测失败，错误代码: {result}")
        return None
    
    # 将结果转换为二维列表
    color_hs = []
    for i in range(18):
        h = color_hs_buf[i*2]
        s = color_hs_buf[i*2+1]
        color_hs.append([h, s])
    
    return color_hs

# 定义工作线程类
class MotionWorker(threading.Thread):
    def __init__(self, mc, save_queue, tweak_queue):
        super().__init__()
        self.mc = mc
        self.save_queue = save_queue
        self.tweak_queue = tweak_queue
        self._stop_event = threading.Event()
    
    def run(self):
        try:
            logger.info("工作线程开始执行动作序列")
            self.mc.two_finger_clamp()
            # 在不安装魔方的情况下，调整延时，使爪尖图像清晰不模糊
            delay = 0.02
            # 请求保存图像1
            time.sleep(delay)
            self.save_queue.put(1)
            time.sleep(delay)
            self.mc.motions(['R1', 'L*F', 'R0',   'L1', 'R+F', 'L0',   'R2', 'R+N', 'R0'])
            
            # 请求保存图像2
            time.sleep(delay)
            self.save_queue.put(2)
            time.sleep(delay)

            self.mc.motions(['L1', 'R*F', 'L0',   'R1', 'L+F', 'R0',   'L2', 'L+N', 'L0'])
            
            # 请求保存图像3
            time.sleep(delay)
            self.save_queue.put(3)
            time.sleep(delay)
            
            # 等待识别结果
            logger.info("等待识别结果...")
            result = self.tweak_queue.get()  # 阻塞直到获取结果
            if result is None:
                logger.warning("未收到识别结果，跳过旋转操作")
            else:
                logger.info(f"收到识别结果，开始旋转操作: {result}")
                try:
                    self.mc.motions(result)
                except Exception as e:
                    logger.error(f"旋转操作出错: {str(e)}")
            
            # 最后初始化位置
            self.mc.two_finger_init()
        except Exception as e:
            logger.error(f"工作线程发生错误: {str(e)}")
        finally:
            self.save_queue.put(None)  # 发送完成信号
    
    def stop(self):
        self._stop_event.set()

# 定义可拖动圆圈的类
class DraggablePoints:
    def __init__(self, frame_width, frame_height):
        self.points = []
        self.radius = 20
        self.dragging_index = -1
        
        # 尝试从配置文件加载点坐标
        if os.path.exists(POINTS_CONFIG_FILE):
            try:
                with open(POINTS_CONFIG_FILE, 'r') as f:
                    loaded_points = json.load(f)
                    # 验证加载的数据
                    if isinstance(loaded_points, list) and len(loaded_points) == 6:
                        for pt in loaded_points:
                            if (isinstance(pt, list) and len(pt) == 2 and 
                                isinstance(pt[0], int) and isinstance(pt[1], int)):
                                self.points.append((pt[0], pt[1]))
                        if len(self.points) == 6:
                            logger.info(f"已从 {POINTS_CONFIG_FILE} 加载点坐标")
                            return
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        
        # 如果没有配置文件或加载失败，使用默认位置 (两行三列)
        logger.info("使用默认点坐标")
        for i in range(2):
            for j in range(3):
                x = int((j + 0.5) * frame_width / 3)
                y = int((i + 0.5) * frame_height / 2)
                self.points.append((x, y))
    
    def draw(self, frame):
        # 绘制连线
        connections = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        
        # 绘制所有连线
        for (start, end) in connections:
            cv2.line(frame, self.points[start], self.points[end], (0, 255, 0), 2)
        
        # 绘制所有圆圈和编号
        for i, (x, y) in enumerate(self.points):
            color = (0, 0, 255) if i == self.dragging_index else (255, 0, 0)
            cv2.circle(frame, (x, y), self.radius, color, -1)
            cv2.putText(frame, str(i+1), (x-10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def check_drag(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击了某个圆圈
            for i, (px, py) in enumerate(self.points):
                if ((x - px) ** 2 + (y - py) ** 2) <= self.radius ** 2:
                    self.dragging_index = i
                    return True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_index >= 0:
            # 更新被拖动的圆圈位置
            self.points[self.dragging_index] = (x, y)
            return True
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_index = -1
            return True
        
        return False
    
    def print_points(self):
        print("当前六个小圆圈的坐标：")
        for i, (x, y) in enumerate(self.points):
            print(f"点 {i+1}: ({x}, {y})")
        print("")
    
    def save_points(self):
        """将当前点坐标保存到文件"""
        try:
            # 将点坐标转换为可序列化的格式
            points_list = [[int(x), int(y)] for x, y in self.points]
            with open(POINTS_CONFIG_FILE, 'w') as f:
                json.dump(points_list, f)
            logger.info(f"点坐标已保存到 {POINTS_CONFIG_FILE}")
        except Exception as e:
            logger.error(f"保存点坐标失败: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        serial_port = sys.argv[1]
    else:
        serial_port = cube_motion.DEFAULT_SERIAL_PORT  # 默认值
    
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
        
    # 配置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 设置长度
    #cap.set(cv2.CAP_PROP_AUTO_WB, 0)          # 手动白平衡
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 手动曝光 (0.25表示手动模式)        

    # 获取第一帧以确定尺寸
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，退出...")
        cap.release()
        exit()
    
    # 创建可拖动点对象
    draggable_points = DraggablePoints(frame.shape[1], frame.shape[0])

    save_path_template = "../temp/captured_image_{}.jpg"
    # 确保目录存在
    filename = save_path_template.format(1)
    save_dir = os.path.dirname(filename)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"创建目录: {save_dir}")

    # 创建线程间通信队列
    save_queue = queue.Queue()
    tweak_queue = queue.Queue()
    motion_worker = None

    # 创建窗口并设置鼠标回调
    cv2.namedWindow('Camera Preview')
    
    # 修复: 使用全局变量或闭包解决作用域问题
    def mouse_callback(event, x, y, flags, param):
        # 使用全局变量访问draggable_points
        global draggable_points
        if draggable_points.check_drag(event, x, y):
            draggable_points.print_points()
    
    cv2.setMouseCallback('Camera Preview', mouse_callback)
    
    # 初始化时间显示相关变量
    start_time = None
    total_time = None
    
    # ------------------------自动检测魔方放置的算法-------------------------------
    # 定义状态
    STATE_INIT = 0          # 初始状态，舍弃前10帧
    STATE_BACKGROUND = 1    # 采集背景颜色
    STATE_DETECTING = 2     # 检测魔方放置
    STATE_STABILIZING = 3   # 检测稳定性
    STATE_WORKING = 4       # 工作线程运行中
    STATE_DETECTING_REMOVE = 5  # 检测魔方移除
    
    # 阈值设置
    BACKGROUND_DISCARD_FRAMES = 10  # 舍弃的前10帧
    STABILIZATION_FRAMES = 10       # 需要的稳定帧数
    STABILIZATION_PERCENT = 0.15    # 0.15%的像素发生变化
    THRESH_DIFF = 40                # 像素变化阈值
    SAT_HUE_THRESHOLD = 30          # 变化阈值
    CHANGE_THRESHOLD = 9            # 变化色块数量阈值
    CHANGE_THRESHOLD_REMOVE = 2     # 变化色块数量阈值

    # 状态变量
    state = STATE_INIT
    frame_counter = 0
    background_hs = None            # 背景颜色数据
    prev_hs = None                  # 前一帧的颜色数据
    stable_counter = 0              # 稳定帧计数器
    last_change_value = 0           # 记录上一次的变化量
    prev_gray = None
    
    def start_work_thread():
        """启动工作线程的封装函数"""
        global motion_worker, start_time, total_time
        logger.info("启动工作线程执行动作序列")
        # 重置时间变量
        start_time = time.time()
        total_time = None
        motion_worker = MotionWorker(mc, save_queue, tweak_queue)
        motion_worker.daemon = True  # 设置为守护线程
        motion_worker.start()

    try:
        with serial.Serial(serial_port, baudrate=cube_motion.BAUD_RATE, bytesize=serial.EIGHTBITS,
                           parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
            print("串口连接成功，回零点，请稍候")
            zero = cube_motion.cmd_zero(ser)
            mc = cube_motion.MotionCtrl(ser, zero[0], zero[1], zero[2], zero[3])
            mc.two_finger_init()
            print("进入交互模式")
            img_list = [None, None, None]
            while True:
                # 读取一帧
                ret, frame = cap.read()
                if not ret:
                    print("无法获取帧，退出...")
                    break
                
                # 创建帧的副本用于绘制
                display_frame = frame.copy()
                
                # 在副本上绘制可拖动点和连线
                draggable_points.draw(display_frame)
                
                # 显示时间信息
                if motion_worker is not None and motion_worker.is_alive() and start_time != None:
                    # 计算当前耗时
                    elapsed_time = time.time() - start_time
                    time_text = f"Time: {elapsed_time:.1f}s"
                elif total_time == None and start_time != None:
                    # 显示总耗时
                    total_time = time.time() - start_time
                    time_text = f"Total Time: {total_time:.1f}s"
                elif total_time != None:
                    time_text = f"Total Time: {total_time:.1f}s"
                else:
                    time_text = "Press space key to start."
                
                if time_text:
                    cv2.putText(display_frame, time_text, (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
                
                # ------------------------自动检测状态机-------------------------------
                if state == STATE_INIT:
                    # 舍弃前10帧，让摄像头稳定
                    frame_counter += 1
                    if frame_counter >= BACKGROUND_DISCARD_FRAMES:
                        state = STATE_BACKGROUND
                        frame_counter = 0
                        logger.info("进入背景采集状态")
                
                elif state == STATE_BACKGROUND:
                    # 采集背景颜色
                    try:
                        background_hs = get_avg_color_hs(frame, draggable_points.points)
                        if background_hs:
                            logger.info("背景颜色采集成功")
                            state = STATE_DETECTING
                        else:
                            logger.warning("背景颜色采集失败，重试...")
                    except Exception as e:
                        logger.error(f"背景颜色采集异常: {str(e)}")
                
                elif state == STATE_DETECTING:
                    # 检测魔方放置
                    current_hs = get_avg_color_hs(frame, draggable_points.points)
                    # 计算变化量
                    changed_blocks = 0
                    for i in range(18):
                        # 计算色相差异（考虑圆形特性）
                        h_diff = abs(current_hs[i][0] - background_hs[i][0])
                        h_diff = min(h_diff, 180 - h_diff)
                        # 计算饱和度差异
                        s_diff = abs(current_hs[i][1] - background_hs[i][1])
                        
                        # 如果色相或饱和度变化超过阈值，认为该色块有变化
                        if h_diff > SAT_HUE_THRESHOLD or s_diff > SAT_HUE_THRESHOLD:
                            changed_blocks += 1

                    last_change_value = changed_blocks         
                    # 如果超过一半的色块有变化，认为魔方已放置
                    if changed_blocks >= CHANGE_THRESHOLD:
                        logger.info("检测到魔方放置，进入稳定检测状态")
                        state = STATE_STABILIZING
                        prev_hs = current_hs
                        prev_gray = None
                        stable_counter = 0

                elif state == STATE_DETECTING_REMOVE:
                    # 检测魔方放置
                    current_hs = get_avg_color_hs(frame, draggable_points.points)
                    # 计算变化量
                    changed_blocks = 0
                    for i in range(18):
                        # 计算色相差异（考虑圆形特性）
                        h_diff = abs(current_hs[i][0] - background_hs[i][0])
                        h_diff = min(h_diff, 180 - h_diff)
                        # 计算饱和度差异
                        s_diff = abs(current_hs[i][1] - background_hs[i][1])
                        
                        # 如果色相或饱和度变化超过阈值，认为该色块有变化
                        if h_diff > SAT_HUE_THRESHOLD or s_diff > SAT_HUE_THRESHOLD:
                            changed_blocks += 1

                    last_change_value = changed_blocks
                    # 魔方已经移除
                    if changed_blocks <= CHANGE_THRESHOLD_REMOVE:
                        logger.info("检测到魔方移除，等待魔方放置")
                        state = STATE_DETECTING
                        prev_hs = current_hs
                        stable_counter = 0
                
                elif state == STATE_STABILIZING:
                    # 将当前帧转换为灰度图
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_gray is None:
                        # 第一帧初始化
                        prev_gray = current_gray
                        stable_counter = 0
                    else:
                        # 计算帧间差异
                        diff = cv2.absdiff(prev_gray, current_gray)
                        
                        # 应用阈值
                        _, thresh = cv2.threshold(diff, THRESH_DIFF, 255, cv2.THRESH_BINARY)
                        
                        # 计算非零像素数量
                        change_pixels = cv2.countNonZero(thresh)
                        total_pixels = thresh.shape[0] * thresh.shape[1]
                        change_percent = (change_pixels / total_pixels) * 100
                        
                        logger.info(f"稳定检测 - 变化像素: {change_percent:.2f}%")
                        
                        # 如果变化小于阈值，增加稳定计数器
                        if change_percent < 0.1:  # 0.1%的变化阈值
                            stable_counter += 1
                            logger.info(f"稳定帧: {stable_counter}/{STABILIZATION_FRAMES}")
                            
                            # 如果达到稳定帧数要求，启动工作线程
                            if stable_counter >= STABILIZATION_FRAMES:
                                logger.info("场景稳定，启动魔方还原")
                                start_work_thread()
                                state = STATE_WORKING
                        else:
                            stable_counter = 0  # 重置稳定计数器
                        
                        # 更新前一帧
                        prev_gray = current_gray
                
                elif state == STATE_WORKING:
                    # 工作线程运行中，不需要额外处理
                    pass
                
                state_names = {
                    STATE_INIT: "Initializing",
                    STATE_BACKGROUND: "Background Capturing",
                    STATE_DETECTING: f"Detecting(Change:{last_change_value})",
                    STATE_STABILIZING: f"Stabilizing({stable_counter}/{STABILIZATION_FRAMES})",
                    STATE_DETECTING_REMOVE: f"Detecting cube remove({last_change_value})",
                    STATE_WORKING: "Working"
                }
                state_text = f"State: {state_names.get(state, 'Unknown')}"
                cv2.putText(display_frame, state_text, (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 显示实时预览
                cv2.imshow('Camera Preview', display_frame)
                
                # 检查保存队列
                if not save_queue.empty():
                    image_index = save_queue.get()
                    if image_index is None:  # 工作线程完成信号
                        # 工作完成后重置状态
                        state = STATE_DETECTING_REMOVE
                        logger.info("返回STATE_DETECTING_REMOVE状态")
                    else:
                        # 捕获并保存图像
                        ret, frame = cap.read()
                        if ret:
                            # 注意：保存的图像是原始帧，不包含绘制的圆圈和连线
                            filename = save_path_template.format(image_index)
                            cv2.imwrite(filename, frame)
                            logger.info(f"已保存: {filename}")
                            img_list[image_index - 1] = frame
                            if image_index == 3:
                                # 识别颜色
                                cube_str = detect_cube_color(img_list, draggable_points.points)
                                print(f"检测到的魔方状态: {cube_str}")
                                
                                # 求解魔方
                                if cube_str is not None:
                                    motion_sequence = solve_cube(cube_str)
                                else:
                                    motion_sequence = None
                                # 发送信号给工作线程开始旋转魔方
                                tweak_queue.put(motion_sequence)  # 将识别结果传递给工作线程
                
                # 检查按键（等待1ms）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27是ESC键
                    break
                elif key == ord('c') or key == 32:  # 32是空格键
                    if motion_worker is None or not motion_worker.is_alive():
                        logger.info("手动启动魔方还原")
                        start_work_thread()
                    else:
                        logger.info("已有工作线程在运行，忽略新请求")
    
    except serial.SerialException as e:
        logger.error(f"打开串口 {serial_port} 失败: {str(e)}")
        cube_motion.list_serial_ports()
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序发生错误: {str(e)}")
        traceback.print_exc()
        logger.error("请检查电机控制器的连接")
    finally:
        # 确保工作线程停止
        if motion_worker and motion_worker.is_alive():
            motion_worker.stop()
            motion_worker.join(timeout=1.0)
        
        # 保存点坐标到文件
        draggable_points.save_points()
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")