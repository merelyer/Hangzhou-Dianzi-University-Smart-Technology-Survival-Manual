#!/usr/bin/python3
# 有个问题，如果在vscode下直接运行这个程序，有可能会无法正常拖动圆圈，将实时预览窗口切换最小化再切回就能恢复
# 这个可能是输入焦点不在实时预览窗口引起的
# 在终端下运行没有这个问题（偶发的，没解决思路，也不影响使用）

import cube_motion
import cube_color_detect
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
import cube_optimizer
import kociemba

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def solve_cube(facelets):
    start_time_solve = time.time()
    initial_state = "RU"
    try:
        solution = kociemba.solve(facelets)
        logger.info(f"魔方解法：{solution}")
        motion_sequence, _ = cube_optimizer.solution_to_motion(initial_state, solution)

    except Exception as e:
        logger.error(f"不可解的魔方 - {str(e)}")
        traceback.print_exc()
        motion_sequence = None
    logger.info(f"魔方求解耗时: {(time.time() - start_time_solve)*1000:.2f}ms")
    return motion_sequence

# 配置文件路径
POINTS_CONFIG_FILE = "points_config.json"

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
                
                # 显示实时预览
                cv2.imshow('Camera Preview', display_frame)
                
                # 检查保存队列
                if not save_queue.empty():
                    image_index = save_queue.get()
                    if image_index is None:  # 工作线程完成信号
                        motion_worker = None
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
                                start_time_detect = time.time()
                                cube_str = cube_color_detect.color_detect(img_list, draggable_points.points)
                                logger.info(f"识别完成! 时间: {(time.time() - start_time_detect)*1000:.2f}ms")
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
                        logger.info("启动工作线程执行动作序列")
                        # 重置时间变量
                        start_time = time.time()
                        total_time = None
                        motion_worker = MotionWorker(mc, save_queue, tweak_queue)
                        motion_worker.daemon = True  # 设置为守护线程
                        motion_worker.start()
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