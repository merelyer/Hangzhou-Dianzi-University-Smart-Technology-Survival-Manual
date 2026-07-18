import ctypes
import platform
import sys
import cv2
import json
import numpy as np
# 根据平台加载动态库
system = platform.system()
c_lib_name = "./libcube"
if system == "Windows":
    c_lib_name += ".dll"
elif system == "Linux":
    c_lib_name += ".so"
elif system == "Darwin":
    c_lib_name += ".dylib"
else:
    raise OSError("Unsupported platform")

try:
    lib = ctypes.cdll.LoadLibrary(c_lib_name)
except Exception as e:
    print(f"加载{c_lib_name}失败：{str(e)}")
    sys.exit(1)

# 设置函数原型
lib.solve_motion.argtypes = [
    ctypes.c_char_p,  # facelets
    ctypes.c_char_p,  # initial_state
    ctypes.c_char_p   # output_sequence
]
lib.solve_motion.restype = ctypes.c_int

def solve_cube_c(facelets: str, initial_state: str = "RU") -> str:
    """
    调用C函数求解魔方
    
    :param facelets: 魔方状态字符串 (如"BBBFULRUBUURFRRRDFDFLUFDLRUUUFFDRLDLRRFLLBBLFDLUBBDDBD")
    :param initial_state: 初始机械状态 (可选)
    :return: 机械动作序列
    """
    # 创建足够大的缓冲区接收结果
    output_buf = ctypes.create_string_buffer(1024)
    
    # 调用C函数
    result = lib.solve_motion(
        facelets.encode('utf-8'),
        initial_state.encode('utf-8'),
        output_buf
    )
    
    if result < 0:
        raise RuntimeError(f"Solver failed with error code: {result}")
    
    return output_buf.value.decode('utf-8')

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
        raise RuntimeError(f"颜色检测失败，错误代码: {result}")
    
    # 返回前54个字符(实际魔方状态)
    return cube_str_buf.value[:54].decode('utf-8')


# 使用示例
if __name__ == "__main__":
    # 示例魔方状态 (已解状态)
    facelets = "BBBFULRUBUURFRRRDFDFLUFDLRUUUFFDRLDLRRFLLBBLFDLUBBDDBD"
    
    try:
        solution = solve_cube_c(facelets)
        print(f"机械动作序列: {solution}")
    except Exception as e:
        print(f"求解失败: {str(e)}")

    # 读取JSON文件
    try:
        with open('../software/points_config.json', 'r') as f:
            points = json.load(f)
        points = np.array(points)
    except Exception as e:
        print(f"错误：无法读取JSON文件 - {str(e)}")
        exit()

    if len(points) < 6:
        print("错误：JSON文件中需要至少6个点")
        exit()
        
    # 1. 读取三张图像
    img_list = []
    for i in range(1, 4):
        img_path = f'../img/testcase/1/captured_image_{i}.jpg'
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img_list.append(img)
    
    # 2. 检测魔方颜色
    try:
        cube_state = detect_cube_color(img_list, points)
        print(f"检测到的魔方状态: {cube_state}")
        
        # 3. 求解魔方动作序列
        solution = solve_cube_c(cube_state)
        print(f"机械动作序列: {solution}")
    except Exception as e:
        print(f"处理失败: {str(e)}")

