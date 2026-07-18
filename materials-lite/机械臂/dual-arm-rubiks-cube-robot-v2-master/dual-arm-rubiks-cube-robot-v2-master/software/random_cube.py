import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from functools import reduce
import numpy as np

# 定义魔方布局
layout = [
    [-1, -1, -1,  0,  1,  2, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1,  3,  4,  5, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1,  6,  7,  8, -1, -1, -1, -1, -1, -1],
    [36, 37, 38, 18, 19, 20,  9, 10, 11, 45, 46, 47],
    [39, 40, 41, 21, 22, 23, 12, 13, 14, 48, 49, 50],
    [42, 43, 44, 24, 25, 26, 15, 16, 17, 51, 52, 53],
    [-1, -1, -1, 27, 28, 29, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 30, 31, 32, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 33, 34, 35, -1, -1, -1, -1, -1, -1]
]

# 基准颜色配置 (H, S, V)
# BASE_COLORS = {
#     'U': (61, 160, 175),
#     'R': (51, 12, 143),
#     'F': (0, 178, 193),
#     'D': (103, 177, 141),
#     'L': (33, 140, 156),
#     'B': (9, 170, 183)
# }



# 分组定义 (面: [索引列表], 渐变方向, 渐变趋势)
FACE_GROUPS = {
    'U': {'indices': [0,  1,  2,  3,  4,  5,  6,  7,  8 ], 'direction': 'vertical',   'trend': 'decreasing'},
    'R': {'indices': [9,  10, 11, 12, 13, 14, 15, 16, 17], 'direction': 'horizontal', 'trend': 'decreasing'},
    'F': {'indices': [18, 19, 20, 21, 22, 23, 24, 25, 26], 'direction': 'horizontal', 'trend': 'increasing'},
    'D': {'indices': [27, 28, 29, 30, 31, 32, 33, 34, 35], 'direction': 'horizontal', 'trend': 'decreasing'},
    'L': {'indices': [36, 37, 38, 39, 40, 41, 42, 43, 44], 'direction': 'vertical',   'trend': 'increasing'},
    'B': {'indices': [45, 46, 47, 48, 49, 50, 51, 52, 53], 'direction': 'vertical',   'trend': 'decreasing'}
}

# 打乱状态字符串示例
# SCRAMBLED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
# SCRAMBLED_STATE = "BFBDULFDDFUDFRUDRLDFRDFRRLBFFRLDBLUFRRLDLBUBULUURBLUBB"

def generate_scrambled_data(scrambled_state, base_colors):
    """
    根据打乱状态生成模拟数据，模拟光照不均匀效果
    返回格式: [index, H, S, V, label]
    """
    # 创建空数据列表
    data = []
    
    # 处理每个面
    for face_label, group in FACE_GROUPS.items():
        # 获取该组的索引列表
        indices = group['indices']
        direction = group['direction']
        trend = group['trend']
        
        # 创建3x3网格位置
        positions = [(i, j) for i in range(3) for j in range(3)]
        
        # 计算每个位置的光照因子 (基于实测数据规律)
        light_factors = np.zeros((3, 3))
        FA_MAX = 1.00
        FA_AVG = 0.85
        FA_MIN = 0.70
        if direction == 'vertical':
            # 垂直渐变
            if trend == 'decreasing':  # 从上到下变暗
                light_factors = np.array([
                    [FA_MAX, FA_MAX, FA_MAX],  # 顶部行 - 最亮
                    [FA_AVG, FA_AVG, FA_AVG],  # 中间行 - 中等
                    [FA_MIN, FA_MIN, FA_MIN]   # 底部行 - 最暗
                ])
            else:  # 从上到下变亮
                light_factors = np.array([
                    [FA_MIN, FA_MIN, FA_MIN],  # 顶部行 - 最暗
                    [FA_AVG, FA_AVG, FA_AVG],  # 中间行 - 中等
                    [FA_MAX, FA_MAX, FA_MAX]   # 底部行 - 最亮
                ])
        else:  # 水平渐变
            if trend == 'decreasing':  # 从左到右变暗
                light_factors = np.array([
                    [FA_MAX, FA_AVG, FA_MIN],  # 左亮右暗
                    [FA_MAX, FA_AVG, FA_MIN],
                    [FA_MAX, FA_AVG, FA_MIN]
                ])
            else:  # 从左到右变亮
                light_factors = np.array([
                    [FA_MIN, FA_AVG, FA_MAX],  # 左暗右亮
                    [FA_MIN, FA_AVG, FA_MAX],
                    [FA_MIN, FA_AVG, FA_MAX]
                ])
        
        # 添加组内随机变化 (模拟实测数据中的细微变化)
        random_variation = np.random.uniform(-0.05, 0.05, size=(3, 3))
        light_factors += random_variation
        
        # 处理每个格子
        for idx, (i, j) in zip(indices, positions):
            # 获取实际标签 (来自打乱状态)
            lbl = scrambled_state[idx]
            
            # 使用实际标签对应的基准颜色
            base_h, base_s, base_v = base_colors[lbl]
            
            # 获取该位置的光照因子
            light_factor = light_factors[i, j]
            
            # 添加随机噪声 (模拟实拍效果)
            if base_s<48:
                h_noise = np.random.uniform(-15, 15)
            else:
                h_noise = np.random.uniform(-3, 3)  # H值噪声范围小
            s_noise = np.random.uniform(-5, 5)  # S值中等噪声
            v_noise = np.random.uniform(-8, 8)  # V值噪声较大
            
            # 应用光照因子和噪声
            h = base_h + h_noise
            if h > 180:
                h -= 180
            if h < 0:
                h += 180
            s = max(0, min(255, base_s * light_factor + s_noise))
            v = max(0, min(255, base_v * light_factor + v_noise))
            
            # 添加到数据列表
            data.append([idx, h, s, v, lbl])
    
    # 按索引排序
    data.sort(key=lambda x: x[0])
    return data

def create_color_and_label_dict(data):
    """创建颜色和标签字典"""
    colors = {}
    labels = {}
    for d in data:
        idx, h, s, v, lbl = d
        # 归一化HSV值到0-1范围
        h_norm = h / 179.0
        s_norm = s / 255.0
        v_norm = v / 255.0
        # 转换为RGB
        rgb = hsv_to_rgb((h_norm, s_norm, v_norm))
        colors[idx] = rgb
        labels[idx] = f"{idx}:{lbl}"
    return colors, labels
    
def plot_cube_net(colors, labels):
    """绘制魔方展开图"""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()  # 反转Y轴使原点在左上角
    ax.axis('off')  # 隐藏坐标轴

    # 绘制每个格子
    for y, row in enumerate(layout):
        for x, idx in enumerate(row):
            if idx != -1:
                # 计算矩形位置（左上角坐标）
                rect_x = x
                rect_y = y
                
                # 创建矩形
                rect = patches.Rectangle(
                    (rect_x, rect_y), 1, 1, 
                    linewidth=1, edgecolor='black',
                    facecolor=colors[idx]
                )
                ax.add_patch(rect)
                
                # 添加标签
                ax.text(
                    rect_x + 0.5, rect_y + 0.5, 
                    labels[idx], 
                    ha='center', va='center', 
                    fontsize=10, fontweight='bold',
                    color='black'
                )

    # 调整布局
    plt.tight_layout()
    plt.show()

def format_scrambled_data(scrambled_data):
    output_lines = []
    for item in scrambled_data:
        # 提取并四舍五入HSV值为整数
        idx = item[0]
        h = round(item[1])
        s = round(item[2])
        v = round(item[3])
        label = item[4]
        
        # 格式化为字符串行
        line = f"{idx}\t{h}\t{s}\t{v}\t{label}"
        output_lines.append(line)
    
    # 用换行符连接所有行
    return "\n".join(output_lines)

# 1、主色选择红、绿、橙、蓝、黄、白共六种颜色
# 2、颜色分布为上黄下白前蓝后绿左橙右红，或者这个配色的旋转（旋转时要保持色块关系，参考下面的c程序的旋转部分）
# 3、白色的色相完全随机，其他五种颜色的色相在各自的范围内随机（均匀分布）
# 4、白色的饱和度在0-40范围内随机产生（均匀分布），其他颜色的饱和度在80-255范围内随机产生，但是要做到大部分情况下其他颜色的饱和度是接近的，
#    可以先产生一个饱和度基准（中心值在160左右），在此基础上增加偏移，基准和偏移都按照正态分布产生随机数
# 5、所有颜色的亮度，在80-255范围内随机产生，（正态分布，中心值在160左右）
# 6、如果存在两种颜色的色相，差值小于8，需要保证这两种颜色的饱和度或者亮度有明显差异，饱和度差值绝对值+亮度差值绝对值大于50，如果不符合，重新随机产生
def generate_random_base_colors():
    """
    随机生成基准颜色配置
    返回格式: {'U': (H, S, V), ...}
    """
    # 标准配色（前蓝后绿左橙右红上黄下白）
    std_colors = {
        'U': 'yellow',
        'D': 'white',
        'F': 'blue',
        'B': 'green',
        'L': 'orange',
        'R': 'red'
    }
    
    # 基础旋转函数 (绕X/Y/Z轴)
    def rot_x(face):
        # 绕X轴旋转90度 (左右轴：L->L，R->R不变)
        return {'U':'F', 'F':'D', 'D':'B', 'B':'U', 'L':'L', 'R':'R'}.get(face, face)
    
    def rot_y(face):
        # 绕Y轴旋转90度 (垂直轴：U->U, D->D不变)
        return {'F':'L', 'L':'B', 'B':'R', 'R':'F', 'U':'U', 'D':'D'}.get(face, face)
    
    def rot_z(face):
        # 绕Z轴旋转90度 (前后轴：F->F, B->B不变)
        return {'U':'R', 'R':'D', 'D':'L', 'L':'U', 'F':'F', 'B':'B'}.get(face, face)
    
    # 生成所有可能的旋转组合
    def apply_rotations(face, rotations):
        """应用一系列旋转函数"""
        return reduce(lambda f, rot_fn: rot_fn(f), rotations, face)
    
    # 生成所有24种有效旋转
    rotation_fns = []
    axes = [rot_x, rot_y, rot_z]
    
    # 生成所有可能的旋转序列 (最多4次旋转)
    for i in range(4):  # X轴旋转次数
        for j in range(4):  # Y轴旋转次数
            for k in range(4):  # Z轴旋转次数
                rotations = []
                # 添加X轴旋转
                rotations.extend([rot_x] * i)
                # 添加Y轴旋转
                rotations.extend([rot_y] * j)
                # 添加Z轴旋转
                rotations.extend([rot_z] * k)
                
                # 添加到列表
                rotation_fns.append(lambda f, r=rotations: apply_rotations(f, r))
    
    # 去重，只保留唯一的旋转映射
    unique_rotations = []
    seen = set()
    
    for rot_fn in rotation_fns:
        # 计算旋转映射
        mapping = tuple((face, rot_fn(face)) for face in sorted(std_colors.keys()))
        if mapping not in seen:
            seen.add(mapping)
            unique_rotations.append(rot_fn)
    
    # 2. 选择随机旋转
    rotation_idx = np.random.randint(0, len(unique_rotations))
    rotation_fn = unique_rotations[rotation_idx]
    
    # 3. 创建旋转映射
    rotated_map = {face: rotation_fn(face) for face in std_colors}
    
    def gen_color_params(color_name):
        """生成单个颜色参数"""
        s_range = (100, 255)
        v_range = (100, 255)
        s_center = 160
        v_center = 160
        while True:
            # 色相生成
            if color_name == 'white':
                h = np.random.uniform(0, 179)
            else:
                # 根据颜色名称设置不同的色相中心值和标准差
                if color_name == 'red':
                    h_mean, h_std = 0, 10
                elif color_name == 'orange':
                    h_mean, h_std = 9, 10
                elif color_name == 'yellow':
                    h_mean, h_std = 33, 10
                elif color_name == 'green':
                    h_mean, h_std = 61, 10
                elif color_name == 'blue':
                    h_mean, h_std = 103, 10
                else:
                    h_mean, h_std = 0, 10
                
                # 生成正态分布的色相
                h = np.random.normal(h_mean, h_std)
                # 将色相调整到0-179之间
                h = h % 180
                if h < 0:
                    h += 180
            
            # 饱和度处理
            if color_name == "white":
                s = np.random.uniform(0, 20)
            else:
                base_s = np.random.normal(s_center, 20)
                s = np.clip(base_s + np.random.normal(0, 5), s_range[0], s_range[1])
            
            # 亮度处理
            v = np.random.normal(v_center, 20)
            v = np.clip(v, v_range[0], v_range[1])
            
            yield (h, s, v)

    # 修改后的color_generators定义
    SV_RANGE = (100, 255)
    color_generators = {
        'red':    gen_color_params('red'),
        'orange': gen_color_params('orange'),
        'yellow': gen_color_params('yellow'),
        'green':  gen_color_params('green'),
        'blue':   gen_color_params('blue'),
        'white':  gen_color_params('white')
    }
    
    # 6. 生成并检查颜色
    max_attempts = 100
    for _ in range(max_attempts):
        # 生成颜色值
        colors = {name: next(gen) for name, gen in color_generators.items()}
        
        # 检查颜色差异 (排除白色)
        valid = True
        non_white = [c for name, c in colors.items() if name != 'white']
        for i in range(len(non_white)):
            for j in range(i+1, len(non_white)):
                h1, s1, v1 = non_white[i]
                h2, s2, v2 = non_white[j]
                
                # 计算色相差值 (考虑环形)
                h_diff = min(abs(h1-h2), 180-abs(h1-h2))
                
                if h_diff < 8:  # 色相太接近
                    s_diff = abs(s1-s2)
                    v_diff = abs(v1-v2)
                    if s_diff + v_diff < 50:  # 饱和度+亮度差异不足
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            break
    
    # 7. 应用旋转后的颜色映射
    return {
        face: colors[std_colors[rotated_map[face]]]
        for face in ['U', 'R', 'F', 'D', 'L', 'B']
    }



# 定义魔方初始状态
SOLVED_CUBE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

# 定义转动枚举（使用数值代替枚举）
L = 0; L3 = 1; L2 = 2; R = 3; R3 = 4; R2 = 5
U = 6; U3 = 7; U2 = 8; D = 9; D3 = 10; D2 = 11
F = 12; F3 = 13; F2 = 14; B = 15; B3 = 16; B2 = 17

# 边缘块映射关系 (棱块位置 -> 面索引)
edge_to_face = [
    [7, 19], [5, 10], [1, 46], [3, 37], 
    [28, 25], [32, 16], [34, 52], [30, 43],
    [23, 12], [21, 41], [48, 14], [50, 39]
]

# 角块映射关系 (角块位置 -> 面索引)
corner_to_face = [
    [8, 20, 9], [2, 11, 45], [0, 47, 36], [6, 38, 18],
    [29, 15, 26], [27, 24, 44], [33, 42, 53], [35, 51, 17]
]

# 棱块名称映射 (位置和朝向 -> 颜色对)
edge_index = [
    "UF", "UR", "UB", "UL", "DF", "DR", "DB", "DL", "FR", "FL", "BR", "BL",
    "FU", "RU", "BU", "LU", "FD", "RD", "BD", "LD", "RF", "LF", "RB", "LB"
]

# 角块名称映射 (位置和朝向 -> 颜色三连)
corner_index = [
    "UFR", "URB", "UBL", "ULF", "DRF", "DFL", "DLB", "DBR",
    "FRU", "RBU", "BLU", "LFU", "RFD", "FLD", "LBD", "BRD",
    "RUF", "BUR", "LUB", "FUL", "FDR", "LDF", "BDL", "RDB"
]

# 转动表 - 棱块位置
route_tab_ep = [
    [0,1,2,11,4,5,6,9,8,3,10,7],  # L
    [0,1,2,9,4,5,6,11,8,7,10,3],  # L3
    [0,1,2,7,4,5,6,3,8,11,10,9],  # L2
    [0,8,2,3,4,10,6,7,5,9,1,11],  # R
    [0,10,2,3,4,8,6,7,1,9,5,11],  # R3
    [0,5,2,3,4,1,6,7,10,9,8,11],  # R2
    [1,2,3,0,4,5,6,7,8,9,10,11],  # U
    [3,0,1,2,4,5,6,7,8,9,10,11],  # U3
    [2,3,0,1,4,5,6,7,8,9,10,11],  # U2
    [0,1,2,3,7,4,5,6,8,9,10,11],  # D
    [0,1,2,3,5,6,7,4,8,9,10,11],  # D3
    [0,1,2,3,6,7,4,5,8,9,10,11],  # D2
    [9,1,2,3,8,5,6,7,0,4,10,11],  # F
    [8,1,2,3,9,5,6,7,4,0,10,11],  # F3
    [4,1,2,3,0,5,6,7,9,8,10,11],  # F2
    [0,1,10,3,4,5,11,7,8,9,6,2],  # B
    [0,1,11,3,4,5,10,7,8,9,2,6],  # B3
    [0,1,6,3,4,5,2,7,8,9,11,10]   # B2
]

# 转动表 - 棱块朝向
route_tab_er = [
    [0,0,0,0,0,0,0,0,0,0,0,0],  # L
    [0,0,0,0,0,0,0,0,0,0,0,0],  # L3
    [0,0,0,0,0,0,0,0,0,0,0,0],  # L2
    [0,0,0,0,0,0,0,0,0,0,0,0],  # R
    [0,0,0,0,0,0,0,0,0,0,0,0],  # R3
    [0,0,0,0,0,0,0,0,0,0,0,0],  # R2
    [0,0,0,0,0,0,0,0,0,0,0,0],  # U
    [0,0,0,0,0,0,0,0,0,0,0,0],  # U3
    [0,0,0,0,0,0,0,0,0,0,0,0],  # U2
    [0,0,0,0,0,0,0,0,0,0,0,0],  # D
    [0,0,0,0,0,0,0,0,0,0,0,0],  # D3
    [0,0,0,0,0,0,0,0,0,0,0,0],  # D2
    [1,0,0,0,1,0,0,0,1,1,0,0],  # F
    [1,0,0,0,1,0,0,0,1,1,0,0],  # F3
    [0,0,0,0,0,0,0,0,0,0,0,0],  # F2
    [0,0,1,0,0,0,1,0,0,0,1,1],  # B
    [0,0,1,0,0,0,1,0,0,0,1,1],  # B3
    [0,0,0,0,0,0,0,0,0,0,0,0]   # B2
]

# 转动表 - 角块位置
route_tab_cp = [
    [0,1,6,2,4,3,5,7],  # L
    [0,1,3,5,4,6,2,7],  # L3
    [0,1,5,6,4,2,3,7],  # L2
    [4,0,2,3,7,5,6,1],  # R
    [1,7,2,3,0,5,6,4],  # R3
    [7,4,2,3,1,5,6,0],  # R2
    [1,2,3,0,4,5,6,7],  # U
    [3,0,1,2,4,5,6,7],  # U3
    [2,3,0,1,4,5,6,7],  # U2
    [0,1,2,3,5,6,7,4],  # D
    [0,1,2,3,7,4,5,6],  # D3
    [0,1,2,3,6,7,4,5],  # D2
    [3,1,2,5,0,4,6,7],  # F
    [4,1,2,0,5,3,6,7],  # F3
    [5,1,2,4,3,0,6,7],  # F2
    [0,7,1,3,4,5,2,6],  # B
    [0,2,6,3,4,5,7,1],  # B3
    [0,6,7,3,4,5,1,2]   # B2
]

# 转动表 - 角块朝向
route_tab_cr = [
    [0,0,2,1,0,2,1,0],  # L
    [0,0,2,1,0,2,1,0],  # L3
    [0,0,0,0,0,0,0,0],  # L2
    [2,1,0,0,1,0,0,2],  # R
    [2,1,0,0,1,0,0,2],  # R3
    [0,0,0,0,0,0,0,0],  # R2
    [0,0,0,0,0,0,0,0],  # U
    [0,0,0,0,0,0,0,0],  # U3
    [0,0,0,0,0,0,0,0],  # U2
    [0,0,0,0,0,0,0,0],  # D
    [0,0,0,0,0,0,0,0],  # D3
    [0,0,0,0,0,0,0,0],  # D2
    [1,0,0,2,2,1,0,0],  # F
    [1,0,0,2,2,1,0,0],  # F3
    [0,0,0,0,0,0,0,0],  # F2
    [0,2,1,0,0,0,2,1],  # B
    [0,2,1,0,0,0,2,1],  # B3
    [0,0,0,0,0,0,0,0]   # B2
]

class Cube:
    def __init__(self):
        # 棱块位置 (0-11) 和朝向 (0/1)
        self.ep = list(range(12))
        self.er = [0] * 12
        # 角块位置 (0-7) 和朝向 (0/1/2)
        self.cp = list(range(8))
        self.cr = [0] * 8

def cube_from_face_54(cube_str):
    """从54面字符串初始化魔方状态 (仅用于复原状态)"""
    c = Cube()
    # 棱块处理 (直接从已解决状态初始化)
    c.ep = list(range(12))
    c.er = [0] * 12
    # 角块处理
    c.cp = list(range(8))
    c.cr = [0] * 8
    return c  # 原始验证逻辑已跳过，因只用于初始状态

def cube_route(cube, d):
    """执行指定转动操作"""
    # 复制当前状态
    ep_tmp = [0] * 12
    er_tmp = [0] * 12
    cp_tmp = [0] * 8
    cr_tmp = [0] * 8
    
    # 应用棱块转动表
    for i in range(12):
        src_idx = route_tab_ep[d][i]
        ep_tmp[i] = cube.ep[src_idx]
        er_tmp[i] = cube.er[src_idx] ^ route_tab_er[d][i]  # 异或处理翻转
    
    # 应用角块转动表
    for i in range(8):
        src_idx = route_tab_cp[d][i]
        cp_tmp[i] = cube.cp[src_idx]
        cr_tmp[i] = (cube.cr[src_idx] + route_tab_cr[d][i]) % 3  # 加法取模处理旋转
    
    # 更新魔方状态
    cube.ep = ep_tmp
    cube.er = er_tmp
    cube.cp = cp_tmp
    cube.cr = cr_tmp

def cube_to_face_54(cube):
    """将魔方状态转换为54面字符串"""
    # 初始化所有面，包含默认中心块
    cube_str = [''] * 54
    centers = [4, 13, 22, 31, 40, 49]
    colors = ['U', 'R', 'F', 'D', 'L', 'B']
    for i, pos in enumerate(centers):
        cube_str[pos] = colors[i]
    
    # 填充棱块
    for i in range(12):
        idx_a, idx_b = edge_to_face[i]
        edge_name = edge_index[cube.ep[i] + 12 * cube.er[i]]
        cube_str[idx_a] = edge_name[0]
        cube_str[idx_b] = edge_name[1]
    
    # 填充角块
    for i in range(8):
        idx_a, idx_b, idx_c = corner_to_face[i]
        corner_name = corner_index[cube.cp[i] + 8 * cube.cr[i]]
        cube_str[idx_a] = corner_name[0]
        cube_str[idx_b] = corner_name[1]
        cube_str[idx_c] = corner_name[2]
    
    return ''.join(cube_str)

def scramble():
    """生成随机打乱状态"""
    # 初始化复原状态魔方
    cube = cube_from_face_54(SOLVED_CUBE)
    
    # 随机转动0-100次
    steps = np.random.randint(0, 100)
    for _ in range(steps):
        move = np.random.randint(0, 17)  # 随机选择0-17的操作
        cube_route(cube, move)
    
    # 返回54面字符串
    return cube_to_face_54(cube)

# 生成100组数据并保存到文件
if __name__ == "__main__":
    # 设置基础随机种子
    base_seed = 4222523
    output_lines = []  # 存储所有输出内容
    data_size = 100
    # 生成data_size组数据
    for i in range(data_size):
        # 设置当前组的随机种子（基础种子+组号）
        np.random.seed(base_seed + i)
        
        # 生成模拟数据（考虑光照不均匀）
        base_colors = generate_random_base_colors()
        scrambled_state = scramble()
        scrambled_data = generate_scrambled_data(scrambled_state, base_colors)
        formatted_output = format_scrambled_data(scrambled_data)
        
        # 添加分隔线和组号标题
        output_lines.append(f"# ------ {i} ------")
        output_lines.append(formatted_output)
        
        # 如果是第最后一组，准备绘制魔方展开图
        if i == data_size - 1:
            colors, labels = create_color_and_label_dict(scrambled_data)
    
    # 将所有数据写入文件
    with open("temp/cube_data.txt", "w") as f:
        f.write("\n".join(output_lines))
    
    # 仅绘制第100组的魔方展开图
    plot_cube_net(colors, labels)