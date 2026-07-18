import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import time
import trimesh  # 添加trimesh库用于精确碰撞检测

# pip3 install trimesh python-fcl

# 单位立方体的顶点和面
unit_cube_vertices = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5]
])

unit_cube_faces = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
]

# 将四边形面转换为三角形面
triangle_faces = []
for face in unit_cube_faces:
    triangle_faces.append([face[0], face[1], face[2]])
    triangle_faces.append([face[0], face[2], face[3]])
triangle_faces = np.array(triangle_faces)

# 变换矩阵函数
def translate(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def scale(sx, sy, sz):
    return np.array([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1]
    ])

def rot_z(angle):
    theta = np.radians(angle)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta),  0, 0],
        [0,             0,              1, 0],
        [0,             0,              0, 1]
    ])

def rot_y(angle):
    theta = np.radians(angle)
    return np.array([
        [np.cos(theta),  0, np.sin(theta), 0],
        [0,              1, 0,             0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0,              0, 0,             1]
    ])

def rot_x(angle):
    theta = np.radians(angle)
    return np.array([
        [1, 0,             0,              0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta),  0],
        [0, 0,             0,              1]
    ])

# 应用变换矩阵到立方体
def transform_cube(vertices, matrix):
    homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    transformed_vertices = np.dot(homogeneous_vertices, matrix.T)
    return transformed_vertices[:, :3]

# 使用trimesh进行精确碰撞检测
def check_collision(vertices1, vertices2):
    mesh1 = trimesh.Trimesh(vertices=vertices1, faces=triangle_faces)
    mesh2 = trimesh.Trimesh(vertices=vertices2, faces=triangle_faces)
    manager = trimesh.collision.CollisionManager()
    manager.add_object('mesh1', mesh1)
    manager.add_object('mesh2', mesh2)
    return manager.in_collision_internal()

# 核心功能：计算最小手指距离
def calculate_min_offset(cube_angle, arm_angle):
    """
    计算给定魔方角度和手臂角度时的最小手指距离
    
    参数:
        cube_angle: 魔方旋转角度 (0-90度)
        arm_angle: 手臂旋转角度 (0-90度)
    
    返回:
        最小不碰撞的手指距离 (mm)
    """
    # 魔方变换矩阵
    matrix_cube = rot_z(cube_angle) @ scale(56, 56, 56)
    cube_vertices = transform_cube(unit_cube_vertices, matrix_cube)
    
    # 左旋转臂变换矩阵
    matrix_arm_left = translate(0, -56/2-15, 0) @ rot_y(arm_angle) @ rot_x(-90)
    
    # 二分法搜索范围
    low = 53.0
    high = 88.0
    precision = 0.1
    min_safe_offset = high
    
    # 二分搜索
    while high - low > precision:
        mid = (low + high) / 2.0
        offset = round(mid, 1)
        
        # 创建左臂的四个部分
        parts = []
        
        # 左臂第一部分
        matrix_C1 = matrix_arm_left @ translate(-offset/2, 0, 0) @ translate(-2, 0, 14) @ scale(4, 12, 28)
        C1_vertices = transform_cube(unit_cube_vertices, matrix_C1)
        parts.append(C1_vertices)
        
        # 左臂第二部分
        matrix_C2 = matrix_arm_left @ translate(-offset/2, 0, 0) @ translate(3, 0, 7) @ scale(6, 12, 14)
        C2_vertices = transform_cube(unit_cube_vertices, matrix_C2)
        parts.append(C2_vertices)
        
        # 左臂第三部分
        matrix_C3 = matrix_arm_left @ translate(offset/2, 0, 0) @ rot_z(180) @ translate(-2, 0, 14) @ scale(4, 12, 28)
        C3_vertices = transform_cube(unit_cube_vertices, matrix_C3)
        parts.append(C3_vertices)
        
        # 左臂第四部分
        matrix_C4 = matrix_arm_left @ translate(offset/2, 0, 0) @ rot_z(180) @ translate(3, 0, 7) @ scale(6, 12, 14)
        C4_vertices = transform_cube(unit_cube_vertices, matrix_C4)
        parts.append(C4_vertices)
        
        # 检查碰撞
        collision_detected = False
        for part_vertices in parts:
            if check_collision(cube_vertices, part_vertices):
                collision_detected = True
                break
        
        if collision_detected:
            low = mid + precision
        else:
            min_safe_offset = min(min_safe_offset, offset)
            high = mid - precision
    
    return min_safe_offset

# 绘制立方体
def draw_cube(ax, vertices, faces, color, alpha=0.8):
    for face in faces:
        polygon = Poly3DCollection([vertices[face]], alpha=alpha)
        polygon.set_facecolor(color)
        polygon.set_edgecolor('k')
        ax.add_collection3d(polygon)

# 创建交互式可视化
def create_interactive_visualization(min_offsets):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    ax.set_proj_type('ortho')
    ax.set_box_aspect([1, 1, 1])
    
    # 初始角度值
    angle_right = 45
    
    # 更新函数
    def update(val):
        angle_right = slider_angle.val
        angle_left = 90 - angle_right
        arm_offset = min_offsets[round(angle_right)]
        
        ax.clear()
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_zlim(-40, 40)
        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type('persp')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 魔方
        matrix_cube = rot_z(angle_right) @ scale(56, 56, 56)
        cube_vertices = transform_cube(unit_cube_vertices, matrix_cube)
        draw_cube(ax, cube_vertices, unit_cube_faces, 'gold', alpha=0.5)
        
        # 右旋转臂
        matrix_arm_right = rot_z(angle_right) @ translate(0, 0, -56/2-15)
        
        # 右臂各部分
        matrix_B1 = matrix_arm_right @ translate(-28, 0, 0) @ translate(-2, 0, 14) @ scale(4, 12, 28)
        B1_vertices = transform_cube(unit_cube_vertices, matrix_B1)
        draw_cube(ax, B1_vertices, unit_cube_faces, 'royalblue')
        
        matrix_B2 = matrix_arm_right @ translate(-28, 0, 0) @ translate(3, 0, 7) @ scale(6, 12, 14)
        B2_vertices = transform_cube(unit_cube_vertices, matrix_B2)
        draw_cube(ax, B2_vertices, unit_cube_faces, 'cornflowerblue')
        
        matrix_B3 = matrix_arm_right @ translate(28, 0, 0) @ rot_z(180) @ translate(-2, 0, 14) @ scale(4, 12, 28)
        B3_vertices = transform_cube(unit_cube_vertices, matrix_B3)
        draw_cube(ax, B3_vertices, unit_cube_faces, 'royalblue')
        
        matrix_B4 = matrix_arm_right @ translate(28, 0, 0) @ rot_z(180) @ translate(3, 0, 7) @ scale(6, 12, 14)
        B4_vertices = transform_cube(unit_cube_vertices, matrix_B4)
        draw_cube(ax, B4_vertices, unit_cube_faces, 'cornflowerblue')
        
        # 左旋转臂
        matrix_arm_left = translate(0, -56/2-15, 0) @ rot_y(angle_left) @ rot_x(-90)
        
        # 左臂各部分
        matrix_C1 = matrix_arm_left @ translate(-arm_offset/2, 0, 0) @ translate(-2, 0, 14) @ scale(4, 12, 28)
        C1_vertices = transform_cube(unit_cube_vertices, matrix_C1)
        draw_cube(ax, C1_vertices, unit_cube_faces, 'firebrick')
        
        matrix_C2 = matrix_arm_left @ translate(-arm_offset/2, 0, 0) @ translate(3, 0, 7.5) @ scale(6, 12, 15)
        C2_vertices = transform_cube(unit_cube_vertices, matrix_C2)
        draw_cube(ax, C2_vertices, unit_cube_faces, 'lightcoral')
        
        matrix_C3 = matrix_arm_left @ translate(arm_offset/2, 0, 0) @ rot_z(180) @ translate(-2, 0, 14) @ scale(4, 12, 28)
        C3_vertices = transform_cube(unit_cube_vertices, matrix_C3)
        draw_cube(ax, C3_vertices, unit_cube_faces, 'firebrick')
        
        matrix_C4 = matrix_arm_left @ translate(arm_offset/2, 0, 0) @ rot_z(180) @ translate(3, 0, 7.5) @ scale(6, 12, 15)
        C4_vertices = transform_cube(unit_cube_vertices, matrix_C4)
        draw_cube(ax, C4_vertices, unit_cube_faces, 'lightcoral')
        
        # 添加信息
        ax.text2D(0.05, 0.95, "cube", transform=ax.transAxes, color='gold')
        ax.text2D(0.05, 0.90, "right arm", transform=ax.transAxes, color='royalblue')
        ax.text2D(0.05, 0.85, "left arm", transform=ax.transAxes, color='firebrick')
        ax.text2D(0.05, 0.80, f"angle_right: {angle_right:.1f}°", transform=ax.transAxes, color='black')
        ax.text2D(0.05, 0.75, f"min arm_offset: {arm_offset:.1f}mm", transform=ax.transAxes, color='black')
        
        fig.canvas.draw_idle()
    
    # 添加滑块
    axcolor = 'lightgoldenrodyellow'
    ax_right_angle = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    slider_angle = Slider(ax_right_angle, 'angle', 0, 90, valinit=angle_right)
    slider_angle.on_changed(update)
    
    # 初始绘图
    update(None)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 预计算最小 arm_offset 值
    print("开始预计算最小 arm_offset...")
    start_time = time.time()
    
    angles = np.arange(0, 91, 1)
    min_offsets = []
    
    for angle in angles:
        min_offset = calculate_min_offset(angle, 90 - angle)
        min_offsets.append(min_offset)
        # print(f"角度: {angle :.1f}°, 最小 arm_offset: {min_offset:.1f}mm")
    
    end_time = time.time()
    print(f"预计算完成! 耗时: {end_time - start_time:.2f}秒")
    print(min_offsets)
    
    # 绘制最小 arm_offset 函数
    plt.figure(figsize=(10, 6))
    plt.plot(angles, min_offsets, 'b-', linewidth=2)
    plt.title('min arm_offset')
    plt.xlabel('angle_right (deg)')
    plt.ylabel('min arm_offset (mm)')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    
    # 创建交互式可视化
    create_interactive_visualization(min_offsets)