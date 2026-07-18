import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 读取图片作为背景
image_path = '../temp/captured_image_1.bmp'
background_image = Image.open(image_path)
img_width, img_height = background_image.size

# 从C程序输出复制结果
data = """
461 239
512 223
507 326
457 331

512 223
578 201
573 320
507 326

578 201
667 173
661 312
573 320

457 331
507 326
503 431
452 424

507 326
573 320
568 440
503 431

573 320
661 312
656 453
568 440

452 424
503 431
498 536
448 517

503 431
568 440
563 561
498 536

568 440
656 453
650 595
563 561

667 173
748 206
745 322
662 308

748 206
809 231
807 333
745 322

809 231
857 250
856 341
807 333

662 308
745 322
742 442
656 448

745 322
807 333
806 438
742 442

807 333
856 341
855 434
806 438

656 448
742 442
738 567
650 595

742 442
806 438
804 546
738 567

806 438
855 434
854 530
804 546

Region  0: H= 65, S=250, dominant=168%
Region  1: H= 66, S=250, dominant=168%
Region  2: H= 65, S=239, dominant=168%
Region  3: H= 65, S=243, dominant=168%
Region  4: H= 64, S=243, dominant=168%
Region  5: H= 64, S=245, dominant=168%
Region  6: H= 65, S=240, dominant=168%
Region  7: H= 65, S=238, dominant=168%
Region  8: H= 65, S=234, dominant=168%
Region  9: H=  1, S=207, dominant=168%
Region 10: H=  0, S=207, dominant=168%
Region 11: H=  1, S=197, dominant=168%
Region 12: H=  1, S=216, dominant=168%
Region 13: H=  0, S=218, dominant=168%
Region 14: H=  1, S=207, dominant=168%
Region 15: H=  0, S=220, dominant=168%
Region 16: H=  0, S=218, dominant=168%
Region 17: H=  1, S=211, dominant=168%

""".strip().split('\n\n')

# 分离坐标数据和区域颜色数据
quad_data = data[:18]  # 前18个块是四边形坐标
color_data = data[18]  # 最后一个块是颜色信息

# 解析颜色信息
color_info = []
for line in color_data.strip().split('\n'):
    match = re.search(r'H=\s*(\d+),\s*S=\s*(\d+)', line)
    if match:
        h = int(match.group(1))
        s = int(match.group(2))
        color_info.append((h, s))

# 计算整体边界
all_points = []
for quad in quad_data:
    for line in quad.strip().split('\n'):
        if line and not line.startswith('Region'):
            x, y = map(int, line.split())
            all_points.append((x, y))

xs, ys = zip(*all_points)
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

# 配置绘图 - 原点在左上角
plt.figure(figsize=(12, 12 * img_height / img_width))
ax = plt.gca()
ax.invert_yaxis()  # 反转Y轴，使原点在左上角
ax.set_aspect('equal')

# 显示背景图片
ax.imshow(background_image, extent=[0, img_width, img_height, 0], alpha=0.8)

# 添加网格线
ax.grid(True, color='gray', linestyle='--', alpha=0.3)

# 绘制四边形和中心点
for i, quad in enumerate(quad_data):
    points = []
    for line in quad.strip().split('\n'):
        if line:
            x, y = map(int, line.split())
            points.append((x, y))
    
    if len(points) == 4:
        # 计算四边形中心点
        center_x = sum(p[0] for p in points) / 4.0
        center_y = sum(p[1] for p in points) / 4.0
        
        # 获取该区域的HSV颜色
        if i < len(color_info):
            h, s = color_info[i]
            
            # 边框颜色
            h1 = h / 180.0
            s1 = s / 255.0
            v1 = 0.5
            border_color = hsv_to_rgb([h1, s1, v1])
            
            # 中心点颜色 
            v2 = 1
            center_color = hsv_to_rgb([h1, s1, v2])
        else:
            border_color = 'black'
            center_color = 'gray'
        
        # 连接成四边形
        poly = points + [points[0]]
        xs, ys = zip(*poly)
        
        # 绘制四边形边
        plt.plot(xs, ys, color=border_color, linewidth=2.5, alpha=0.9)
        
        # 绘制中心点（黑框圆圈）
        circle_outer = plt.Circle((center_x, center_y), radius=18, 
                                 edgecolor='black', facecolor='none', 
                                 linewidth=1.5, zorder=10)
        circle_inner = plt.Circle((center_x, center_y), radius=16, 
                                 edgecolor='none', facecolor=center_color, 
                                 alpha=0.9, zorder=10)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)
        
        # 添加区域编号
        plt.text(center_x, center_y, str(i), color='black', 
                fontsize=9, weight='bold', 
                ha='center', va='center', zorder=11)

# 添加坐标轴和标签
plt.title('图像上的四边形划分与颜色标识 (左上角为原点)', fontsize=16)
plt.xlabel('X坐标', fontsize=12)
plt.ylabel('Y坐标', fontsize=12)

# 设置坐标轴范围
plt.xlim(min(0, min_x) - 50, max(img_width, max_x) + 50)
plt.ylim(max(img_height, max_y) + 50, min(0, min_y) - 50)  # 注意Y轴反转

plt.tight_layout()
plt.savefig('quadrilaterals_with_color_markers.png', dpi=150, bbox_inches='tight')
plt.show()