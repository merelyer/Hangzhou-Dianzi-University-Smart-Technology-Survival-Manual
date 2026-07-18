# 魔方定位程序（半成品，感觉这个还是手动做比较精确）
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def draw_polygon(img, points, color=(0, 255, 0), thickness=2):
    """在图像上绘制多边形（六边形）"""
    if len(points) < 3:
        return img
        
    # 创建点的副本以确保排序不会影响原始数据
    points = points.copy()
    
    # 按y坐标排序
    points = sorted(points, key=lambda x: x[1])
    
    # 分组为上、中、下三行点
    upper = sorted(points[:2], key=lambda x: x[0])
    middle = sorted(points[2:4], key=lambda x: x[0])
    lower = sorted(points[4:], key=lambda x: x[0])
    
    # 重新排序：左上、右上、左中、右中、左下、右下
    points = np.array([upper[0], upper[1], middle[0], middle[1], lower[0], lower[1]])
    
    # 绘制多边形边界
    for i in range(6):
        start_point = tuple(points[i])
        end_point = tuple(points[(i+1) % 6])
        cv2.line(img, start_point, end_point, color, thickness)
    
    return img

def combine_diff_results(img1, img2, img3):
    """计算1-2的diff和1-3的diff，然后合并差异结果"""
    # 计算1-2的差异
    diff12 = cv2.absdiff(img1, img2)
    
    # 计算1-3的差异
    diff13 = cv2.absdiff(img1, img3)
    
    # 合并差异结果：取两个差异图中较大的值
    diff_combined = cv2.max(diff12, diff13)
    
    # 转换为灰度图并阈值化
    diff_gray = cv2.cvtColor(diff_combined, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作减少噪点
    kernel = np.ones((32, 32), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh, diff_combined

def compare_images(img1_path, img2_path, img3_path):
    
    # 读取图像，都一样大
    logging.info(f"读取图像: {img1_path}, {img2_path}, {img3_path}")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)
    
    if img1 is None or img2 is None or img3 is None:
        logging.error("无法读取图像，请检查文件路径")
        return
    
    
    # 合并1-2和1-3的差异结果
    thresh, diff_combined = combine_diff_results(img1, img2, img3)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建轮廓可视化图像
    contour_img = np.zeros_like(img1)
    
    # 创建用于绘制六边形的图像
    poly_img = np.zeros_like(img1)
    
    for i, contour in enumerate(contours):
        # 过滤小的噪点区域
        if cv2.contourArea(contour) > 300:  
            # 绘制轮廓
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
            
            # 为轮廓添加文本标签
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(contour_img, f"C{i+1}", (cX, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算六边形的6个顶点
            points = []
            points.append((x, y))                    # 左上
            points.append((x + w, y))                # 右上
            points.append((x, y + h//2))             # 左中
            points.append((x + w, y + h//2))         # 右中
            points.append((x, y + h))                # 左下
            points.append((x + w, y + h))            # 右下
            
            # 绘制六边形
            draw_polygon(img3, points, color=(0, 0, 255), thickness=2)
            poly_img = draw_polygon(poly_img, points, color=(0, 255, 255), thickness=2)
            
            # 为六边形添加标签
            cv2.putText(poly_img, f"P{i+1}", (x + w//2, y + h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # 可视化结果
    plt.figure(figsize=(10, 7))
    
    # 合并后的处理结果
    plt.subplot(221), plt.imshow(diff_combined, cmap='jet')
    plt.title('合并差异热力图'), plt.axis('off')
    
    plt.subplot(222), plt.imshow(thresh, cmap='gray')
    plt.title('合并差异二值图'), plt.axis('off')
    
    plt.subplot(223), plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('检测到的轮廓'), plt.axis('off')
    
    plt.tight_layout()
    
    # 单独显示多边形标记图像
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(poly_img, cv2.COLOR_BGR2RGB))
    plt.title('多边形标记概览图'), plt.axis('off')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    # 替换为你的实际图像路径
    img1_path = '../img/testcase/1/captured_image_1.jpg'
    img2_path = '../img/testcase/1/captured_image_2.jpg'
    img3_path = '../img/testcase/1/captured_image_3.jpg'
    
    compare_images(img1_path, img2_path, img3_path)