import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main_color_extraction(img_path, h_bins=30, s_bins=8):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 计算总像素数
    total_pixels = img.shape[0] * img.shape[1]
    
    # 计算H-S二维直方图
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    
    # 初始化主色变量
    top_colors = []
    cumulative = 0.0
    # 复制直方图避免修改原始数据
    temp_hist = hist.copy()

    # 最多循环4次，每次提取一个最大值
    for _ in range(4):
        # 如果累计占比已超过50%，提前退出
        if cumulative >= total_pixels * 0.5:
            break
            
        # 找到当前直方图中最大值及其位置
        max_val = np.max(temp_hist)
            
        pos = np.unravel_index(np.argmax(temp_hist), temp_hist.shape)
        
        # 计算当前bin的中心值
        h_center = (pos[0] + 0.5) * (180 / h_bins)
        s_center = (pos[1] + 0.5) * (256 / s_bins)
        
        # 累加像素值
        cumulative += max_val
        
        # 添加到颜色列表
        top_colors.append((h_center, s_center, max_val))
        
        # 将已提取的位置置零避免重复选择
        temp_hist[pos] = 0
    
    # 计算实际提取的像素占比
    actual_percentage = cumulative / total_pixels * 100
    
    # 创建主色调结果（假设calculate_dominant函数已实现）
    print(f"颜色占比：{actual_percentage:.1f}%, 包含bin数量：{len(top_colors)}")
    dominant_h, dominant_s = calculate_dominant(top_colors)
    
    # 可视化（假设visualize_results函数已实现）
    visualize_results(img, dominant_h, dominant_s, hist, h_bins, s_bins)

def calculate_dominant(top_colors):
    """计算主色调的加权平均值"""
    total_weight = sum(bin_value for _, _, bin_value in top_colors)
    weighted_h = sum(h * bin_value for h, _, bin_value in top_colors) / total_weight
    weighted_s = sum(s * bin_value for _, s, bin_value in top_colors) / total_weight
    return weighted_h, weighted_s

def visualize_results(img, dominant_h, dominant_s, hist, h_bins, s_bins):
    """结果可视化函数"""
    # 创建主色调块
    color_block = np.zeros((img.shape[0], img.shape[1]//5, 3), dtype=np.uint8)
    color_block_hsv = np.array([[[dominant_h, dominant_s, 200]]], dtype=np.uint8)
    color_block_bgr = cv2.cvtColor(color_block_hsv, cv2.COLOR_HSV2BGR)[0][0]
    color_block[:,:] = color_block_bgr

    # 拼接主色调块
    combined = np.hstack((img, color_block))

    print(f"Dominant: H={dominant_h:.1f}, S={dominant_s:.1f}")
    
    # 显示结果图像
    cv2.imshow('Image with Dominant Color', combined)
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    
    # 显示热力图
    plt.imshow(np.log(hist + 1),  # 使用对数刻度增强可视化
               extent=[0, 256, 0, 180], 
               aspect='auto', 
               cmap='viridis',
               origin='lower')
    
    plt.colorbar(label='Log Pixel Count')
    plt.title('2D Histogram (H-S)')
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    
    # 标记主色调位置
    plt.plot(dominant_s, dominant_h, 'ro', markersize=10)
    plt.text(dominant_s+5, dominant_h, 'Dominant', color='red', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 可轻松修改参数
    H_BINS = 30  # H通道的bin数量
    S_BINS = 8  # S通道的bin数量
    
    main_color_extraction('../temp/2.png', h_bins=H_BINS, s_bins=S_BINS)