# python版本魔方颜色识别程序设计目标（开发中，尚未详细测试,目前还不保证都稳定）：
# 对光照条件不敏感
# 可处理魔方表面的轻微反光问题
# 对魔方配色不敏感
# 自动剔除魔方表面的LOGO、污渍、缺损
# 兼容实色、贴纸魔方
# 运行速度快，在低配置的电脑上，识别时间＜50ms

# C语言优化版本设计目标：
# 内存使用量＜2MB
# 在嵌入式系统上，识别时间＜50ms

import cv2
import numpy as np
import json
import time
import logging
import colorsys
import math
from collections import defaultdict

# 观察到cv2.cvtColor函数在第一次调用时耗时非常长，后续耗时可几乎忽略
# 2025-06-06 23:38:20,404 - __main__ - INFO - 转换为Lab颜色空间耗时: 86.37ms
# 2025-06-06 23:38:20,404 - __main__ - INFO - 转换为Lab颜色空间耗时: 0.01ms
# 2025-06-06 23:38:20,404 - __main__ - INFO - 转换为Lab颜色空间耗时: 0.00ms
# 2025-06-06 23:38:20,404 - __main__ - INFO - 转换为Lab颜色空间耗时: 0.00ms
# ......
# 预热OpenCV颜色转换模块
cv2.cvtColor(np.uint8([[[0, 0, 0]]]), cv2.COLOR_BGR2LAB)  # 第一次调用，耗时可能较长，但后续调用会很快

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 先聚类，再合并相似的颜色，合并后挑选主色调
def extract_dominant_color(image, need_plot=False):
    COLOR_DISTANCE_THRESHOLD = 30.0  # 定义颜色相似度阈值（根据实际需求调整）， BGR空间欧氏距离阈值
    initial_K = 6 # 设置初始聚类数量（可根据图像复杂度调整）

    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.75)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-means聚类
    compactness, labels, centers = cv2.kmeans(
        pixels, initial_K, None, criteria, 1, flags
    )
    
    # 计算每个聚类的像素数量
    counts = np.bincount(labels.flatten())
    
    # 聚类合并阶段（动态减少过细分割）
    merged_centers = []
    merged_counts = []
    merged_indices = []  # 记录原始聚类索引

    # 初始化：每个聚类作为一个独立组
    for i in range(len(centers)):
        merged_centers.append(centers[i])
        merged_counts.append(counts[i])
        merged_indices.append([i])
    
    # 合并相近颜色组
    changed = True
    while changed and len(merged_centers) > 1:
        changed = False
        min_dist = float('inf')
        merge_pair = (0, 0)
        
        # 查找最近的两个组
        for i in range(len(merged_centers)):
            for j in range(i+1, len(merged_centers)):
                dist = np.linalg.norm(merged_centers[i] - merged_centers[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)
        
        # 如果最近距离小于阈值则合并
        if min_dist < COLOR_DISTANCE_THRESHOLD:
            i_idx, j_idx = merge_pair
            
            # 计算合并后的新中心（加权平均）
            total_count = merged_counts[i_idx] + merged_counts[j_idx]
            new_center = (merged_centers[i_idx] * merged_counts[i_idx] + 
                          merged_centers[j_idx] * merged_counts[j_idx]) / total_count
            
            # 创建新组
            merged_centers.append(new_center)
            merged_counts.append(total_count)
            merged_indices.append(merged_indices[i_idx] + merged_indices[j_idx])
            
            # 移除旧组（先移除索引大的，再移除小的）
            removed_indices = sorted([i_idx, j_idx], reverse=True)
            for idx in removed_indices:
                merged_centers.pop(idx)
                merged_counts.pop(idx)
                merged_indices.pop(idx)
            
            changed = True

    # 找出主色组
    dominant_idx = np.argmax(merged_counts)
    dominant_color = merged_centers[dominant_idx].astype(np.uint8)
    dominant_percentage = merged_counts[dominant_idx] / np.sum(merged_counts)

    # 创建主色像素映射
    mask = None
    if need_plot:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # 解决维度问题：直接使用展平的labels
        flat_labels = labels.flatten()
        
        # 初始化布尔数组，与flat_labels形状一致
        label_mask = np.zeros_like(flat_labels, dtype=bool)
        
        # 标记所有属于主色组的像素
        for orig_idx in merged_indices[dominant_idx]:
            label_mask = np.logical_or(label_mask, (flat_labels == orig_idx))
        
        # 直接使用展平后的mask.flat赋值
        mask.flat[label_mask] = 255

    return dominant_color, dominant_percentage, mask

def process_image_block(block, block_number,need_plot):
    """
    处理单个图像块，提取主色并用透明粉色覆盖非主色像素
    注意block_number和block_idx不一样，block_idx是图像处理时的块顺序，block_number是魔方求解算法要求的顺序
    :param block: 输入图像块 (30x30像素)
    :param block_number: 块编号
    :return: 处理后的图像块数据和信息
    """
    # 提取主色调和主色像素掩膜
    dominant_color, dominant_percentage, mask = extract_dominant_color(block, need_plot)
    
    # 打印结果
    logger.debug(f"块 {block_number}: BGR={dominant_color}, 占比={dominant_percentage:.1%}")
    if need_plot:
        # 创建掩膜 
        gray_overlay = np.full_like(block, (255, 0, 255), dtype=np.uint8)  
        alpha = 0.9  # 透明度
        
        # 反转掩膜（非主色区域为True）
        non_dominant_mask = (mask == 0)
        
        # 使用透明度混合覆盖粉色
        block[non_dominant_mask] = cv2.addWeighted(
            block[non_dominant_mask], 1 - alpha,
            gray_overlay[non_dominant_mask], alpha,
            0
        )
        
        # 计算主色的相对亮度
        b, g, r = dominant_color
        brightness = (0.299 * r) + (0.587 * g) + (0.114 * b)
        
        # 根据主色亮度选择对比色
        # 亮色用黑色标注，暗色用白色标注
        contrast_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        # 标记序号（使用对比色文字）
        cv2.putText(block, str(block_number), (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, contrast_color, 1)  # 对比色文本
    
    return {
        'block': block,
        'color': dominant_color,
    }
# 创建固定大小的聚类函数（每个聚类必须恰好9个元素）
def fixed_size_kmeans(colors_rgb, fixed_centers):
    # 检查输入数据必须是54个
    if len(colors_rgb) != 54:
        logger.error("Error: Input must contain exactly 54 samples")
        return None
    
    # 转换颜色为HSV格式
    n_colors = 54
    hsv_colors = np.zeros((n_colors, 3))
    for i, rgb in enumerate(colors_rgb):
        r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_colors[i] = [h, s, v]
    
    # 获取所有样本的饱和度
    saturations = hsv_colors[:, 1]
    
    # 1. 找出6个中心点中饱和度最低的那个
    center_saturations = saturations[fixed_centers]
    min_center_idx_in_centers = np.argmin(center_saturations)
    background_center_index = fixed_centers[min_center_idx_in_centers]
    
    # 2. 创建需要保留的中心点列表（剩下的5个）
    protected_centers = [center for i, center in enumerate(fixed_centers) 
                         if i != min_center_idx_in_centers]
    
    # 3. 找出所有非中心点中饱和度最低的8个
    non_center_indices = [i for i in range(n_colors) if i not in fixed_centers]
    non_center_saturations = saturations[non_center_indices]
    
    # 获取8个最低饱和度的非中心点索引
    sorted_non_center_indices = np.argsort(non_center_saturations)[:8]
    background_indices = [non_center_indices[i] for i in sorted_non_center_indices]
    
    # 4. 构建9个背景样本索引（1个中心点 + 8个其他）
    background_indices = [background_center_index] + background_indices
    
    # 5. 主色样本包括所有其他点和剩余的5个中心点
    main_indices = [i for i in range(n_colors) if i not in background_indices]
    main_colors = colors_rgb[main_indices]
    main_hsv = hsv_colors[main_indices]
    
    # 创建映射：从主色样本位置回到原始索引
    idx_mapping = {pos: orig_idx for pos, orig_idx in enumerate(main_indices)}
    
    # 将色相转换为环形坐标
    hues = main_hsv[:, 0] * 2 * np.pi
    hue_features = np.column_stack((np.cos(hues), np.sin(hues)))
    
    # 1. 设置固定中心点（5个剩余的预定义中心点）
    # 找出这些中心点在新主色样本中的位置
    center_positions = {}
    for center in protected_centers:
        for pos, orig_idx in idx_mapping.items():
            if orig_idx == center:
                center_positions[center] = pos
                break
    
    # 准备初始聚类中心
    centers = []
    for center in protected_centers:
        pos = center_positions[center]
        h, s, v = main_hsv[pos]
        hue_rad = h * 2 * np.pi
        centers.append([math.cos(hue_rad), math.sin(hue_rad)])
    centers = np.array(centers)
    
    # 2. 初始化标签数组
    n_main = len(main_colors)
    labels = -1 * np.ones(n_main, dtype=int)
    
    # 3. 固定中心点分配到对应聚类
    for cluster_id, center in enumerate(protected_centers):
        pos = center_positions[center]
        labels[pos] = cluster_id
    
    # 4. KMeans迭代
    cluster_centers = centers.copy()
    max_iter = 100
    tol = 1e-4
    
    for iter_num in range(max_iter):
        # 分配点到聚类（除了固定的中心点）
        for i in range(n_main):
            if labels[i] != -1:  # 跳过固定的点
                continue
                
            min_dist = float('inf')
            best_cluster = -1
            
            # 找出最近的聚类中心
            for cluster_id, center in enumerate(cluster_centers):
                dist = np.linalg.norm(hue_features[i] - center)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_id
            
            labels[i] = best_cluster
        
        # 更新聚类中心
        cluster_centers_prev = cluster_centers.copy()
        
        for cluster_id in range(5):
            cluster_points = hue_features[labels == cluster_id]
            if len(cluster_points) > 0:
                cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
        
        # 检查收敛条件
        center_diff = np.sum([np.linalg.norm(cluster_centers[i] - cluster_centers_prev[i])
                              for i in range(5)])
        if center_diff < tol:
            break
    
    # 5. 创建聚类索引结果
    clusters = []
    
    # 背景聚类
    clusters.append(background_indices)
    
    # 主色聚类
    for cluster_id in range(5):
        # 获取该聚类在主色样本中的位置
        cluster_positions = np.where(labels == cluster_id)[0]
        # 转换为主色样本中的原始索引
        cluster_indices = [idx_mapping[pos] for pos in cluster_positions]
        clusters.append(cluster_indices)
    return clusters

def color_detect(img_list, points, need_plot=False):
    # ---------------- 步骤1：透视变换，并且拆分54个魔方色块，识别每个色块的主色调 ----------------
    start_time = time.time()
    width, height = 90, 90
    group1_indices = [0, 1, 3, 4]
    group2_indices = [1, 2, 4, 5]
    dst_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) 
    imgOutput1_list = [None] * 3
    imgOutput2_list = [None] * 3

    # 透视变换
    for i in range(3):
        img = img_list[i]
        pts1 = np.float32([points[i] for i in group1_indices])
        matrix1 = cv2.getPerspectiveTransform(pts1, dst_pts)
        imgOutput1 = cv2.warpPerspective(img, matrix1, (width, height))
        imgOutput1_list[i] = imgOutput1

        pts2 = np.float32([points[i] for i in group2_indices])
        matrix2 = cv2.getPerspectiveTransform(pts2, dst_pts)
        imgOutput2 = cv2.warpPerspective(img, matrix2, (width, height))
        imgOutput2_list[i] = imgOutput2

    #  调整BGR彩色图像的亮度，使用了对光照不敏感的颜色分类方式，不需要分了
    # # imgOutput1和imgOutput2的光照条件有差异，导致亮度不同，三张pts1是完全一致的，三张pts2也是完全一致的。
    # # 需要进行亮度调整，以三张imgOutput1为模板，将imgOutput2的亮度调整到一致的状态
    # # 三张imgOutput2的调整方式应当完全一致
    # # 计算三张imgOutput1的总体均值和标准差（LAB空间）
    # L1_all = []
    # for img in imgOutput1_list:
    #     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #     L, A, B = cv2.split(lab)
    #     L1_all.append(L)
    # L1_concatenated = np.concatenate(L1_all)
    # mean1_L = np.mean(L1_concatenated)
    # std1_L = np.std(L1_concatenated)

    # # 计算三张imgOutput2的总体均值和标准差（LAB空间）
    # L2_all = []
    # for img in imgOutput2_list:
    #     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #     L, A, B = cv2.split(lab)
    #     L2_all.append(L)
    # L2_concatenated = np.concatenate(L2_all)
    # mean2_L = np.mean(L2_concatenated)
    # std2_L = np.std(L2_concatenated)

    # logger.debug(f"mean1_L={mean1_L:.1f}, std1_L={std1_L:.1f}")
    # logger.debug(f"mean2_L={mean2_L:.1f}, std2_L={std2_L:.1f}")

    # # 为所有imgOutput2应用相同的亮度调整（使用总体统计量）
    # for i in range(3):
    #     lab = cv2.cvtColor(imgOutput2_list[i], cv2.COLOR_BGR2LAB)
    #     L, A, B = cv2.split(lab)
    #     L = L.astype(np.float32)
        
    #     # 线性变换：对齐imgOutput1的亮度和对比度
    #     L = (L - mean2_L) * (std1_L / (std2_L + 1e-7)) + mean1_L
    #     L = np.clip(L, 0, 255).astype(np.uint8)  # 确保值在[0,255]范围内
        
    #     lab_adjusted = cv2.merge([L, A, B])
    #     imgOutput2_list[i] = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    logger.info(f"透视变换耗时: {(time.time() - start_time)*1000:.2f}ms")

    start_time = time.time()
    all_blocks = [None] * 54
    block_idx = 0
    index_mapping = [
        6,3,0,7,4,1,8,5,2,
        47,50,53,46,49,52,45,48,51,
        17,16,15,14,13,12,11,10,9,
        26,25,24,23,22,21,20,19,18,
        38,41,44,37,40,43,36,39,42,
        27,28,29,30,31,32,33,34,35
    ]

    for i in range(3):
        imgOutput1 = imgOutput1_list[i]
        imgOutput2 = imgOutput2_list[i]
        for imgOutput in [imgOutput1, imgOutput2]:
            for y in range(0, height, 30):
                for x in range(0, width, 30):
                    block = imgOutput[y:y+30, x:x+30]
                    block_number = index_mapping[block_idx]
                    processed_block = process_image_block(block, block_number, need_plot)
                    all_blocks[block_number] = processed_block# 直接按 block_number 存储到对应位置
                    block_idx += 1
    logger.info(f"主色调提取耗时: {(time.time() - start_time)*1000:.2f}ms")
    
    # ---------------- 步骤2：对54组主色调数据进行聚类处理 ----------------
    start_time = time.time()
    colors_rgb = []  # 直接使用RGB值
    for block_data in all_blocks:
        b, g, r = block_data['color']  # OpenCV读取的是BGR顺序
        colors_rgb.append([r, g, b])   # 转换为RGB顺序
    # 转换类型
    colors_rgb_np = np.array(colors_rgb, dtype=np.float32)
    # 确定初始聚类中心
    fixed_centers = [4, 13, 22, 31, 40, 49] 
    # 执行固定大小的聚类
    clusters = fixed_size_kmeans(colors_rgb_np, fixed_centers)
    # 验证每个聚类是否包含初始中心点，是否包含9个元素
    cube_str_list = ['X'] * 54
    block_colors = [None] * 54 # 仅供调试
    center_name = ['U', 'R', "F", "D", "L", "B"] 
    valid_clusters = True
    for cluster_id, blocks in enumerate(clusters):
        if len(blocks) != 9:
            valid_clusters = False
            logger.error(f"聚类 {cluster_id} 包含 {len(blocks)} 个元素 (需要 9 个): {blocks}")
        else:
            logger.info(f"聚类 {cluster_id} 包含 9 个元素: {blocks}")
            # 检查哪个固定中心点在此聚类中
            for i, center_idx in enumerate(fixed_centers):
                if center_idx in blocks:
                    block_color = all_blocks[center_idx]['color']
                    for block_idx in blocks:
                        cube_str_list[block_idx] = center_name[i]
                        block_colors[block_idx] = block_color
                    break
            else:
                valid_clusters = False
                logger.error(f"聚类 {cluster_id} 不包含任何固定中心点!")

    logger.info(f"魔方颜色分类耗时: {(time.time() - start_time)*1000:.2f}ms")
    # 定义布局映射表 [行][列] = 块索引
    result_rows = 9
    result_cols = 12
    layout_mapping = [
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
    if valid_clusters:
        cube_str = ''.join(cube_str_list)
        logger.info(f"魔方识别结果：{cube_str}")
        if logger.isEnabledFor(logging.DEBUG):
            for row in range(result_rows):
                out = [' '] * result_cols
                for col in range(result_cols):
                    block_idx = layout_mapping[row][col]
                    if block_idx <0 or block_idx >= len(all_blocks):
                        continue
                    out[col] = cube_str_list[block_idx]
                logger.debug(' '.join(out))
    
    if need_plot:
        # 创建9行12列的拼接图像
        block_size = 30
        result_width = result_cols * block_size
        result_height = result_rows * block_size 
        result = np.zeros((result_height, result_width, 3), dtype=np.uint8)

        
        # 按照布局映射表放置图像块
        for row in range(result_rows):
            for col in range(result_cols):
                block_idx = layout_mapping[row][col]
                # 跳过空位置
                if block_idx <0 or block_idx >= len(all_blocks):
                    continue
                # 获取对应的图像块
                block_data = all_blocks[block_idx]['block']
                # 计算放置位置
                y_start = row * block_size
                y_end = y_start + block_size
                x_start = col * block_size
                x_end = x_start + block_size
                # 放置图像块
                result[y_start:y_end, x_start:x_end] = block_data

        # 用识别出来的颜色绘制魔方的展开图
        if valid_clusters:
            block_size = 8
            result_width = result_cols * block_size
            result_height = result_rows * block_size 
            offset_x = 250
            offset_y = 190
            for row in range(result_rows):
                for col in range(result_cols):
                    block_idx = layout_mapping[row][col]
                    # 跳过空位置
                    if block_idx <0 or block_idx >= len(all_blocks):
                        continue
                    # 获取对应的颜色
                    block_color = block_colors[block_idx]
                    # 计算放置位置
                    y_start = row * block_size + offset_y
                    y_end = y_start + block_size
                    x_start = col * block_size + offset_x
                    x_end = x_start + block_size
                    # 绘制矩形
                    result[y_start:y_end, x_start:x_end] = block_color

        # 显示和保存结果
        # 1. 缩放原始图像为统一尺寸
        resized_imgs = []
        for img in img_list:
            resized = cv2.resize(img, (320, 180))
            resized_imgs.append(resized)

        # 2. 缩放结果图像
        resized_result = cv2.resize(result, (960, 720))  # 宽度与三张图总和相同

        # 3. 创建顶部行：三张原始图像水平拼接
        top_row = np.hstack(resized_imgs)

        # 4. 创建完整布局：顶部行 + 结果图像垂直拼接
        if top_row.shape[1] != resized_result.shape[1]:
            # 确保宽度相同（960px）
            resized_result = cv2.resize(resized_result, (960, 720))
            
        final_image = np.vstack((top_row, resized_result))

        # 5. 添加分隔线和标题（可选）
        # 添加水平分隔线
        cv2.line(final_image, (0, 180), (final_image.shape[1], 180), (0, 255, 0), 2)

        # 添加标题文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(3):
            cv2.putText(final_image, f'Original {i}', (10 + i*320, 20), 
                        font, 0.7, (0, 255, 0), 2)
        cv2.putText(final_image, 'Cube Blocks Result', (10, 180 + 30), 
                    font, 0.7, (0, 255, 0), 2)

        # 6. 在单个窗口中显示拼接后的图像
        cv2.namedWindow('Image Composition', cv2.WINDOW_NORMAL)
        cv2.imshow('Image Composition', final_image)
        cv2.resizeWindow('Image Composition', 640, 600)
        #cv2.imwrite('../temp/blocks_with_colors_results_annotated.jpg', result)

        #cv2.waitKey(0)

        # 使用matplotlib，以三维图的形式绘制RGB空间的聚类结果
        # 将RGB转换为HSV
        colors_hsv = []
        for color in colors_rgb:
            # 归一化RGB到[0,1]
            r, g, b = [x/255.0 for x in color]
            # 转换为HSV（返回的h在[0,1], s在[0,1], v在[0,1]）
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # 转换到常规HSV范围：H[0-360], S[0-100], V[0-100]
            h = h * 360
            s = s * 100
            v = v * 100
            colors_hsv.append((h, s, v))
        
        # 提取HSV分量
        h_vals = [c[0] for c in colors_hsv]  # 色相分量 (0-360)
        s_vals = [c[1] for c in colors_hsv]  # 饱和度分量 (0-100)
        v_vals = [c[2] for c in colors_hsv]  # 明度分量 (0-100)

        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # ====== 新增：转换HSV到锥形笛卡尔坐标 ======
        def hsv_to_cone(h, s, v):
            """将HSV值转换为锥形笛卡尔坐标"""
            # 将色相转换为弧度（0-2π）
            h_rad = np.radians(h)
            # 圆锥底面的半径（与饱和度成正比）
            radius = (s / 100.0) * (v / 100.0)  # 考虑明度对饱和半径的影响
            # 转换为笛卡尔坐标
            x = radius * np.cos(h_rad)
            y = radius * np.sin(h_rad)
            z = v / 100.0  # 高度归一化到[0,1]
            return x, y, z

        # 转换所有HSV点到锥形坐标
        points_x, points_y, points_z = [], [], []
        for h, s, v in zip(h_vals, s_vals, v_vals):
            x, y, z = hsv_to_cone(h, s, v)
            points_x.append(x)
            points_y.append(y)
            points_z.append(z)

        # ====== 新增：创建圆锥表面网格 ======
        def create_cone_mesh():
            """创建表示圆锥的网格"""
            # 创建网格点
            h = np.linspace(0, 2*np.pi, 50)  # 色相角度
            v = np.linspace(0, 1, 10)        # 明度高度
            H, V = np.meshgrid(h, v)         # 创建网格
            
            # 计算表面点坐标
            X = V * np.cos(H)  # 当s=100%时，半径=高度
            Y = V * np.sin(H)
            Z = V              # z轴即明度
            
            return X, Y, Z

        # 创建圆锥网格
        X, Y, Z = create_cone_mesh()

        # 绘制半透明的圆锥表面
        ax.plot_surface(X, Y, Z, color='gray', alpha=0.1, rstride=1, cstride=2, linewidth=0)
        ax.plot_wireframe(X, Y, Z, color='black', alpha=0.3, rcount=3, ccount=12)

        # 定义6种不同的标记区分聚类
        markers = ['o', 'v', '^', '<', '>', 's']

        # 绘制每个聚类（使用实际颜色和指定标记）
        for cluster_id in range(6):
            cluster_indices = clusters[cluster_id]
            
            # 获取当前聚类的HSV坐标
            cluster_x = [points_x[i] for i in cluster_indices]
            cluster_y = [points_y[i] for i in cluster_indices]
            cluster_z = [points_z[i] for i in cluster_indices]
            
            # 转换为归一化的实际颜色（使用原始RGB）
            actual_colors = []
            for idx in cluster_indices:
                r, g, b = colors_rgb[idx]  # 原始RGB值
                actual_colors.append([r/255.0, g/255.0, b/255.0])
            
            # 为图例创建一个虚拟点
            ax.scatter([], [], [], 
                    marker=markers[cluster_id], 
                    s=100,
                    color='grey',          # 图例点统一用灰色
                    edgecolor='black',     # 保留黑色边框
                    label=f'Cluster {cluster_id}')
            
            # 绘制实际数据点（使用实际颜色和聚类标记）
            ax.scatter(cluster_x, cluster_y, cluster_z, 
                    c=actual_colors,        # 实际颜色（RGB）
                    marker=markers[cluster_id], 
                    s=100, 
                    edgecolor='black',
                    depthshade=True)

        # 设置坐标轴标签和范围
        ax.set_xlabel('X (S*cos(H))', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (S*sin(H))', fontsize=12, labelpad=10)
        ax.set_zlabel('Value (V)', fontsize=12, labelpad=10)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(0, 1)

        # 添加颜色圆形标记
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            ax.quiver(0, 0, 0, 
                    np.cos(angle), np.sin(angle), 0,
                    color=colorsys.hsv_to_rgb(angle/(2*np.pi), 1, 1),
                    length=1.1, 
                    arrow_length_ratio=0.05,
                    linewidth=1.5)

        ax.set_title('Cube Colors in HSV Cone Space', fontsize=16)

        # 添加图例
        ax.legend(loc='upper right', fontsize=10, markerscale=0.8)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.4)

        # 调整视角使其从上方观察
        ax.view_init(elev=20, azim=65)
        ax.dist = 8  # 稍微缩小距离

        # 保存并显示图表
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        cv2.destroyAllWindows()
    if valid_clusters:
        return cube_str
    else:
        return None


if __name__ == "__main__":
    # 绘图比较慢，所以只在调试时绘制
    need_plot = True
    if need_plot:
        logging.getLogger('matplotlib').propagate = False  # 完全阻止传播
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    # 读取JSON文件
    try:
        with open('points_config.json', 'r') as f:
            points = json.load(f)
        points = np.array(points)
    except Exception as e:
        logger.error(f"错误：无法读取JSON文件 - {str(e)}")
        exit()

    if len(points) < 6:
        logger.error("错误：JSON文件中需要至少6个点")
        exit()

    # 读取图片
    img_list = []
    for i in (1,2,3):
        img_path = f'../img/testcase/1/captured_image_{i}.jpg'
        img = cv2.imread(img_path)
        img_list.append(img)
        if img is None:
            logger.error(f"错误：无法读取图像文件 '{img_path}'")
            exit()

    # 识别颜色
    time_start = time.time()
    color_detect(img_list, points, need_plot)
    logger.info(f"处理完成! 时间: {time.time()-time_start:.3f}秒")
