import numpy as np

class ArmCalibrator:
    def __init__(self):
        self.H = None  # 变换矩阵
        
    def calibrate(self, src_points, dst_points):
        """
        通过4个对应点计算变换矩阵
        :param src_points: 机械臂坐标系中的4个点 [左上, 左下, 右上, 右下]
        :param dst_points: 实际坐标系中的对应4个点 [左上, 左下, 右上, 右下]
        """
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("需要提供4个源点和4个目标点")
            
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        
        # 构建线性方程组 A * h = b
        A = []
        b = []
        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
            b.extend([u, v])
        
        A = np.array(A, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        
        # 解线性方程组
        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 使用最小二乘法作为备选方案
            h = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # 重构变换矩阵 (3x3)
        self.H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1]
        ], dtype=np.float32)
    
    def transform(self, point):
        """
        将机械臂坐标系的点转换到实际坐标系
        :param point: 机械臂坐标系中的点 (x, y)
        :return: 实际坐标系中的点 (u, v)
        """
        if self.H is None:
            raise RuntimeError("请先进行标定")
        
        x, y = point
        src_vec = np.array([x, y, 1], dtype=np.float32)
        dst_vec = self.H @ src_vec
        
        # 齐次坐标归一化
        u = dst_vec[0] / dst_vec[2]
        v = dst_vec[1] / dst_vec[2]
        return u, v

# 使用示例
if __name__ == "__main__":
    # 1. 准备标定数据（示例数据）
    # 机械臂坐标系中的点 [左上, 左下, 右上, 右下]
    src_points = [
        (1, 14),   # 左上
        (1, 1),   # 左下
        (10, 14),   # 右上
        (10, 1)    # 右下
    ]
    
    # 实际物理坐标系中的对应点 [左上, 左下, 右上, 右下]
    dst_points = [
        (-485.011,-72.702),    # 左上
        (-225.16,-71.033),    # 左下
        (-486.678,106.837),    # 右上
        (-227.454,108.095)     # 右下
    ]
    
    # 2. 创建并初始化标定器
    calibrator = ArmCalibrator()
    calibrator.calibrate(src_points, dst_points)
    
    # 3. 测试点转换（机械臂坐标系中的点）
    test_point = (5.5, 7.5)  # 托盘中心点
    transformed_point = calibrator.transform(test_point)
    print(transformed_point[0])
    # print(f"机械臂坐标 {test_point} -> 实际坐标 {transformed_point}")