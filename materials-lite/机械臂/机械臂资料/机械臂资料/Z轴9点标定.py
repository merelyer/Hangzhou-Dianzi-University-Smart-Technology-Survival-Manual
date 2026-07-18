import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class ZCoordinatePredictor:
    def __init__(self, degree=2):
        """
        初始化Z坐标预测器
        :param degree: 多项式回归的阶数（默认2阶）
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
    
    def fit(self, pixel_coords, z_coords):
        """
        训练预测模型
        :param pixel_coords: 9个像素坐标的数组，形状为(9, 2)
        :param z_coords: 对应的9个Z坐标值，形状为(9,)
        """
        # 生成多项式特征
        X_poly = self.poly.fit_transform(pixel_coords)
        # 训练线性回归模型
        self.model.fit(X_poly, z_coords)
    
    def predict(self, pixel_coord):
        """
        预测目标点的Z坐标
        :param pixel_coord: 目标点的像素坐标，形状为(2,)
        :return: 预测的Z坐标
        """
        # 转换为多项式特征
        point_poly = self.poly.transform([pixel_coord])
        # 预测Z坐标
        return self.model.predict(point_poly)[0]

# 示例使用
if __name__ == "__main__":
    # 1. 准备标定数据（9个点的像素坐标和对应的Z坐标）
    # 格式: [[u1, v1], [u2, v2], ...], [z1, z2, ...]
    pixel_coords = np.array([
        [157 , 64],   [579 , 77],   [1044 , 46],
        [178 , 403],   [628 , 393],   [1072 , 360],
        [154 , 607],   [650 , 536],    [1098 , 625]
    ])
    z_coords = np.array([152.201,151.501,152.501,150.901,149.701,150.853,151.053,149.053,149.853])

    # 2. 创建并训练预测模型
    predictor = ZCoordinatePredictor(degree=2)
    predictor.fit(pixel_coords, z_coords)

    # 3. 预测新目标点的Z坐标
    test_point = np.array([650, 300])  # 测试点像素坐标
    predicted_z = predictor.predict(test_point)
    
    print(f"目标点像素坐标 ({test_point[0]}, {test_point[1]})")
    print(f"预测的Z坐标: {predicted_z:.2f} mm")