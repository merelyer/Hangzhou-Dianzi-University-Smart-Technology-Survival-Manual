import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scurve import Scurve, ScurveState, ScurveReturnStatus
# 同侧旋转臂以及手指的联动代码
# 旋转臂控制使用S加减速、手指联动使用凸优化方案
# 凸优化运行较慢，但是使用cvxpy库时，代码比较简洁，适合提前计算而不是实时计算

# 依赖的第三方包
# pip3 install numpy cvxpy matplotlib ecos


# --------------- 旋转臂参数 ---------------
# 转速单位转换 RPM -> 编码器计数/控制周期
RPM_TO_SPEED = 16384.0 / 5000.0 / 60.0
# 旋转臂最大速度，电机本身的，减速之前的(上一版本是1200RPM)
# 1000RPM：表现良好
# 1100RPM：误差较大（编码器值100-300），但是不会碰到魔方
# 1200RPM：误差较大（编码器值100-400），但是不会碰到魔方
ARM_VMAX = 1000 * RPM_TO_SPEED
# 旋转臂最大加速度，电机本身的，减速之前的(上一版本是600/1024=0.586)
ARM_ACCEL = 0.6
# 旋转臂最大加加速度
ARM_JERK  = 0.05
# 旋转臂旋转90度，对应电机编码器的变化量（1/2圈）
ARM_STEPS = 8192 

#  --------------- 手指参数 --------------- 
# 手指移动速度，这个惯性小，也不会变形，可以很快（上一版本为1500）
# 目前的规划算法，1300和1500性能差异不大
FINGER_VMAX  = 1300 * RPM_TO_SPEED
# 手指电机加速度（上一版本为1600/1024 = 1.56）
FINGER_ACCEL = 1.3
# 手指离开魔方表面时的速度，以及手指撞击魔方表面时的速度
# 动作要轻柔，否则魔方可能会轻微变形，影响稳定性
# (如果追求性能，可以调高一些，400-500RPM都没问题，直接最高速度撞上去也勉强能用)
FINGER_V_CUBE_S1  = 200 * RPM_TO_SPEED
FINGER_V_CUBE_S3  = 200 * RPM_TO_SPEED

# --------------- 尺寸参数 ---------------
# 56*√2 = 79.2mm，但是魔方是圆角的，实测值是77mm，留6mm余量，取83mm，机械结构允许的范围是52-86mm
# 手指经过魔方面对角线时，两手指之间的距离
FINGER_DIAG  = 83.0
# 两手指之间的距离的上限
FINGER_LIMIT = 85.0
# 夹紧状态手指位置
FINGER_CLAMP = 53.0
# 魔方的边长，也是运动规划开始、以及结束时，手指的位置
CUBE_SIZE    = 56.0
# 手指零点，位于距离内侧限位点0.5mm的位置，此时两手指间距52mm
FINGER_ZERO  = 52.0
# 手指电机每转一圈，手指移动距离31.4mm，两手指间距增加2 * 31.4mm
FINGER_FACTOR = 16384.0 / (2 * 31.416)

# --------------- 联动参数 ---------------
# 如果出现“求解手指轨迹出错”报错，可以改大一点
FINGER_CLAMP_TIME_S1         = 58  # 1 = 0.2ms
FINGER_ONLY_SEQUANCE_TIME_S1 = 49
FINGER_ONLY_SEQUANCE_TIME_S3 = 49
FINGER_CLAMP_TIME_S3         = 59


def finger_dist2enc(x): 
    return (x - FINGER_ZERO) * FINGER_FACTOR

def finger_enc2dist(x): 
    return x / FINGER_FACTOR + FINGER_ZERO

# 输入：angle，旋转臂角度
# 输出：两手指之间的最小间距，以及是否达到最大值
def arm_angle_to_finger_dist(angle):
    # 手指和魔方之间的安全距离
    gap = 2
    # 手指宽度
    finger_width = 12.0

    if(angle > 90.0 or angle < 0):
        return 0
    # 利用对称性简化计算
    if(angle > 45):
        angle = 90 - angle

    # 四边形ABCD，已知A、C为直角，已知AD、CD、角B(取值范围（0,90°] )，求BC
    # 注意到A、B、C、D四点共圆，且BD为直径
    # 推导可得出 BC = (AD + CD * cos B) / sin B
    AD = CUBE_SIZE / 2 + gap
    CD = finger_width / 2
    angle_B = math.pi/2 - math.radians(angle)

    BC = (AD + CD * math.cos(angle_B)) / math.sin(angle_B)
    # FINGER_MAX = CUBE_SIZE/math.sqrt(2) + gap
    if BC > FINGER_DIAG / 2:
        return FINGER_DIAG
    else:
        return BC * 2


# 编写一个函数，产生一段运动路径finger_encoder_final
# 长度和数组finger_encoder_min_list一致
# 对于第i个数据，finger_encoder_min_list[i] <= finger_encoder_final[i] <= finger_encoder_max_list[i]
# 最大速度（相邻元素的差分）绝对值不能超过 FINGER_VMAX 
# 初速度和末速度必须为0
# 最大加速度（相邻元素的二次差分）绝对值不能超过 FINGER_ACCEL
# 只需要给出这个函数
# 步长i，为0.2ms,运动规划时应该用不到这个信息
# 例如444个数据点，对应444*0.2 = 88.8ms
# 不在意性能，可以使用较为复杂的算法
def generate_finger_trajectory(finger_encoder_min_list, finger_encoder_max_list,
                               finger_speed_limit_time_s1, finger_speed_limit_time_s3):
    N = len(finger_encoder_min_list)
    x = cp.Variable(N)
    
    # 目标函数：最小化加速度变化（平滑轨迹）
    objective = cp.Minimize(cp.sum_squares(x[2:] - 2*x[1:N-1] + x[0:N-2]))
    
    constraints = []
    # 位置约束
    for i in range(N):
        constraints.append(x[i] >= finger_encoder_min_list[i])
        constraints.append(x[i] <= finger_encoder_max_list[i])
    
    # 速度约束（一阶差分）
    for i in range(N-1):
        constraints.append(cp.abs(x[i+1] - x[i]) <= FINGER_VMAX)
    
    # 加速度约束（二阶差分）
    for i in range(1, N-1):
        constraints.append(cp.abs(x[i+1] - 2*x[i] + x[i-1]) <= FINGER_ACCEL)
    
    # 初速度和末速度为零
    constraints.append(x[1] - x[0] == 0)
    constraints.append(x[N-1] - x[N-2] == 0)
    
    # 特定时间段的速度限制（S1区域）
    start_idx_s1 = finger_speed_limit_time_s1 - 20
    end_idx_s1 =   finger_speed_limit_time_s1 + 20
    for i in range(start_idx_s1, end_idx_s1):
        constraints.append(cp.abs(x[i+1] - x[i]) <= FINGER_V_CUBE_S1)
    
    # 特定时间段的速度限制（S3区域）
    start_idx_s3 = finger_speed_limit_time_s3 - 20
    end_idx_s3 =   finger_speed_limit_time_s3 + 20
    for i in range(start_idx_s3, end_idx_s3):
        constraints.append(cp.abs(x[i+1] - x[i]) <= FINGER_V_CUBE_S3)
    
    # 求解问题
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    print(f"求解器状态: {prob.status}")
    if prob.status == cp.OPTIMAL or cp.OPTIMAL_INACCURATE:
        return x.value
    else:
        return None

def calc_motion_adjust_arm(need_plot):
    sc_arm = Scurve(1)

    sc_arm.setPositionOutput(0)
    sc_arm.setPositionTarget(ARM_STEPS) 
    sc_arm.setVelocityStart(0.0)
    sc_arm.setVelocityStop(0.0)
    sc_arm.setVelocityMax(ARM_VMAX)
    sc_arm.setAccelerationMax(ARM_ACCEL)
    sc_arm.setDecelerationMax(ARM_ACCEL)
    sc_arm.setJerkMax(ARM_JERK)
    sc_arm.startProfile()

    arm_encoder_data = []
    arm_speed_data = []
    arm_time_steps = []
    finger_min_pos_list = []
    finger_encoder_min_list = []
    finger_encoder_max_list = []
    step = 0
    # 第一阶段：只有手指移动，旋转臂固定
    finger_speed_limit_time_s1 = FINGER_CLAMP_TIME_S1
    for i in range(FINGER_ONLY_SEQUANCE_TIME_S1 + FINGER_CLAMP_TIME_S1):
        arm_encoder_data.append(0)
        arm_speed_data.append(0)
        arm_time_steps.append(step)
        finger_pos = arm_angle_to_finger_dist(0)
        finger_min_pos_list.append(finger_pos)
        
        if step == 0:
            finger_encoder_max_list.append(finger_dist2enc(FINGER_CLAMP))
            finger_encoder_min_list.append(finger_dist2enc(FINGER_CLAMP))
        elif step < FINGER_CLAMP_TIME_S1:
            finger_encoder_max_list.append(finger_dist2enc(CUBE_SIZE))
            finger_encoder_min_list.append(finger_dist2enc(FINGER_CLAMP))
        else:
            finger_encoder_max_list.append(finger_dist2enc(FINGER_LIMIT))
            finger_encoder_min_list.append(finger_dist2enc(CUBE_SIZE))
        step += 1
    # 第二阶段：旋转臂和手指联动
    while True:
        state = sc_arm.run()
        if state == ScurveState.CURVE_BUSY or state == ScurveState.CURVE_ONEND:
            pos, vel, acc, jrk = sc_arm.getOutputs()
            arm_encoder_data.append(pos)
            arm_speed_data.append(vel/RPM_TO_SPEED)
            arm_time_steps.append(step)
            
            angle = pos * 90 / 8192
            finger_pos = arm_angle_to_finger_dist(angle)
            finger_min_pos_list.append(finger_pos)  # distance (mm)
            finger_encoder_min_list.append(finger_dist2enc(finger_pos) + pos / 2)
            finger_encoder_max_list.append(finger_dist2enc(FINGER_LIMIT) + pos / 2)
            step += 1
        else:
            break
    # 第三阶段：只有手指移动，旋转臂固定
    finger_speed_limit_time_s3 = step + FINGER_ONLY_SEQUANCE_TIME_S3
    for i in range(FINGER_ONLY_SEQUANCE_TIME_S3 + FINGER_CLAMP_TIME_S3):
        pos = 8192
        arm_encoder_data.append(pos)
        arm_speed_data.append(0)
        arm_time_steps.append(step)
        
        finger_pos = arm_angle_to_finger_dist(0)
        finger_min_pos_list.append(finger_pos)  # distance (mm)

        if i == FINGER_ONLY_SEQUANCE_TIME_S3 + FINGER_CLAMP_TIME_S3 - 1:
            finger_encoder_max_list.append(finger_dist2enc(FINGER_CLAMP) + pos / 2)
            finger_encoder_min_list.append(finger_dist2enc(FINGER_CLAMP) + pos / 2)
        elif i > FINGER_ONLY_SEQUANCE_TIME_S3:
            finger_encoder_max_list.append(finger_dist2enc(CUBE_SIZE) + pos / 2)
            finger_encoder_min_list.append(finger_dist2enc(FINGER_CLAMP) + pos / 2)
        else:
            finger_encoder_max_list.append(finger_dist2enc(FINGER_LIMIT) + pos / 2)
            finger_encoder_min_list.append(finger_dist2enc(CUBE_SIZE) + pos / 2)
        step += 1

    # print(f"总耗时: {step*0.2:.1f}ms")
    # print("finger_encoder_min_list")
    # print(np.array(finger_encoder_min_list).astype(int))
    # print("finger_encoder_max_list")
    # print(np.array(finger_encoder_max_list).astype(int))
    finger_encoder_final = generate_finger_trajectory(
        finger_encoder_min_list,
        finger_encoder_max_list,
        finger_speed_limit_time_s1,
        finger_speed_limit_time_s3
    )
    if finger_encoder_final is None:
        print("求解手指轨迹出错")
        return None, None

    if (need_plot):
        # array = np.array(finger_encoder_final)
        array_v = np.diff(finger_encoder_final, n=1) / RPM_TO_SPEED
        # array_a = np.diff(finger_encoder_final, n=2)
        # print(array)
        # print(array_a)


        # 创建图形和网格布局
        plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # 1. 左边：极坐标图（第一象限）
        ax_polar = plt.subplot(gs[:, 0], projection='polar')  # 占据整个左侧

        # 绘制正方形（边长为56，顶点在45°位置）
        radius = (CUBE_SIZE * math.sqrt(2)) / 2
        angles = [math.radians(a) for a in [45, 135, 225, 315, 45]]
        radii = [radius] * 5
        ax_polar.plot(angles, radii, 'y-', linewidth=2, label=f'Cube (size={CUBE_SIZE}mm)')

        # 绘制arm_angle_to_finger_dist曲线
        angles_deg = np.linspace(0, 90, 90) 
        radii_values = [arm_angle_to_finger_dist(angle)/2 for angle in angles_deg]
        angles_rad = np.deg2rad(angles_deg)
        ax_polar.plot(angles_rad, radii_values, 'r-', linewidth=2, label='Min')

        # 反推手指实际运动轨迹并绘制
        # 根据公式：finger_encoder_final = finger_dist2enc(finger_pos) + 手臂编码位置 / 2
        # 所以：finger_pos = finger_enc2dist(finger_encoder_final - arm_position_data / 2)
        finger_pos_list = []
        finger_radii = []  # 极坐标中的半径（手指到中心的距离）
        finger_angles = [] # 极坐标中的角度（手臂角度）

        for i in range(len(finger_encoder_final)):
            # 计算实际手指位置（两指间距）
            finger_enc = finger_encoder_final[i] - arm_encoder_data[i] / 2
            finger_dist = finger_enc2dist(finger_enc)
            finger_pos_list.append(finger_dist)
            
            # 计算手臂角度（转换为弧度）
            angle = arm_encoder_data[i] * 90 / ARM_STEPS
            finger_angles.append(math.radians(angle))
            
            # 极坐标中的半径是手指到中心的距离（即手指间距的一半）
            finger_radii.append(finger_dist / 2)

        # 绘制手指运动轨迹
        ax_polar.plot(finger_angles, finger_radii, 'g-', linewidth=2, label='Final')
        ax_polar.scatter(finger_angles[0], finger_radii[0], color='green', s=50, zorder=5, label='Start')
        ax_polar.scatter(finger_angles[-1], finger_radii[-1], color='purple', s=50, zorder=5, label='End')

        # 设置极坐标图只显示第一象限
        ax_polar.set_theta_offset(0)  # 0度在正右方
        ax_polar.set_theta_direction(-1)  # 角度顺时针方向增加
        ax_polar.set_thetamin(0)   # 最小角度0度
        ax_polar.set_thetamax(90)  # 最大角度90度
        ax_polar.grid(True)
        ax_polar.legend(loc='best')
        ax_polar.set_title('Polar Diagram of Cube and Finger Distance Function', pad=20)

        # 2. 右上：手臂位置时间序列
        ax_speed = plt.subplot(gs[0, 1])
        # ax_arm.plot(arm_time_steps, arm_position_data, 'b-', linewidth=2)
        ax_speed.plot(arm_time_steps[1:], array_v, 'b-', linewidth=2, label='Finger')
        ax_speed.plot(arm_time_steps, arm_speed_data, 'r-', linewidth=2, label='Arm')
        ax_speed.legend(loc='best')
        ax_speed.set_title('Speed vs Time', fontsize=12)
        ax_speed.set_xlabel('Time (0.2ms each step)', fontsize=10)
        ax_speed.set_ylabel('Speed(RPM)', fontsize=10)
        ax_speed.grid(True, linestyle='--', alpha=0.7)

        # 3. 右下：手指位置时间序列
        ax_finger = plt.subplot(gs[1, 1])
        ax_finger.plot(arm_time_steps, finger_encoder_min_list, 'r-', linewidth=2, label='Min')
        ax_finger.plot(arm_time_steps, finger_encoder_max_list, 'b-', linewidth=2, label='Max')
        ax_finger.plot(arm_time_steps, finger_encoder_final, 'g-', linewidth=2, label='Final')
        ax_finger.legend(loc='best')
        ax_finger.set_title('Finger Position vs Time', fontsize=12)
        ax_finger.set_xlabel('Time (0.2ms each step)', fontsize=10)
        ax_finger.set_ylabel('Position', fontsize=10)
        ax_finger.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout(pad=3.0)
        plt.show()

    # 将finger_encoder_final改为以0作为起始
    finger_encoder_start_with_zero = [0] * step
    for i in range(step):
        finger_encoder_start_with_zero[i] = finger_encoder_final[i] - finger_encoder_final[0]

    # 四舍五入
    for i in range(step):
        arm_encoder_data[i] = round(arm_encoder_data[i])
        finger_encoder_start_with_zero[i] = round(finger_encoder_start_with_zero[i])
        
    # 确认数据的起止点
    assert(arm_encoder_data[0] == 0)
    assert(arm_encoder_data[-1] == 8192)
    assert(finger_encoder_start_with_zero[0] == 0)
    assert(finger_encoder_start_with_zero[-1] == 4096)
    return arm_encoder_data, finger_encoder_start_with_zero

if __name__ == "__main__":
    calc_motion_adjust_arm(True)