import time
import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scurve import Scurve, ScurveState, ScurveReturnStatus
from finger_vs_arm_angle import calculate_min_offset
# 翻转魔方的同时，对侧手臂旋转90度的控制代码
# 旋转臂控制使用S加减速、手指联动使用凸优化方案
# 凸优化运行较慢，但是使用cvxpy库时，代码比较简洁，适合提前计算而不是实时计算

# 依赖的第三方包
# pip3 install numpy cvxpy matplotlib ecos

    
# 使用motion_flip_and_adjust.py计算min_offsets
# 预计算最小 arm_offset 值
print("开始预计算最小 arm_offset...")
start_time = time.time()
import time
angles = np.arange(0, 91, 1)
min_offsets = []

for angle in angles:
    min_offset = calculate_min_offset(angle, 90 - angle)
    min_offsets.append(min_offset)

end_time = time.time()
print(f"预计算完成! 耗时: {end_time - start_time:.2f}秒")
print(min_offsets)

# --------------- 旋转臂参数 ---------------
# 转速单位转换 RPM -> 编码器计数/控制周期
RPM_TO_SPEED = 16384.0 / 5000.0 / 60.0
# 旋转臂最大速度，电机本身的，减速之前的(上一版本是360RPM)
ARM_VMAX = 360 * RPM_TO_SPEED
# 旋转臂最大加速度，电机本身的，减速之前的(上一版本是140/1024=0.137)
ARM_ACCEL = 0.14
# 旋转臂最大加加速度
ARM_JERK  = 0.01

ARM_ACCEL_NO_LOAD = 0.6

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
# 两手指之间的距离的上限
FINGER_LIMIT = 85.5
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
FINGER_ONLY_SEQUANCE_TIME_S1 = 58
FINGER_ONLY_SEQUANCE_TIME_S3 = 58
FINGER_CLAMP_TIME_S3         = 58


def finger_dist2enc(x): 
    return (x - FINGER_ZERO) * FINGER_FACTOR

def finger_enc2dist(x): 
    return x / FINGER_FACTOR + FINGER_ZERO

# 输入：angle，旋转臂角度
# 输出：两手指之间的最小间距，以及是否达到最大值
def arm_angle_to_finger_dist(angle):
    assert(angle >= 0 and angle <= 90)
    FINGER_BASE_THICKNESS = 7   # 手指小凸台尺寸
    FINGER_TOP_THICKNESS  = 4   # 手指指尖厚度
    GAP                   = 2   # 手指和魔方之间的安全距离，单位mm
    # 开始翻面时的手指位置
    FINGER_CUBE_FLIP = CUBE_SIZE + 2 * (FINGER_BASE_THICKNESS - FINGER_TOP_THICKNESS)
    arm_offset = min_offsets[round(angle)] + GAP
    if arm_offset > FINGER_LIMIT:
        arm_offset = FINGER_LIMIT
    if arm_offset < FINGER_CUBE_FLIP:
        arm_offset = FINGER_CUBE_FLIP
    return arm_offset

# 输入手臂角度（取值范围0-180），输出两手指之间的最小距离
# 这个韩式只适用于对侧手臂水平的场景
def arm_angle_to_finger_dist_simple(angle, target_angle = 90):
    FINGER_BASE_THICKNESS = 7   # 手指小凸台尺寸
    FINGER_TOP_THICKNESS  = 4   # 手指指尖厚度
    FINGER_TOP_HEIGHT     = 13  # 手指指尖高度
    GAP                   = 2   # 手指和魔方之间的安全距离，单位mm
    # 开始翻面时的手指位置
    FINGER_CUBE_FLIP = CUBE_SIZE + 2 * (FINGER_BASE_THICKNESS - FINGER_TOP_THICKNESS)
    angle = math.radians(angle)
    target_angle = math.radians(target_angle)
    critical_angle = math.pi / 4 - math.asin((CUBE_SIZE / 2 - FINGER_TOP_HEIGHT) / (CUBE_SIZE / math.sqrt(2)))
    # print(math.degrees(critical_angle))
    if angle > target_angle - critical_angle:
        angle = target_angle - angle
    elif angle < critical_angle:
        pass
    else:
        angle = critical_angle
    dist = 2 * ( CUBE_SIZE / math.sqrt(2) * math.cos(math.pi / 4 - angle) + GAP ) 
    if dist < FINGER_CUBE_FLIP:
        dist = FINGER_CUBE_FLIP
    return dist
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
    # 旋转臂旋转90度，对应电机编码器的变化量（1/2圈）
    ARM_STEPS = 8192 
    # 翻面-180°
    sc_arm_flip = Scurve(1)
    sc_arm_flip.setPositionOutput(0)
    sc_arm_flip.setPositionTarget(-ARM_STEPS * 2) 
    sc_arm_flip.setVelocityStart(0.0)
    sc_arm_flip.setVelocityStop(0.0)
    sc_arm_flip.setVelocityMax(ARM_VMAX)
    sc_arm_flip.setAccelerationMax(ARM_ACCEL)
    sc_arm_flip.setDecelerationMax(ARM_ACCEL)
    sc_arm_flip.setJerkMax(ARM_JERK)
    sc_arm_flip.startProfile()
    # 对侧手指调整90°
    sc_arm_no_load = Scurve(1)
    sc_arm_no_load.setPositionOutput(0)
    sc_arm_no_load.setPositionTarget(ARM_STEPS) 
    sc_arm_no_load.setVelocityStart(0.0)
    sc_arm_no_load.setVelocityStop(0.0)
    sc_arm_no_load.setVelocityMax(ARM_VMAX)
    sc_arm_no_load.setAccelerationMax(ARM_ACCEL)
    sc_arm_no_load.setDecelerationMax(ARM_ACCEL_NO_LOAD)
    sc_arm_no_load.setJerkMax(ARM_JERK)
    sc_arm_no_load.startProfile()

    arm_encoder_data_flip = []
    arm_encoder_data_no_load = []
    arm_speed_data = []
    arm2_speed_data = []
    arm_time_steps = []
    finger_min_pos_list = []
    finger_encoder_min_list = []
    finger_encoder_max_list = []
    step = 0
    # 第一阶段：只有手指移动，旋转臂固定
    finger_speed_limit_time_s1 = FINGER_CLAMP_TIME_S1
    for i in range(FINGER_ONLY_SEQUANCE_TIME_S1 + FINGER_CLAMP_TIME_S1):
        arm_encoder_data_flip.append(0)
        arm_encoder_data_no_load.append(0)
        arm_speed_data.append(0)
        arm2_speed_data.append(0)
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
        state_arm_flip = sc_arm_flip.run()
        state_no_load = sc_arm_no_load.run()
        if state_arm_flip == ScurveState.CURVE_BUSY or state_arm_flip == ScurveState.CURVE_ONEND:
            if state_no_load == ScurveState.CURVE_BUSY or state_no_load == ScurveState.CURVE_ONEND:
                pos, vel, _, _ = sc_arm_no_load.getOutputs()
            else:
                pos = ARM_STEPS
                vel = 0
            arm_encoder_data_no_load.append(pos)
            arm_speed_data.append(vel/RPM_TO_SPEED)
            arm_time_steps.append(step)
            angle_no_load = abs(pos * 90 / ARM_STEPS)
            pos_flip, vel_flip, _, _ = sc_arm_flip.getOutputs()
            arm2_speed_data.append(vel_flip/RPM_TO_SPEED)
            arm_encoder_data_flip.append(pos_flip)
            angle_flip = abs(pos_flip * 90 / ARM_STEPS)
            if angle_flip <= 90:
                finger_pos = arm_angle_to_finger_dist(angle_flip)
            else:
                finger_pos = arm_angle_to_finger_dist_simple(angle_flip, 180)
            finger_min_pos_list.append(finger_pos)  # distance (mm)
            finger_encoder_min_list.append(finger_dist2enc(finger_pos) + pos / 2)
            finger_encoder_max_list.append(finger_dist2enc(FINGER_LIMIT) + pos / 2)
            step += 1
        else:
            break
    # 第三阶段：只有手指移动，旋转臂固定
    finger_speed_limit_time_s3 = step + FINGER_ONLY_SEQUANCE_TIME_S3
    for i in range(FINGER_ONLY_SEQUANCE_TIME_S3 + FINGER_CLAMP_TIME_S3):
        pos = ARM_STEPS
        arm_encoder_data_no_load.append(pos)
        arm_encoder_data_flip.append( - ARM_STEPS * 2)
        arm_speed_data.append(0)
        arm2_speed_data.append(0)
        arm_time_steps.append(step)
        
        finger_pos = arm_angle_to_finger_dist_simple(0)
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

    finger_encoder_final = generate_finger_trajectory(
        finger_encoder_min_list,
        finger_encoder_max_list,
        finger_speed_limit_time_s1,
        finger_speed_limit_time_s3
    )
    if finger_encoder_final is None:
        print("求解手指轨迹出错")
        return None

    # 计算手指速度
    finger_speed_data = []
    for i in range(len(finger_encoder_final)):
        if i == 0:
            # 第一个点速度为0
            finger_speed_data.append(0.0)
        else:
            # 速度 = (当前位置 - 前一个位置) / RPM_TO_SPEED
            speed = (finger_encoder_final[i] - finger_encoder_final[i-1]) / RPM_TO_SPEED
            finger_speed_data.append(speed)

    # 将finger_encoder_final改为以0作为起始
    finger_encoder_start_with_zero = [0] * step
    for i in range(step):
        finger_encoder_start_with_zero[i] = finger_encoder_final[i] - finger_encoder_final[0]

    # 四舍五入
    for i in range(step):
        arm_encoder_data_no_load[i] = round(arm_encoder_data_no_load[i])
        arm_encoder_data_flip[i] = round(arm_encoder_data_flip[i])
        finger_encoder_start_with_zero[i] = round(finger_encoder_start_with_zero[i])

    # 确认数据的起止点
    assert(arm_encoder_data_no_load[0] == 0)
    assert(arm_encoder_data_no_load[-1] == ARM_STEPS)
    assert(arm_encoder_data_flip[0] == 0)
    assert(arm_encoder_data_flip[-1] == - 2 * ARM_STEPS)
    assert(finger_encoder_start_with_zero[0] == 0)
    assert(finger_encoder_start_with_zero[-1] == ARM_STEPS / 2)
    
    # 返回绘图数据
    plot_data = {
        'arm_time_steps': arm_time_steps,
        'arm_speed_data': arm_speed_data,
        'arm2_speed_data': arm2_speed_data,
        'finger_speed_data': finger_speed_data,
        'finger_encoder_min_list': finger_encoder_min_list,
        'finger_encoder_max_list': finger_encoder_max_list,
        'finger_encoder_final': finger_encoder_final,
    }
    if need_plot:
        return arm_encoder_data_flip, arm_encoder_data_no_load, finger_encoder_start_with_zero, plot_data
    else:
        return arm_encoder_data_flip, arm_encoder_data_no_load, finger_encoder_start_with_zero

if __name__ == "__main__":
    # 创建2x2的图形布局
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 计算90度和180度的运动数据
    arm_encoder_data_flip, arm_encoder_data_no_load, _, plot_data_180 = calc_motion_adjust_arm(True)
    
    # 绘制XXXXX
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(plot_data_180['arm_time_steps'], np.array(arm_encoder_data_flip) / (-8192.0 / 90), 'r-', label='flip arm angle')
    ax1.plot(plot_data_180['arm_time_steps'], np.array(arm_encoder_data_no_load) / (8192.0 / 90), 'b-', label='no load arm angle')
    ax1.set_title(f"Angle vs Time (+90°)")
    ax1.set_xlabel('Time (0.2ms each step)')
    ax1.set_ylabel('Angle (deg)')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制180度的速度曲线
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(plot_data_180['arm_time_steps'], plot_data_180['arm_speed_data'], 'r-', label='Arm(90 deg) Speed')
    ax3.plot(plot_data_180['arm_time_steps'], plot_data_180['arm2_speed_data'], 'y-', label='Arm(180 deg) Speed')
    ax3.plot(plot_data_180['arm_time_steps'], plot_data_180['finger_speed_data'], 'b-', label='Finger Speed')
    ax3.set_title(f"Speed vs Time (-90°)")
    ax3.set_xlabel('Time (0.2ms each step)')
    ax3.set_ylabel('Speed (RPM)')
    ax3.legend()
    ax3.grid(True)
    
    # 绘制180度的手指位置
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(plot_data_180['arm_time_steps'], plot_data_180['finger_encoder_min_list'], 'r-', label='Min Position')
    ax4.plot(plot_data_180['arm_time_steps'], plot_data_180['finger_encoder_max_list'], 'b-', label='Max Position')
    ax4.plot(plot_data_180['arm_time_steps'], plot_data_180['finger_encoder_final'], 'g-', label='Final Trajectory')
    ax4.set_title(f"Finger Position (-90°)")
    ax4.set_xlabel('Time (0.2ms each step)')
    ax4.set_ylabel('Encoder Position')
    ax4.legend()
    ax4.grid(True)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()