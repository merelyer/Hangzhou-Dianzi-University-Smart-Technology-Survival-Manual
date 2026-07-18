# 手臂空转手指联动逻辑
# 手指离开魔方一点点，例如2mm
# 后续始终保持这个距离，进行90°空转
#  0°：60mm
# 30°：81mm
# 60°：81mm
# 90°：60mm

import numpy as np
import matplotlib.pyplot as plt
import math

# 设置为True时，仿真空转后手指不锁紧的场景
# NO_FINGER_RETURN_MODE = False
NO_FINGER_RETURN_MODE = True

V_NO_LOAD = 600  # 旋转臂空转最大速度
A_NO_LOAD = 300  # 旋转臂空转最大加速度，在停转不抖动的前提下，尽可能大
V_FINGER = 1500  # 手指移动速度，这个惯性小，也不会变形，可用很快
A_FINGER = 1600
FINGER_ZERO   = 52.0                  # 零点位于距离内侧限位点0.5mm的位置，此时两手指间距52mm
def finger_dist2enc(x): 
    FINGER_FACTOR = 16384.0 / (2 * 31.416)# 手指电机每转一圈，手指移动距离31.4mm，两手指间距增加2 * 31.4mm
    return int((x - FINGER_ZERO) * FINGER_FACTOR)
FINGER_CLAMP  = finger_dist2enc(54.0) # 手指锁紧，这是闭环控制的，可填写比实际需求小一点的数字
FINGER_INIT   = finger_dist2enc(56.2) # 手指初始位置，用于放置魔方，52+2*2 = 56mm，与魔方边长保持一致(有的魔方偏大，可以改大点)
FINGER_FLIP   = finger_dist2enc(78.0) # 翻转魔方时，手指移动的最远位置，最小74mm，留4mm余量
FINGER_MAX    = finger_dist2enc(81.0) # 56*√2 = 79.2mm， 魔方是圆角的，实测值是77mm，留4mm余量，取81mm，机械结构最大支持到84mm

rpm_to_speed = 16384.0 / 5000.0 / 60.0
FINGER_FACTOR = 16384.0 / (2 * 31.416)

# 情景1，能加速到最高速度vmax：匀加速(Ta)----匀速(Tv)----匀减速(Td)
# 情景2，不能加速到最高速度：匀加速(Ta)----匀减速(Td)
# x0   初始位置
# x1   目标位置
# v0   初始速度
# v1   目标速度
# vmax 允许的最大速度
# a    加速度的绝对值
class Trapezoid:
    # 改写为C语言实现时，异常处理用打印报错并返回错误代码的方式替代，全部浮点运算使用float
    def calculate_motion(self, x0, x1, v0_input, v1_input, vmax_input, a_input):
        v0 = v0_input * rpm_to_speed
        v1 = v1_input * rpm_to_speed
        vmax = vmax_input * rpm_to_speed
        a = a_input / 1024.0

        self.x0 = x0
        self.x1 = x1
        self.v0 = v0
        self.v1 = v1
        vmax = abs(vmax)
        a = abs(a)
        
        # 校验速度是否超过允许范围（考虑方向）
        if vmax < 1e-6:
            raise ValueError("过小的最大速度")
        if a < 1e-6:
            raise ValueError("过小的加速度")
        if abs(v0) > vmax or abs(v1) > vmax:
            raise ValueError("初始速度或目标速度超过最大允许速度")

        # 运动方向判断
        S = x1 - x0  # 总位移
        direction = 1 if S >= 0 else -1  # 1:正方向 -1:负方向
        
        # 运动参数初始化
        self.a_accel = a * direction  # 加速阶段加速度
        self.a_decel = -a * direction # 减速阶段加速度
        signed_vmax = vmax * direction  # 最大速度

        # 计算理论加速/减速时间
        Ta_max = (signed_vmax - v0) / self.a_accel
        Td_max = (v1 - signed_vmax) / self.a_decel
        
        # 计算理论位移
        Sa_max = (signed_vmax**2 - v0**2) / (2 * self.a_accel)
        Sd_max = (v1**2 - signed_vmax**2) / (2 * self.a_decel)
        S_needed = Sa_max + Sd_max

        # 判断实际位移是否足够
        Tv = (S - S_needed) / signed_vmax
        if Tv >= 0:
            # 情景1：能加速到最大速度
            self.Ta = Ta_max
            self.Td = Td_max
            self.Tv = Tv
            self.v_c = signed_vmax
            self.T = self.Ta + self.Tv + self.Td
        else:
            # 情景2：无法达到最大速度 
            # 计算中间速度v_c（考虑方向）
            v_c_sq = (2 * a * abs(S) + v0**2 + v1**2) / 2
            v_c_abs = math.sqrt(v_c_sq)
            v_c = v_c_abs * direction 
            # 重新计算加速/减速时间
            self.Ta = (v_c - v0) / self.a_accel
            self.Td = (v1 - v_c) / self.a_decel
            self.Tv = 0
            self.v_c = v_c
            self.T = self.Ta + self.Td

        if self.Ta < 0 or self.Td < 0:
            raise ValueError("计算得到负时间，参数错误")
        
        self.x_acc_end = self.x0 + self.v0 * self.Ta + 0.5 * self.a_accel * self.Ta**2
        self.x_vel_end = self.x_acc_end + self.v_c * self.Tv

    # 改写为C语言实现时，只需要返回x，不需要a和v，需要仔细优化，缩短运行时间
    def get_pos(self, t):
        if t < 0:
            a = 0
            v = self.v0
            x = self.x0
        elif t > self.T:
            a = 0
            v = self.v1
            x = self.x1
        elif t <= self.Ta:
            dt = t
            a = self.a_accel
            v = self.v0 + a * dt
            x = self.x0 + self.v0 * dt + 0.5 * a * dt**2
        elif t <= self.Ta + self.Tv:
            dt = t - self.Ta
            a = 0
            v = self.v_c
            x = self.x_acc_end + self.v_c * dt
        else:
            dt = t - self.Ta - self.Tv
            a = self.a_decel
            v = self.v_c + a * dt
            x = self.x_vel_end + self.v_c * dt + 0.5 * a * dt**2

        return (x, v, a)
    

# 控制步骤
# 阶段1：手指加速，当移动到指定位置（手臂启动延时）时，手臂开始加速，直到手臂转到30度
# 阶段2：等手臂旋转到30度时，手指变为跟随手臂的速度
# 阶段3：等手臂旋转到60度时，手臂缩回

# 分别计算手臂运动到30°和60°时的速度和时间（假设这两个事件都在加减速节点发生，简化计算，实际过程也是这样的）
trap_arm = Trapezoid()
arm_90_deg = 8192
arm_20_deg = 8192 *(20/90)
arm_70_deg = 8192 *(70/90)
trap_arm.calculate_motion(0, arm_90_deg, 0, 0, 2*V_NO_LOAD, 2*A_NO_LOAD)# 带有2:1减速，所以需要两倍速度
arm_totel_time = trap_arm.T
arm_20_deg_time = math.sqrt(2 * arm_20_deg / trap_arm.a_accel)
arm_20_deg_speed = trap_arm.a_accel * arm_20_deg_time / rpm_to_speed
print(f"手臂运动到20°时的时间={arm_20_deg_time:.2f}, 手臂运动到20°时的速度={arm_20_deg_speed:.2f}")
# 利用加减速的对称性简化计算
arm_70_deg_time = trap_arm.T - arm_20_deg_time
arm_70_deg_speed = arm_20_deg_speed
print(f"手臂运动到70°时的时间={arm_70_deg_time:.2f}, 手臂运动到70°时的速度={arm_20_deg_speed:.2f}")

# 阶段1
finger_add_offset = FINGER_MAX - FINGER_INIT
trap_finger_stage_1 = Trapezoid()
trap_finger_stage_1.calculate_motion(0, 
                                     arm_20_deg/2 + finger_add_offset, 
                                     0, 
                                     arm_20_deg_speed/2, 
                                     V_FINGER, 
                                     A_FINGER)
time_stage_1 = trap_finger_stage_1.T
print(f"阶段1耗时={time_stage_1:.2f}")

# 计算手臂启动延时
stage_1_arm_delay_time = time_stage_1 - arm_20_deg_time
print(f"阶段1手臂启动延时={stage_1_arm_delay_time:.2f}")
pos_stage_1_arm_start, _, _ = trap_finger_stage_1.get_pos(stage_1_arm_delay_time)
print(f"阶段1手臂启动时手指的位置={pos_stage_1_arm_start/FINGER_FACTOR:.2f}")

if NO_FINGER_RETURN_MODE == False:
    # 阶段2
    trap_finger_stage_2 = Trapezoid()
    trap_finger_stage_2.calculate_motion(trap_finger_stage_1.x1, 
                                        arm_70_deg/2 + finger_add_offset, 
                                        arm_20_deg_speed/2, 
                                        arm_70_deg_speed/2, 
                                        V_NO_LOAD, 
                                        A_NO_LOAD)
    time_stage_2 = trap_finger_stage_2.T
    # 阶段3
    trap_finger_stage_3 = Trapezoid()
    trap_finger_stage_3.calculate_motion(trap_finger_stage_2.x1, 
                                        arm_90_deg/2, 
                                        arm_70_deg_speed/2, 
                                        0, 
                                        V_FINGER, 
                                        A_FINGER)
    time_stage_3 = trap_finger_stage_3.T
else:
    # 阶段2
    trap_finger_stage_2 = Trapezoid()
    trap_finger_stage_2.calculate_motion(trap_finger_stage_1.x1, 
                                        arm_90_deg/2 + finger_add_offset, 
                                        arm_20_deg_speed/2, 
                                        0, 
                                        V_NO_LOAD, 
                                        A_NO_LOAD)
    time_stage_2 = trap_finger_stage_2.T
    time_stage_3 = 0

plot_p_arm = []
plot_v_arm = []
plot_p_finger = []
plot_v_finger = []
if NO_FINGER_RETURN_MODE == False:
    totel_time = time_stage_1 + time_stage_2 + time_stage_3
    # print(time_stage_1,time_stage_2,time_stage_3)
    for t in np.arange(0, totel_time, 1):
        x_arm, v_arm, a = trap_arm.get_pos(t - stage_1_arm_delay_time)
        angle_arm = x_arm * 90 / 8192
        plot_p_arm.append(angle_arm)
        plot_v_arm.append(v_arm)
        
        if t < time_stage_1:
            x_finger, v_finger, a2 = trap_finger_stage_1.get_pos(t)
            plot_p_finger.append((x_finger - x_arm/2)/FINGER_FACTOR)
            plot_v_finger.append(v_finger)
        elif t < time_stage_1 + time_stage_2:
            x_finger, v_finger, a2 = trap_finger_stage_2.get_pos(t-time_stage_1)
            plot_p_finger.append((x_finger - x_arm/2)/FINGER_FACTOR)
            plot_v_finger.append(v_finger)
        else:
            x_finger, v_finger, a2 = trap_finger_stage_3.get_pos(t-time_stage_1-time_stage_2)
            plot_p_finger.append((x_finger - x_arm/2)/FINGER_FACTOR)
            plot_v_finger.append(v_finger)
else:
    totel_time = time_stage_1 + time_stage_2
    for t in np.arange(0, totel_time, 1):
        x_arm, v_arm, a = trap_arm.get_pos(t - stage_1_arm_delay_time)
        angle_arm = x_arm * 90 / 8192
        plot_p_arm.append(angle_arm)
        plot_v_arm.append(v_arm)
        
        if t < time_stage_1:
            x_finger, v_finger, a2 = trap_finger_stage_1.get_pos(t)
            plot_p_finger.append((x_finger - x_arm/2)/FINGER_FACTOR)
            plot_v_finger.append(v_finger)
        else:
            x_finger, v_finger, a2 = trap_finger_stage_2.get_pos(t-time_stage_1)
            plot_p_finger.append((x_finger - x_arm/2)/FINGER_FACTOR)
            plot_v_finger.append(v_finger)
t = np.arange(0, totel_time, 1)

plt.rcParams['font.sans-serif'] = ['STHeiti']
# 创建2个垂直排列的子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('5 Channel Data Plot')

# 绘制第一个子图,位置
ax1.plot(t, plot_p_arm, label="arm angle(deg)")
ax1.plot(t, plot_p_finger, label="finger pos(mm)")
# 设置y轴刻度和网格线
ax1.set_xticks([0, time_stage_1, time_stage_1 + time_stage_2, time_stage_1 + time_stage_2 + time_stage_3]) 
ax1.set_yticks([0, 30, 60, 90])  # 指定刻度位置
ax1.grid(True)
ax1.legend()

# 绘制第二个子图，速度
ax2.plot(t, plot_v_arm, label="arm speed(rpm)")
ax2.plot(t, plot_v_finger, label="finger speed(rpm)")
ax2.set_xticks([0, stage_1_arm_delay_time, 
                time_stage_1, time_stage_1 + time_stage_2, time_stage_1 + time_stage_2 + time_stage_3]) 
ax2.grid(True)
ax2.legend()

print(f"V_NO_LOAD_20_70_DEG = {round(arm_20_deg_speed)}")
print(f"FINGER_NO_LOAD_START_ARM = {(pos_stage_1_arm_start+FINGER_INIT)/FINGER_FACTOR + FINGER_ZERO:.2f}")
plt.show()