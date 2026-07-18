import matplotlib.pyplot as plt
from scurve import Scurve, ScurveState, ScurveReturnStatus

sc = Scurve(1)
# 转速单位转换 RPM -> 编码器计数/控制周期
RPM_TO_SPEED = 16384.0 / 5000.0 / 60.0
VMAX = 1200 * RPM_TO_SPEED
ACCEL = 0.6
JERK = 0.1

sc.setPositionOutput(0)
sc.setPositionTarget(8192) # 1/2圈
sc.setVelocityStart(0.0)
sc.setVelocityStop(0.0)
sc.setVelocityMax(VMAX)
sc.setAccelerationMax(ACCEL)
sc.setDecelerationMax(ACCEL)
sc.setJerkMax(JERK)

# Start profile generation
status = sc.startProfile()
if status != ScurveReturnStatus.CURVE_SUCCESS:
    print(f"Failed to start profile: {status}")
    exit(1)

# Data recording
time_data = []
position_data = []
velocity_data = []
acceleration_data = []
jerk_data = []

# Execute profile generation
print("Generating motion profile...")
while True:
    state = sc.run()
    if state == ScurveState.CURVE_BUSY or state == ScurveState.CURVE_ONEND:
        # Get current outputs
        pos, vel, acc, jrk = sc.getOutputs()
        # Record data
        time_data.append(sc.getProfileTick()-1)
        position_data.append(pos)
        velocity_data.append(vel)
        acceleration_data.append(acc)
        jerk_data.append(jrk)
    else:
        break

print(f"Profile complete! Total time: {time_data[-1]:.3f}s, Samples: {len(time_data)}")

# Create plots
plt.figure(figsize=(12, 8))

# Position curve
plt.subplot(4, 1, 1)
plt.plot(time_data, position_data, 'b-', linewidth=2)
plt.title('S-Curve Motion Profile')
plt.ylabel('Position (units)')
plt.grid(True)

# Velocity curve
plt.subplot(4, 1, 2)
plt.plot(time_data, velocity_data, 'g-', linewidth=2)
plt.ylabel('Velocity (units/s)')
plt.grid(True)

# Acceleration curve
plt.subplot(4, 1, 3)
plt.plot(time_data, acceleration_data, 'r-', linewidth=2)
plt.ylabel('Acceleration (units/s²)')
plt.grid(True)

# Jerk curve
plt.subplot(4, 1, 4)
plt.plot(time_data, jerk_data, 'm-', linewidth=2)
plt.ylabel('Jerk (units/s³)')
plt.xlabel('Time (s)')
plt.grid(True)

# Save and show plot
plt.tight_layout()
plt.savefig('scurve_profile_en.png', dpi=300)
plt.show()