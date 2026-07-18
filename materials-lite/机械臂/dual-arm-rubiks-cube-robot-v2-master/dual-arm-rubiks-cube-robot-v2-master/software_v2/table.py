#!/usr/bin/python3

def write_table(f, x, label):
    # 固定时间字符串（根据题目要求）
    n = len(x)
    
    # 写入时间注释和表头
    if label is not None:
        f.write(f"{label}[{n}] =\n")
    
    # 处理并写入数据
    for i in range(n):
        # 检查数值范围
        if not (-32768 <= x[i] <= 32767):
            raise ValueError(f"数值 {x[i]} 超出范围 [-32768, 32767]")
        
        # 写入数值（无空格）
        f.write(str(x[i]))
        
        # 添加逗号或换行
        if i == n - 1:  # 最后一个元素
            f.write("\n")
        elif (i + 1) % 16 == 0:  # 每16个换行
            f.write(",\n")
        else:
            f.write(",")

# 读取输入文件
with open("motion.cfg.in", "r") as f_in:
    lines = f_in.readlines()

# 找到自动生成标记的位置
autogen_index = -1
for i, line in enumerate(lines):
    if "# autogen use table.py" in line:
        autogen_index = i
        break

if autogen_index == -1:
    raise ValueError("未找到 '# autogen use table.py' 标记")

# 打开输出文件
with open("motion.cfg", "w") as f_out:
    # 写入标记之前的内容
    for line in lines[:autogen_index]:
        f_out.write(line)
    
    # 写入自动生成标记
    f_out.write("# autogen use table.py\n")
    
    # 生成数据并写入表格
    print("------------ 生成TABLE_NO_LOAD ------------")
    from motion_adjust_arm import calc_motion_adjust_arm as calc_no_load
    arm_encoder, finger_encoder = calc_no_load(False)
    f_out.write("# 手指松开避让魔方，单侧旋转臂空转90°，然后夹紧魔方\n")
    write_table(f_out, finger_encoder, "TABLE_NO_LOAD")
    write_table(f_out, arm_encoder, None)
    f_out.write("\n")
    
    print("------------ 生成TABLE_FLIP_90 ------------")
    from motion_flip import calc_motion_adjust_arm as calc_flip
    arm_encoder, finger_encoder = calc_flip(False, 90)
    f_out.write("# 魔方翻面，对侧手指避让\n")
    write_table(f_out, finger_encoder, "TABLE_FLIP_90")
    write_table(f_out, arm_encoder, None)
    print("------------ 生成TABLE_FLIP_180 ------------")
    arm_encoder, finger_encoder = calc_flip(False, 180)
    write_table(f_out, finger_encoder, "TABLE_FLIP_180")
    write_table(f_out, arm_encoder, None)
    f_out.write("\n")

    print("------------ 生成TABLE_FLIP_ADJUST_90 ------------")
    from motion_flip_and_adjust import calc_motion_adjust_arm as calc_flip_adjust
    arm_encoder, finger_encoder = calc_flip_adjust(False, 90)
    f_out.write("# 翻面+对侧手臂旋转90°联动\n")
    write_table(f_out, finger_encoder, "TABLE_FLIP_ADJUST_90")
    write_table(f_out, arm_encoder, None)
    print("------------ 生成TABLE_FLIP_ADJUST_N90 ------------")
    arm_encoder, finger_encoder = calc_flip_adjust(False, -90)
    write_table(f_out, finger_encoder, "TABLE_FLIP_ADJUST_N90")
    write_table(f_out, arm_encoder, None)
    f_out.write("\n")
    
    print("------------ 生成TABLE_FLIP_ADJUST_180 ------------")
    from motion_flip_and_adjust_180  import calc_motion_adjust_arm as calc_flip_adjust_180
    arm_encoder_data_flip, arm_encoder_data_no_load, finger_encoder_start_with_zero = calc_flip_adjust_180(False)
    f_out.write("# 翻面+对侧手臂旋转180°联动\n")
    write_table(f_out, finger_encoder_start_with_zero, "TABLE_FLIP_ADJUST_180")
    write_table(f_out, arm_encoder_data_no_load, None)
    write_table(f_out, arm_encoder_data_flip, None)