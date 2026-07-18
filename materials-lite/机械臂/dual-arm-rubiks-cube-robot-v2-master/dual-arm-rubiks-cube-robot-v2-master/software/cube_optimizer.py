"""
魔方机械臂动作序列优化程序

该程序用于将魔方还原步骤序列转换为机械臂可执行的最小动作单元序列，并通过动态规划算法优化翻面操作路径，
以最小化总执行时间。程序主要包含以下功能：

1. 状态管理：跟踪机械臂的当前夹持状态（DF/DB/DL等24种状态）和手臂垂直状态
2. 动作转换：
   - 翻面操作（6种类型）：90°/180°向左/向右翻转
   - 旋转操作：90°/180°顺时针/逆时针旋转
3. 时间优化：使用动态规划算法寻找最优动作序列，最小化总执行时间
4. 指令生成：输出机械臂可执行的底层指令序列

关键概念定义：
- 魔方状态：由两个字母表示当前机械臂夹持的面（如'DF'表示左臂夹D面，右臂夹F面）
- 手臂垂直状态：布尔值表示手臂是否处于垂直位置（影响是否需要空转操作）
- 操作类型：
  空转操作（100ms）：['L2', 'L+N', 'L0'] 等
  90°翻面（170ms）：['R1', 'L+F', 'R0'] 等
  180°翻面（240ms）：['L1', 'R*F', 'L0'] 等
  90°旋转（55ms）：['L+T'] 等
  180°旋转（95ms）：['L*T'] 等

算法核心逻辑：
1. 动态规划表(dp)维护状态：(当前魔方状态, 左臂垂直状态, 右臂垂直状态)
2. 对于每个魔方动作（如"R2"）：
   a. 检查当前状态能否直接执行动作
   b. 如果不能，使用BFS搜索最多2步内的翻面路径
   c. 评估所有候选路径的时间消耗
   d. 更新动态规划表，保留最优路径
3. 输出总时间最短的机械臂指令序列

输入输出说明：
- 输入：初始状态(如'DF')，魔方操作序列(如"R2 U F' D")
- 输出：机械臂指令序列(如['L2', 'L+N', 'L0', 'R+T'])

注意事项：
1. 状态转换表(cube_state_transitions)完整定义了24种状态间的转换关系
2. 翻面操作限制在2步内，确保搜索效率
3. 时间计算包含所有操作的空转、翻面和旋转时间
4. 使用日志系统记录运行状态(Debug/Info/Error级别)

执行流程示例：
1. 初始化：状态=DF, 手臂=水平(False)
2. 处理动作"D"：
   - 直接执行：添加旋转指令['L2','L+N','L0','L*T'] (195ms)
3. 处理动作"R2"：
   - 需要翻面：DR状态可执行
   - 添加翻面指令+旋转指令
4. 输出完整指令序列

"""
from collections import deque
import logging
from enum import IntEnum
import time

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 时间常量定义
class TimeCost(IntEnum):
    ARM_BACK = 100       # 空转操作时间
    FLIP_90 = 170    # 90度翻面时间
    FLIP_180 = 240   # 180度翻面时间
    ROTATE_90 = 55   # 90度旋转时间
    ROTATE_180 = 95  # 180度旋转时间
    GRIP_REDUNDANT = -30  # 冗余夹持指令减少时间

class ArmSide(IntEnum):
    LEFT = 1
    RIGHT = 2

class FlipOperation(IntEnum):
    FLIP_90_LEFT_CW = 0
    FLIP_180_LEFT = 1
    FLIP_90_LEFT_CCW = 2
    FLIP_90_RIGHT_CW = 3
    FLIP_180_RIGHT = 4
    FLIP_90_RIGHT_CCW = 5

# 完整的六种操作状态转换表
# 格式: {当前状态: [左90CW, 左180, 左90CCW, 右90CW, 右180, 右90CCW]}
cube_state_transitions = {
    'DF':['DL', 'DB', 'DR', 'RF', 'UF', 'LF'],
    'DB':['DR', 'DF', 'DL', 'LB', 'UB', 'RB'],
    'DL':['DB', 'DR', 'DF', 'FL', 'UL', 'BL'],
    'DR':['DF', 'DL', 'DB', 'BR', 'UR', 'FR'],
    'UF':['UR', 'UB', 'UL', 'LF', 'DF', 'RF'],
    'UB':['UL', 'UF', 'UR', 'RB', 'DB', 'LB'],
    'UL':['UF', 'UR', 'UB', 'BL', 'DL', 'FL'],
    'UR':['UB', 'UL', 'UF', 'FR', 'DR', 'BR'],
    'FL':['FD', 'FR', 'FU', 'UL', 'BL', 'DL'],
    'FR':['FU', 'FL', 'FD', 'DR', 'BR', 'UR'],
    'FU':['FL', 'FD', 'FR', 'RU', 'BU', 'LU'],
    'FD':['FR', 'FU', 'FL', 'LD', 'BD', 'RD'],
    'BL':['BU', 'BR', 'BD', 'DL', 'FL', 'UL'],
    'BR':['BD', 'BL', 'BU', 'UR', 'FR', 'DR'],
    'BU':['BR', 'BD', 'BL', 'LU', 'FU', 'RU'],
    'BD':['BL', 'BU', 'BR', 'RD', 'FD', 'LD'],
    'LF':['LU', 'LB', 'LD', 'DF', 'RF', 'UF'],
    'LB':['LD', 'LF', 'LU', 'UB', 'RB', 'DB'],
    'LU':['LB', 'LD', 'LF', 'FU', 'RU', 'BU'],
    'LD':['LF', 'LU', 'LB', 'BD', 'RD', 'FD'],
    'RF':['RD', 'RB', 'RU', 'UF', 'LF', 'DF'],
    'RB':['RU', 'RF', 'RD', 'DB', 'LB', 'UB'],
    'RU':['RF', 'RD', 'RB', 'BU', 'LU', 'FU'],
    'RD':['RB', 'RU', 'RF', 'FD', 'LD', 'BD'],
}

# 创建状态字符串到索引的映射
state_str_list = list(cube_state_transitions.keys())
state_to_index = {state: idx for idx, state in enumerate(state_str_list)}

def encode_state(state_str, l_arm, r_arm):
    """将状态编码为0-95的整数"""
    state_idx = state_to_index[state_str]
    return (state_idx << 2) | (l_arm << 1) | r_arm

def decode_state(state_int):
    """将整数状态解码为可读字符串和手臂状态"""
    state_idx = state_int >> 2
    l_arm = bool((state_int >> 1) & 1)
    r_arm = bool(state_int & 1)
    return state_str_list[state_idx], l_arm, r_arm

# 创建序列池
flip_sequences = []  # 存储(time_cost, sequence)元组
# 查找表: 96种状态 x 6种操作 -> (新状态, 时间消耗, 指令序列)
flip_table = [[None] * 6 for _ in range(96)]

def build_flip_table():
    """构建翻转操作查找表，支持最多两步操作，使用二维数组存储"""
    global flip_sequences
    
    # 重置序列池
    flip_sequences = []
    
    # 初始化查找表：96x96 的数组，初始值为-1（无效索引）
    global flip_table
    flip_table = [[-1] * 96 for _ in range(96)]
    
    # 添加0步路径（自身）到序列池
    zero_step_index = -1
    zero_step_tuple = (0, tuple())
    if zero_step_tuple not in flip_sequences:
        flip_sequences.append(zero_step_tuple)
    zero_step_index = flip_sequences.index(zero_step_tuple)
    
    # 使用BFS搜索每个状态的两步内所有可能路径
    for start_state_int in range(96):
        logger.debug(f"构建状态 {start_state_int} 的翻转路径...")
        start_state_str, start_l_arm, start_r_arm = decode_state(start_state_int)
        
        # 存储所有找到的路径：{目标状态: (时间消耗, 序列)}
        paths = {}
        
        # 0步路径：自身
        paths[start_state_int] = (0, [])
        
        # 1步路径
        for flip_op in range(6):
            try:
                new_state_str, new_l_arm, new_r_arm, time_cost, sequence = apply_flip_operation(
                    start_state_str, start_l_arm, start_r_arm, flip_op
                )
                new_state_int = encode_state(new_state_str, new_l_arm, new_r_arm)
                
                # 如果是新状态或找到更短路径
                if (new_state_int not in paths or 
                    time_cost < paths[new_state_int][0]):
                    paths[new_state_int] = (time_cost, sequence)
            except Exception as e:
                logger.error(f"应用翻转操作失败: start={start_state_int}, op={flip_op}, error={e}")
        
        # 2步路径
        for flip_op1 in range(6):            
            try:
                s1_str, s1_l_arm, s1_r_arm, time1, seq1 = apply_flip_operation(
                    start_state_str, start_l_arm, start_r_arm, flip_op1
                )
                
                for flip_op2 in range(6):
                    try:
                        s2_str, s2_l_arm, s2_r_arm, time2, seq2 = apply_flip_operation(
                            s1_str, s1_l_arm, s1_r_arm, flip_op2
                        )
                        s2_int = encode_state(s2_str, s2_l_arm, s2_r_arm)
                        
                        total_time = time1 + time2
                        total_sequence = seq1 + seq2
                        
                        # 如果是新状态或找到更短路径
                        if (s2_int not in paths or 
                            total_time < paths[s2_int][0]):
                            paths[s2_int] = (total_time, total_sequence)
                    except Exception as e:
                        logger.error(f"应用第二步翻转失败: start={start_state_int}, op1={flip_op1}, op2={flip_op2}, error={e}")
            except Exception as e:
                logger.error(f"应用第一步翻转失败: start={start_state_int}, op1={flip_op1}, error={e}")
        
        # 将最优路径存入查找表
        for target_state_int, (time_cost, sequence) in paths.items():
            # 在序列池中查找或添加序列
            seq_tuple = (time_cost, tuple(sequence))
            if seq_tuple not in flip_sequences:
                flip_sequences.append(seq_tuple)
            seq_index = flip_sequences.index(seq_tuple)
            
            # 存入二维数组
            flip_table[start_state_int][target_state_int] = seq_index

def print_state(state_int):
    """打印状态的可读表示"""
    state_str, l_arm, r_arm = decode_state(state_int)
    l_status = "垂直" if l_arm else "水平"
    r_status = "垂直" if r_arm else "水平"
    return f"{state_str} (左臂: {l_status}, 右臂: {r_status})"

def print_flip_table():
    """按照指定格式打印翻转操作查找表"""
    # 输出翻转序列池
    print("// 翻转序列池（消耗时间，动作序列），时间计算使用software/cube_optimizer.py进行")
    print(f"const FlipSequence flip_sequences[{len(flip_sequences)}] = {{")
    for i, (time_cost, sequence_tuple) in enumerate(flip_sequences):
        # 将元组转换为空格分隔的字符串，并在末尾添加空格
        sequence_str = " ".join(sequence_tuple) + (" " if sequence_tuple else "")
        # 格式化输出，注意转义双引号
        print(f"    {{{time_cost}, \"{sequence_str}\"}}", end="")
        if i < len(flip_sequences) - 1:
            print(",")
        else:
            print()
    print("};")
    
    # 输出翻转表
    print("\n// 96x96翻转表")
    print("const int8_t flip_table[96][96] = {")
    for i in range(96):
        # 每行开头
        print("    {", end="")
        # 输出96个元素，用逗号分隔
        for j in range(96):
            print(flip_table[i][j], end="")
            if j < 95:
                print(",", end="")
        # 行尾处理
        print("}", end="")
        if i < 95:
            print(",")
        else:
            print()
    print("};")


def apply_flip_operation(state_str, l_arm, r_arm, flip_op_index):
    """应用翻转操作并返回新状态、时间和指令序列"""
    flip_op = FlipOperation(flip_op_index)
    sequence = []
    time_cost = 0

    # 检查是否需要空转操作
    if (flip_op in [FlipOperation.FLIP_90_LEFT_CW, FlipOperation.FLIP_180_LEFT, FlipOperation.FLIP_90_LEFT_CCW] and r_arm):
        sequence.extend(['R2', 'R+N', 'R0'])
        time_cost += TimeCost.ARM_BACK
        r_arm = False
    elif (flip_op in [FlipOperation.FLIP_90_RIGHT_CW, FlipOperation.FLIP_180_RIGHT, FlipOperation.FLIP_90_RIGHT_CCW] and l_arm):
        sequence.extend(['L2', 'L+N', 'L0'])
        time_cost += TimeCost.ARM_BACK
        l_arm = False

    # 执行翻面操作
    if flip_op == FlipOperation.FLIP_90_LEFT_CW:
        sequence.extend(['R1', 'L+F', 'R0'])
        time_cost += TimeCost.FLIP_90
        l_arm = not l_arm
    elif flip_op == FlipOperation.FLIP_180_LEFT:
        sequence.extend(['R1', 'L*F', 'R0'])
        time_cost += TimeCost.FLIP_180
    elif flip_op == FlipOperation.FLIP_90_LEFT_CCW:
        sequence.extend(['R1', 'L-F', 'R0'])
        time_cost += TimeCost.FLIP_90
        l_arm = not l_arm
    elif flip_op == FlipOperation.FLIP_90_RIGHT_CW:
        sequence.extend(['L1', 'R+F', 'L0'])
        time_cost += TimeCost.FLIP_90
        r_arm = not r_arm
    elif flip_op == FlipOperation.FLIP_180_RIGHT:
        sequence.extend(['L1', 'R*F', 'L0'])
        time_cost += TimeCost.FLIP_180
    elif flip_op == FlipOperation.FLIP_90_RIGHT_CCW:
        sequence.extend(['L1', 'R-F', 'L0'])
        time_cost += TimeCost.FLIP_90
        r_arm = not r_arm

    # 优化冗余指令
    i = 0
    while i < len(sequence) - 1:
        if (sequence[i] == 'L0' and sequence[i+1] == 'L1') or \
           (sequence[i] == 'R0' and sequence[i+1] == 'R1'):
            del sequence[i:i+2]
            time_cost += TimeCost.GRIP_REDUNDANT
            continue
        i += 1

    new_state = cube_state_transitions[state_str][flip_op_index]
    return new_state, l_arm, r_arm, time_cost, sequence

def calculate_twist_time_and_state(state_int, action_str):
    """计算旋转操作的时间和状态变化"""
    state_str, l_arm, r_arm = decode_state(state_int)
    face = action_str[0]
    modifier = action_str[1] if len(action_str) > 1 else ''
    sequence = []
    time_cost = 0

    # 确定使用哪只手臂
    if state_str[0] == face:
        arm_side = ArmSide.LEFT
    elif state_str[1] == face:
        arm_side = ArmSide.RIGHT
    else:
        logger.error(f"状态 {state_str} 无法操作目标面 {face}")
        return None, None, None, None

    # 检查是否需要空转操作
    if arm_side == ArmSide.LEFT and r_arm:
        sequence.extend(['R2', 'R+N', 'R0'])
        time_cost += TimeCost.ARM_BACK
        r_arm = False
    elif arm_side == ArmSide.RIGHT and l_arm:
        sequence.extend(['L2', 'L+N', 'L0'])
        time_cost += TimeCost.ARM_BACK
        l_arm = False

    # 执行旋转操作
    if arm_side == ArmSide.LEFT:
        if modifier == "'":  # 逆时针90°
            sequence.append('L-T')
            time_cost += TimeCost.ROTATE_90
            l_arm = not l_arm
        elif modifier == '2':  # 180°
            sequence.append('L*T')
            time_cost += TimeCost.ROTATE_180
        else:  # 顺时针90°
            sequence.append('L+T')
            time_cost += TimeCost.ROTATE_90
            l_arm = not l_arm
    else:  # 右臂操作
        if modifier == "'":  # 逆时针90°
            sequence.append('R-T')
            time_cost += TimeCost.ROTATE_90
            r_arm = not r_arm
        elif modifier == '2':  # 180°
            sequence.append('R*T')
            time_cost += TimeCost.ROTATE_180
        else:  # 顺时针90°
            sequence.append('R+T')
            time_cost += TimeCost.ROTATE_90
            r_arm = not r_arm

    new_state_int = encode_state(state_str, l_arm, r_arm)
    return time_cost, new_state_int, sequence

def cube_tweak_str_time(initial_state, action_str):
    """优化动作序列并返回机械臂指令序列"""
    actions = action_str.split()
    if not actions:
        return [], 0, encode_state(initial_state, False, False)
    
    # 初始化动态规划表
    dp = {}
    initial_state_int = encode_state(initial_state, False, False)
    dp[initial_state_int] = (0, [])  # (累计时间, 指令序列)
    
    # 遍历每个动作
    for idx, action in enumerate(actions):
        new_dp = {}
        logger.info(f"处理动作 {idx+1}/{len(actions)}: {action}")
        
        for state_int, (base_time, base_sequence) in dp.items():
            logger.debug(f"  当前状态: {state_int}, 累计时间: {base_time}ms, 序列长度: {len(base_sequence)}")
            
            # 存储候选状态
            candidate_states = []
            state_str, _, _ = decode_state(state_int)
            
            # 检查当前状态是否可直接执行动作
            if action[0] in state_str:
                candidate_states.append((state_int, 0, []))
            
            # 选项2：使用查找表中的翻转路径
            for target_state_int in range(96):
                seq_index = flip_table[state_int][target_state_int]
                if seq_index == -1:  # 无效路径
                    continue
                
                # 获取序列详情
                time_cost, sequence_tuple = flip_sequences[seq_index]
                sequence = list(sequence_tuple)
                
                # 检查目标状态是否能执行动作
                target_str, _, _ = decode_state(target_state_int)
                if action[0] in target_str:
                    candidate_states.append((target_state_int, time_cost, sequence))
            
            # 处理所有候选状态
            for s1_int, flip_time, flip_sequence in candidate_states:
                try:
                    twist_time, new_state_int, twist_sequence = calculate_twist_time_and_state(
                        s1_int, action
                    )
                except Exception as e:
                    logger.error(f"计算旋转操作失败: {e}")
                    continue
                
                total_time = base_time + flip_time + twist_time
                total_sequence = base_sequence + flip_sequence + twist_sequence
                
                # 更新新状态的DP表
                if new_state_int not in new_dp or total_time < new_dp[new_state_int][0]:
                    new_dp[new_state_int] = (total_time, total_sequence)
                    logger.debug(f"      新状态: {new_state_int}, 时间: {total_time}ms, 序列长度: {len(total_sequence)}")
        
        if not new_dp:
            logger.error(f"无法处理动作 {action}，无有效状态")
            return [], 0, state_int
        
        dp = new_dp
    
    # 找到最小时间对应的序列和状态
    min_time = float('inf')
    best_sequence = []
    best_state = None
    
    for state_int, (total_time, sequence) in dp.items():
        if total_time < min_time:
            min_time = total_time
            best_sequence = sequence
            best_state = state_int
    
    logger.info(f"最小总时间: {min_time}ms, 总指令数: {len(best_sequence)}")
    return best_sequence, min_time, best_state

def adjust_final_position(final_state_int):
    """调整最终位置使手臂水平"""
    state_str, l_arm, r_arm = decode_state(final_state_int)
    adjust_sequence = []
    adjust_time = 0
    
    # 添加调整操作使旋转臂水平
    if l_arm:
        adjust_sequence.extend(['L2', 'L+N', 'L0'])
        adjust_time += TimeCost.ARM_BACK
        l_arm = False
    if r_arm:
        adjust_sequence.extend(['R2', 'R+N', 'R0'])
        adjust_time += TimeCost.ARM_BACK
        r_arm = False
    
    new_state_int = encode_state(state_str, l_arm, r_arm)
    return adjust_sequence, adjust_time, new_state_int

def process_test_cases(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 提取测试用例
    test_cases = []
    
    for line in lines:
        line = line.strip()
        test_cases.append(line)
    
    logger.info(f"找到 {len(test_cases)} 个测试用例")
    
    results = []
    initial_state = 'RU'
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"处理测试用例 {i+1}/{len(test_cases)}")
        logger.info(f"动作序列: {test_case}")
        
        start_time = time.time()
        try:
            sequence, total_time = solution_to_motion(initial_state, test_case)
        except Exception as e:
            logger.error(f"处理测试用例 {i+1} 时出错: {str(e)}")
            sequence = []
            total_time = 0
        
        exec_time = (time.time() - start_time) * 1000
        logger.info(f"处理完成, 耗时: {exec_time:.2f}ms")
        
        results.append({
            "test_case": test_case,
            "sequence": sequence,
            "total_time": total_time,
            "exec_time": exec_time
        })
    
    # 写入结果文件
    with open(output_file, 'w') as f:
        for i, result in enumerate(results):
            f.write(f"测试用例 {i+1}: {result['test_case']}\n")
            if result['sequence']:
                f.write("优化序列: " + ' '.join(result['sequence']) + "\n")
                f.write(f"总耗时: {result['total_time']}ms\n")
                f.write(f"处理时间: {result['exec_time']:.2f}ms\n")
            else:
                f.write("无法生成优化序列\n")
            f.write("\n")
    
    logger.info(f"结果已保存到: {output_file}")

def solution_to_motion(initial_state, action_str):
    """将魔方解法转换为机械臂动作序列"""
    sequence, total_time, final_state_int = cube_tweak_str_time(initial_state, action_str)
    
    # 在生成完整序列后添加后处理步骤
    if sequence:
        adjust_seq, adjust_time, final_state_int = adjust_final_position(final_state_int)
        sequence += adjust_seq
        total_time += adjust_time

        logger.info(' '.join(sequence))
        logger.info(f"总耗时: {total_time}ms")
    else:
        logger.error("无法生成指令序列")
    return sequence, total_time

# 构建翻转操作查找表

build_flip_table()
# print_flip_table()

if __name__ == "__main__":
    # 单条测试用例处理
    action_str = "D R2 U2 L2 B2 D F2 D L2 B2 D L2 F' U B R F2 U L D2 R2"
    initial_state = 'RU'
    
    solution_to_motion(initial_state, action_str)
    
    logger.setLevel(logging.ERROR)
    # 批量处理测试用例
    input_file = "cube_solution_testcase.txt"
    output_file = "cube_solution_results.txt"
    process_test_cases(input_file, output_file)
