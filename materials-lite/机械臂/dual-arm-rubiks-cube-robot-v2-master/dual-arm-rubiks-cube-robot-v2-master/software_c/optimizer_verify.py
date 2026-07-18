from collections import deque
import logging
import re

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 完整的状态转换表
cube_state_transitions = {
    'DF': ['DL', 'DB', 'DR', 'RF', 'UF', 'LF'],
    'DB': ['DR', 'DF', 'DL', 'LB', 'UB', 'RB'],
    'DL': ['DB', 'DR', 'DF', 'FL', 'UL', 'BL'],
    'DR': ['DF', 'DL', 'DB', 'BR', 'UR', 'FR'],
    'UF': ['UR', 'UB', 'UL', 'LF', 'DF', 'RF'],
    'UB': ['UL', 'UF', 'UR', 'RB', 'DB', 'LB'],
    'UL': ['UF', 'UR', 'UB', 'BL', 'DL', 'FL'],
    'UR': ['UB', 'UL', 'UF', 'FR', 'DR', 'BR'],
    'FL': ['FD', 'FR', 'FU', 'UL', 'BL', 'DL'],
    'FR': ['FU', 'FL', 'FD', 'DR', 'BR', 'UR'],
    'FU': ['FL', 'FD', 'FR', 'RU', 'BU', 'LU'],
    'FD': ['FR', 'FU', 'FL', 'LD', 'BD', 'RD'],
    'BL': ['BU', 'BR', 'BD', 'DL', 'FL', 'UL'],
    'BR': ['BD', 'BL', 'BU', 'UR', 'FR', 'DR'],
    'BU': ['BR', 'BD', 'BL', 'LU', 'FU', 'RU'],
    'BD': ['BL', 'BU', 'BR', 'RD', 'FD', 'LD'],
    'LF': ['LU', 'LB', 'LD', 'DF', 'RF', 'UF'],
    'LB': ['LD', 'LF', 'LU', 'UB', 'RB', 'DB'],
    'LU': ['LB', 'LD', 'LF', 'FU', 'RU', 'BU'],
    'LD': ['LF', 'LU', 'LB', 'BD', 'RD', 'FD'],
    'RF': ['RD', 'RB', 'RU', 'UF', 'LF', 'DF'],
    'RB': ['RU', 'RF', 'RD', 'DB', 'LB', 'UB'],
    'RU': ['RF', 'RD', 'RB', 'BU', 'LU', 'FU'],
    'RD': ['RB', 'RU', 'RF', 'FD', 'LD', 'BD'],
}

def motion_to_solution(initial_state, sequence):
    """
    将机械臂动作序列转换为魔方还原字符串
    :param initial_state: 初始魔方状态字符串（如'RU'）
    :param sequence: 机械臂动作序列字符串
    :return: 魔方还原字符串
    """
    state_str = initial_state
    l_arm = False  # 左臂垂直状态（初始水平）
    r_arm = False  # 右臂垂直状态（初始水平）
    moves = []     # 存储还原步骤
    
    # 分割指令序列
    cmds = sequence.split()
    
    for cmd in cmds:
        if cmd.endswith('T'):  # 旋转操作
            arm = cmd[0]
            op_type = cmd[1:]
            
            # 确定操作的面（左臂操作第一个面，右臂操作第二个面）
            face = state_str[0] if arm == 'L' else state_str[1]
            
            # 确定旋转方向
            if op_type == '+T':
                move_str = face
            elif op_type == '-T':
                move_str = face + "'"
            elif op_type == '*T':
                move_str = face + '2'
            else:
                logger.warning(f"未知的旋转操作类型: {cmd}")
                continue
                
            moves.append(move_str)
            
            # 更新手臂状态（90°旋转会翻转状态，180°不会）
            if op_type in ['+T', '-T']:
                if arm == 'L':
                    l_arm = not l_arm
                else:
                    r_arm = not r_arm
                    
        elif cmd.endswith('F'):  # 翻面操作
            arm = cmd[0]
            op_type = cmd[1:]
            
            # 确定操作类型对应的索引
            if arm == 'L':
                if op_type == '+F': index = 0
                elif op_type == '*F': index = 1
                elif op_type == '-F': index = 2
                else: continue
            else:  # R
                if op_type == '+F': index = 3
                elif op_type == '*F': index = 4
                elif op_type == '-F': index = 5
                else: continue
            
            # 更新魔方状态
            next_states = cube_state_transitions.get(state_str, [])
            if index < len(next_states):
                state_str = next_states[index]
            else:
                logger.error(f"状态 {state_str} 没有索引 {index} 的转换")
            
            # 更新手臂状态（90°翻转会翻转状态，180°不会）
            if op_type in ['+F', '-F']:
                if arm == 'L':
                    l_arm = not l_arm
                else:
                    r_arm = not r_arm
                    
        elif cmd in ['L0', 'R0']:  # 手臂调整指令
            if cmd == 'L0':
                l_arm = False
            else:  # R0
                r_arm = False
                
        # 其他指令（L1, R1, L2, R2, L+N, R+N）不影响状态，忽略
        
    return ' '.join(moves)

def parse_log_file(file_path):
    """
    解析日志文件，提取测试用例和指令序列
    :param file_path: 日志文件路径
    :return: 测试用例列表，每个元素是(测试用例编号, 原始魔方字符串, 指令序列)
    """
    test_cases = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 检查是否以"测试用例"开头
        if line.startswith("测试用例"):
            # 提取测试用例编号和魔方字符串
            parts = line.split(':', 1)
            if len(parts) < 2:
                i += 1
                continue
            test_case_id = parts[0].strip()
            cube_str = parts[1].strip()
            
            # 检查下一行是否包含指令序列
            if i+1 < len(lines) and lines[i+1].startswith("指令序列:"):
                seq_line = lines[i+1].split(':', 1)
                if len(seq_line) < 2:
                    i += 1
                    continue
                sequence = seq_line[1].strip()
                test_cases.append((test_case_id, cube_str, sequence))
                i += 3  # 跳过下一行（总时间行）
                continue
        i += 1
        
    return test_cases

def verify_test_cases(test_cases):
    """
    验证测试用例的正确性
    :param test_cases: 测试用例列表，每个元素是(测试用例编号, 原始魔方字符串, 指令序列)
    :return: 验证结果列表，每个元素是(测试用例编号, 原始字符串, 转换字符串, 是否匹配)
    """
    results = []
    initial_state = 'RU'
    
    for test_case in test_cases:
        test_case_id, cube_str, sequence = test_case
        try:
            # 转换指令序列
            converted_str = motion_to_solution(initial_state, sequence)
            # 比较时忽略空格
            original_no_space = cube_str.replace(" ", "")
            converted_no_space = converted_str.replace(" ", "")
            match = original_no_space == converted_no_space
            results.append((test_case_id, cube_str, converted_str, match))
        except Exception as e:
            logger.error(f"处理测试用例 {test_case_id} 时出错: {str(e)}")
            results.append((test_case_id, cube_str, "转换失败", False))
    
    return results

def print_verification_results(results):
    """
    打印验证结果并统计正确率
    :param results: 验证结果列表
    """
    total = len(results)
    correct = sum(1 for r in results if r[3])
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"验证完成，总共 {total} 个测试用例")
    print(f"正确: {correct}, 错误: {total - correct}, 正确率: {accuracy:.2f}%")
    
    print("\n详细结果:")
    for res in results:
        test_case_id, original, converted, match = res
        status = "通过" if match else "失败"
        print(f"{test_case_id}: {status}")
        print(f"  原始: {original}")
        print(f"  转换: {converted}")
        print()

if __name__ == "__main__":
    # 解析日志文件
    test_cases = parse_log_file("log.txt")
    
    # 验证测试用例
    results = verify_test_cases(test_cases)
    
    # 打印结果
    print_verification_results(results)

# if __name__ == "__main__":
#     # 测试用例
#     initial_state = 'RU'
#     sequence = ('R1 L*F R0 R+T R2 R+N L*F R0 L*T R*T R1 L-F R0 L2 L+N R*F L0 L*T R*T '
#                 'R1 L+F R0 L2 L+N R-F L0 R+T L*T R+T L1 R+F L0 L*T L1 R+F L0 R2 R+N R0 '
#                 'L*T R+T L1 R-F L0 L*T R1 L+F R0 L2 L+N R-F L0 R-T L+T R1 L*F R0 L2 L+N '
#                 'R+F L0 R+T L+T R1 L+F R0 L1 R-F L0 R2 R+N R0 L*T R+T R2 R+N L+F R0 L2 '
#                 'L+N R-F L0 R+T L*T R1 L*F R0 R*T')
    
#     result = motion_to_solution(initial_state, sequence)
#     print("输入序列:")
#     print(sequence)
#     print("\n还原结果:")
#     print(result)