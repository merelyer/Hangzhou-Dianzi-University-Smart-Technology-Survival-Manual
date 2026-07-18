from ctypes import *    #pip ctypes库，并导入库
test = CDLL("./Dll1.dll")    #调用当前目录下叫test的dll文件，dll文件是C生成的动态链接库
cube_count=[2,3,0,2,3,4,1]#LR LL ZL ZR O T Line
test.DFS.restype = c_char_p
def get_put_table(cube_count):
    result = test.DFS(cube_count[0], cube_count[1], cube_count[2], cube_count[3], cube_count[4], cube_count[5],
                      cube_count[6])  # 调用库里的函数sum，求和函数
    result = result.decode('gbk')
    cube_list0 = result.split(',')
    length = len(cube_list0) // 4
    cube_list = []
    for i in range(length):
        cube_list.append([])
        cube_list[i].append(cube_list0[i * 4])
        cube_list[i].append(float(cube_list0[i * 4 + 1]))
        cube_list[i].append(float(cube_list0[i * 4 + 2]))
        cube_list[i].append(float(cube_list0[i * 4 + 3]))
    print(cube_list, cube_list0[-1])  # 打印结果
    cube_sum = 0
    for i in cube_count:
        cube_sum = cube_sum + i * 4
    level = cube_sum // 10
    print("极限填满行", level)
    return cube_list
get_put_table(cube_count)