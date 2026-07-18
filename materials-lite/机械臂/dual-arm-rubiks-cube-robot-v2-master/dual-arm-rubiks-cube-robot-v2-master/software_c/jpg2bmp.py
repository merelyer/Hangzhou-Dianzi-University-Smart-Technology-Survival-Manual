# 修改python程序，用于测试
# 依次对testcase/1 ~ testcase/7,7组文件进行测试
# 转换BMP结束后自动调用./color_detect，筛选stdout输出中cube_str开头的行输出
# 每个目录下有且只有三张图片，都转换完成后才能调用color_detect
# 另外测试用例的文件名必须为captured_image_{i}.bmp，i可取1 2 3，不能乱改
# from PIL import Image
# import os

# # 确保输出目录存在
# output_dir = '../temp'
# os.makedirs(output_dir, exist_ok=True)

# for i in (1, 2, 3):
#     # 输入JPG路径
#     img_path = f'../img/testcase/4/captured_image_{i}.jpg'
    
#     # 输出BMP路径
#     bmp_path = os.path.join(output_dir, f'captured_image_{i}.bmp')
    
#     try:
#         # 打开图像并转换为BMP
#         with Image.open(img_path) as img:
#             img.save(bmp_path, 'BMP')
#         print(f'转换成功: {img_path} -> {bmp_path}')
#     except FileNotFoundError:
#         print(f'错误: 文件不存在 {img_path}')
#     except Exception as e:
#         print(f'转换失败: {img_path} - {str(e)}')

from PIL import Image
import os
import subprocess

# 确保输出目录存在
output_dir = '../temp'
os.makedirs(output_dir, exist_ok=True)

# 测试用例目录
testcase_root = '../img/testcase'

# 处理1-7号测试用例
for testcase_id in range(1, 8):
    testcase_dir = os.path.join(testcase_root, str(testcase_id))
    print(f"\n{'='*40}")
    print(f"处理测试用例: testcase/{testcase_id}")
    print(f"{'='*40}")
    
    # 检查测试用例目录是否存在
    if not os.path.exists(testcase_dir):
        print(f"错误: 测试用例目录不存在 {testcase_dir}")
        continue
        
    # 转换该测试用例中的三张图片
    bmp_paths = []
    for img_idx in (1, 2, 3):
        img_filename = f'captured_image_{img_idx}.jpg'
        img_path = os.path.join(testcase_dir, img_filename)
        bmp_path = os.path.join(output_dir, f'captured_image_{img_idx}.bmp')
        bmp_paths.append(bmp_path)
        
        try:
            # 打开图像并转换为BMP
            with Image.open(img_path) as img:
                img.save(bmp_path, 'BMP')
            print(f'转换成功: {img_filename} -> {os.path.basename(bmp_path)}')
        except FileNotFoundError:
            print(f'错误: 文件不存在 {img_filename}')
            break
        except Exception as e:
            print(f'转换失败: {img_filename} - {str(e)}')
            break
    else:  # 仅当三张图片都成功转换时执行
        print(f"\n调用 color_detect 处理测试用例 {testcase_id}")
        try:
            # 运行demo并捕获输出
# 运行demo并捕获输出
            result = subprocess.run([
                    './demo',
                    '-c',
                    '461,239,667,173,857,250,448,517,650,595,854,530',
                    '../temp/captured_image_1.bmp',
                    '../temp/captured_image_2.bmp',
                    '../temp/captured_image_3.bmp'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # 提取并打印所有"cube_str"开头的行
            print("\ncolor_detect 输出(cube_str):")
            found = False
            for line in result.stdout.splitlines():
                if line.startswith('cube_str'):
                    print(line)
                    found = True
            
            if not found:
                print("未找到cube_str开头的输出")
                
            # 如果有错误输出
            if result.stderr:
                print("\ncolor_detect 错误输出:")
                print(result.stderr)
                
        except FileNotFoundError:
            print("错误: 找不到 demo 程序")
        except Exception as e:
            print(f"运行 color_detect 失败: {str(e)}")
    
    print(f"{'='*40}\n")