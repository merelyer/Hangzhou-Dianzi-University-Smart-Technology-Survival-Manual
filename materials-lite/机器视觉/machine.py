import os
from PIL import Image

# 机器视觉总目录
root_dir = r"C:\Users\aa257\Desktop\复习\机器视觉"

# PDF输出目录
pdf_dir = os.path.join(root_dir, "PDF")
os.makedirs(pdf_dir, exist_ok=True)

# 遍历每个章节文件夹
for chapter in os.listdir(root_dir):

    chapter_path = os.path.join(root_dir, chapter)

    # 只处理文件夹
    if not os.path.isdir(chapter_path):
        continue

    # 跳过输出目录
    if chapter == "PDF":
        continue

    print(f"\n处理章节：{chapter}")

    imgs = []

    for f in os.listdir(chapter_path):

        if f.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ):
            imgs.append(os.path.join(chapter_path, f))

    # 按文件名排序
    imgs.sort()

    if len(imgs) == 0:
        print("没有找到图片")
        continue

    print(f"找到 {len(imgs)} 张图片")

    image_list = []

    for img_path in imgs:
        img = Image.open(img_path).convert("RGB")
        image_list.append(img)

    pdf_path = os.path.join(pdf_dir, f"{chapter}.pdf")

    image_list[0].save(
        pdf_path,
        save_all=True,
        append_images=image_list[1:]
    )

    print("生成成功：", pdf_path)

print("\n全部完成！")