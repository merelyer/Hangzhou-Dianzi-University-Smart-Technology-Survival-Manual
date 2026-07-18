import os
import sys
import time
import subprocess
import importlib.util
from xml.sax.saxutils import escape

# 清除代理环境变量，避免 ascii 报错
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy"
]:
    os.environ.pop(k, None)

def install_if_missing(package, import_name=None):
    if import_name is None:
        import_name = package.replace("-", "_")
    if importlib.util.find_spec(import_name) is None:
        print(f"正在安装：{package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing("rapidocr-onnxruntime", "rapidocr_onnxruntime")
install_if_missing("opencv-python", "cv2")
install_if_missing("pillow", "PIL")
install_if_missing("openai", "openai")
install_if_missing("reportlab", "reportlab")

from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root_dir = r"C:\Users\aa257\Desktop\复习\机器视觉"
target_chapter = "9.1_区域特征"

output_dir = os.path.join(root_dir, "中文讲义PDF")
cache_dir = os.path.join(output_dir, "cache", target_chapter)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

DEEPSEEK_API_KEY = "这里换成你的DeepSeek_API_KEY"

client = OpenAI(
    api_key="sk-11c791f06c564d81ae853302d1412c42",
    base_url="https://api.deepseek.com",
    timeout=120
)

ocr = RapidOCR()

font_path = r"C:\Windows\Fonts\msyh.ttc"
pdfmetrics.registerFont(TTFont("MicrosoftYaHei", font_path))

img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def ocr_image(img_path):
    result, _ = ocr(img_path)

    if not result:
        return ""

    texts = []
    for item in result:
        text = item[1]
        if text and text.strip():
            texts.append(text.strip())

    return "\n".join(texts)


def translate_text(text):
    if not text.strip():
        return "本页未识别到明显英文内容。"

    system_prompt = (
        "你是机器视觉课程助教。请把OCR识别出的英文PPT内容翻译成中文，"
        "并整理成适合期末复习的中文讲义。"
        "要求：1. 翻译准确；2. 保留机器视觉专业术语；"
        "3. 保留公式、人名、书名；4. OCR明显错误请自动修正；"
        "5. 只输出中文讲义内容，不要解释。"
    )

    for i in range(8):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                timeout=120
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"DeepSeek连接失败，第 {i + 1} 次重试")
            print(e)
            time.sleep(10)

    return "【本页翻译失败，请检查网络或VPN后重新运行】\n\nOCR原文：\n" + text


def get_cache_file(img_path):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(cache_dir, img_name + ".txt")


def get_slide_translation(img_path):
    cache_file = get_cache_file(img_path)

    if os.path.exists(cache_file):
        print("读取缓存：", os.path.basename(cache_file))
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    raw_text = ocr_image(img_path)
    cn_text = translate_text(raw_text)

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(cn_text)

    return cn_text


def make_pdf(img_paths):
    pdf_path = os.path.join(output_dir, f"{target_chapter}_中文讲义.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title_cn",
        parent=styles["Title"],
        fontName="MicrosoftYaHei",
        fontSize=18,
        leading=24,
        spaceAfter=14
    )

    body_style = ParagraphStyle(
        "body_cn",
        parent=styles["Normal"],
        fontName="MicrosoftYaHei",
        fontSize=11,
        leading=18,
        spaceAfter=8
    )

    story = []
    story.append(Paragraph(escape(target_chapter), title_style))
    story.append(Spacer(1, 12))

    for index, img_path in enumerate(img_paths, start=1):
        print(f"正在处理：{target_chapter} 第 {index} 页")

        cn_text = get_slide_translation(img_path)

        story.append(Paragraph(f"第 {index} 页", title_style))
        story.append(RLImage(img_path, width=500, height=281))
        story.append(Spacer(1, 12))

        for line in cn_text.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(escape(line), body_style))

        story.append(PageBreak())

    doc.build(story)
    print("生成成功：", pdf_path)


chapter_path = os.path.join(root_dir, target_chapter)

if not os.path.exists(chapter_path):
    print("找不到文件夹：", chapter_path)
    input("按回车退出")
    sys.exit()

imgs = [
    os.path.join(chapter_path, f)
    for f in os.listdir(chapter_path)
    if f.lower().endswith(img_exts)
]

imgs.sort()

print("本次只处理：", target_chapter)
print("共找到图片：", len(imgs))

make_pdf(imgs)

print("\n9.1 处理完成！")
print("输出目录：", output_dir)