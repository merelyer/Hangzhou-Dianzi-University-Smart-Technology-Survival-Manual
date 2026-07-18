"""构建轻量 exe — ONNX + Pillow (无 TF/OpenCV)"""
import PyInstaller.__main__
import os, shutil

base = os.path.dirname(os.path.abspath(__file__))

for d in ['build', 'dist']:
    p = os.path.join(base, d)
    if os.path.exists(p): shutil.rmtree(p)

PyInstaller.__main__.run([
    'DigitTester.py',
    '--name=DigitTester',
    '--onedir',
    '--windowed',
    '--add-data', f'model.onnx;.',
    '--hidden-import=onnxruntime',
    '--hidden-import=onnxruntime.capi',
    '--hidden-import=PIL',
    '--hidden-import=PIL.Image',
    '--hidden-import=numpy',
    '--hidden-import=tkinter',
    '--exclude-module=tensorflow',
    '--exclude-module=keras',
    '--exclude-module=cv2',
    '--exclude-module=sklearn',
    '--exclude-module=scipy',
    '--exclude-module=matplotlib',
    '--exclude-module=pandas',
    '--exclude-module=h5py',
    '--noconfirm',
    '--clean',
])
