# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['DigitTester.py'],
    pathex=[],
    binaries=[],
    datas=[('model.onnx', '.')],
    hiddenimports=['onnxruntime', 'onnxruntime.capi', 'PIL', 'PIL.Image', 'numpy', 'tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'keras', 'cv2', 'sklearn', 'scipy', 'matplotlib', 'pandas', 'h5py'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DigitTester',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DigitTester',
)
