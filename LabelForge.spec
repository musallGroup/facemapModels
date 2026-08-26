# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[('D:/Miniconda/envs/fm_front/Library/bin/mkl*.dll', '.'), ('D:/Miniconda/envs/fm_front/Library/bin/libiomp*.dll', '.')],
    datas=[('Z:/Team/Mick/LabelForge/LabelForge_v0_0_1/labelforge/assets', 'labelforge/assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LabelForge',
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
    icon=['Z:/Team/Mick/LabelForge/Corporate_Design/Logos/labelforge_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LabelForge',
)
