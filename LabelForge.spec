# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[('D:/Miniconda/envs/fm_front/Library/bin/mkl*.dll', '.'), ('D:/Miniconda/envs/fm_front/Library/bin/libiomp*.dll', '.')],
    datas=[('labelforge/assets', 'labelforge/assets'),
           ('labelforge/ui/training_workspace/facemap_training_adapter.py', 'labelforge/ui/training_workspace'),
           ('labelforge/ui/training_workspace/facemap_qc.py', 'labelforge/ui/training_workspace')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# Transitive scientific packages can expose ICU/API-set DLLs that Windows
# loads before Qt's matching runtime. Excluding these copies prevents the
# QtCore "specified procedure could not be found" startup failure.
_qt_conflicts = {
    'icudt78.dll',
    'icuuc.dll',
    'api-ms-win-core-fibers-l1-1-1.dll',
    'api-ms-win-core-kernel32-legacy-l1-1-1.dll',
    'api-ms-win-core-sysinfo-l1-2-0.dll',
}
a.binaries = [item for item in a.binaries if item[0].lower() not in _qt_conflicts]
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
    icon=['labelforge/assets/labelforge_icon.ico'],
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
