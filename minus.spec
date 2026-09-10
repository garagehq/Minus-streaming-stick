# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Minus.

Build with:
    pyinstaller minus.spec

Note: Models are external and must be present at runtime:
    - PaddleOCR models: models/paddleocr/ in this repository (or MINUS_OCR_MODEL_DIR)
    - VLM model: /home/radxa/axera_models/minus-v0.1/ (https://huggingface.co/TheGarageDev/Minus-v0.1)

The executable expects these to be in their standard locations.
"""

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all submodules for complex packages
hiddenimports = [
    # GStreamer/GI bindings
    'gi',
    'gi.repository',
    'gi.repository.Gst',
    'gi.repository.GLib',
    'gi.repository.GObject',

    # OpenCV
    'cv2',

    # Numeric/scientific
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',

    # OCR dependencies
    'pyclipper',
    'shapely',
    'shapely.geometry',
    'shapely.ops',

    # VLM dependencies
    'pexpect',
    'ptyprocess',

    # Standard library modules that might be missed
    'logging.handlers',
    'concurrent.futures',
    'dataclasses',
    'pathlib',
    'threading',
    'subprocess',
    're',
    'argparse',
    'signal',
    'time',
    'random',
    'os',
    'sys',
]

# Try to collect GI submodules
try:
    hiddenimports += collect_submodules('gi')
except Exception:
    pass

# Source modules to include
src_modules = [
    ('src/ocr.py', 'src'),
    ('src/vlm.py', 'src'),
    ('src/ad_blocker.py', 'src'),
    ('src/audio.py', 'src'),
    ('src/health.py', 'src'),
    ('src/__init__.py', 'src'),
]

a = Analysis(
    ['minus.py'],
    pathex=[],
    binaries=[],
    datas=src_modules,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='minus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
