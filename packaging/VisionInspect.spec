# VisionInspect — PyInstaller Packaging Spec
# Build:  pyinstaller packaging/VisionInspect.spec   (dari root proyek)
#
# Hasil di dist/VisionInspect/ :
#   VisionInspect.exe       — GUI operator (windowed, tanpa console)
#   VisionInspect-cli.exe   — konsol: --check-db / --encrypt-secret / --db-user-add
#   db.txt.example          — contoh config DB (db.txt asli TIDAK di-bundle)

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

block_cipher = None

# 1. Paket DB eksternal — sub-modul di-lazy-import (get_adapter / _connect),
#    analisis statis PyInstaller bisa melewatinya → daftarkan eksplisit.
_db_hidden = [
    'visioninspect.storage.db_txt',
    'visioninspect.storage.db_cli',
    'visioninspect.storage.credentials',
    'visioninspect.storage.external_db',
    'visioninspect.storage.user_file',
    'visioninspect.storage.multi_auth',
    'visioninspect.storage.secret_store',
    'visioninspect.storage.db_adapters',
    'visioninspect.storage.db_adapters.base',
    'visioninspect.storage.db_adapters.pg',
    'visioninspect.storage.db_adapters.mysql',
    'visioninspect.storage.db_adapters.sqlserver',
    'visioninspect.storage.db_adapters.sqlite_ad',
    'psycopg2', 'pymysql', 'pyodbc',
    'cryptography', 'cryptography.fernet',
]

# 2. UPX kadang merusak extension native (DB driver, OpenSSL) — kecualikan.
_upx_exclude = [
    'libpq.dll', 'libcrypto-*.dll', 'libssl-*.dll',
    'pyodbc*.pyd', '_psycopg*.pyd', 'vcruntime140.dll', 'vcruntime140_1.dll',
]

_datas = [
    (str(root / 'visioninspect' / 'gui' / 'theme.qss'), 'visioninspect/gui/'),
    # contoh config — nongkrong di sebelah exe; db.txt asli JANGAN di-bundle
    (str(root / 'db.txt.example'), '.'),
]

_icon = (str(root / 'packaging' / 'icon.ico')
         if (root / 'packaging' / 'icon.ico').exists() else None)


# ── GUI ──────────────────────────────────────────────────────────────
a_gui = Analysis(
    ['run.py'],
    pathex=[str(root)],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'anomalib', 'anomalib.data', 'anomalib.models', 'anomalib.engine',
        'anomalib.deploy',
        'openvino', 'openvino.preprocess', 'nncf',
        'torch', 'torchvision', 'cv2',
        'serial',          # pyserial → modul bernama `serial`
        'flask', 'werkzeug', 'psutil', 'PIL', 'skimage',
    ] + _db_hidden,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'tests', 'docs'],
    win_no_prefer_redirects=False, win_private_assemblies=False,
    cipher=block_cipher, noarchive=False,
)

# ── CLI (konsol, ringan — tanpa Qt/torch/cv2) ───────────────────────
a_cli = Analysis(
    ['visioninspect/cli.py'],
    pathex=[str(root)],
    binaries=[],
    datas=[],
    hiddenimports=_db_hidden,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'tests', 'docs',
        'PySide6', 'shiboken6', 'torch', 'torchvision', 'anomalib',
        'cv2', 'openvino', 'nncf', 'skimage', 'PIL', 'flask', 'werkzeug',
        'pandas', 'sklearn', 'scipy', 'timm', 'huggingface_hub',
    ],
    win_no_prefer_redirects=False, win_private_assemblies=False,
    cipher=block_cipher, noarchive=False,
)

pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)

exe_gui = EXE(
    pyz_gui, a_gui.scripts, [],
    exclude_binaries=True,
    name='VisionInspect',
    debug=False, bootloader_ignore_signals=False, strip=False,
    upx=True, upx_exclude=_upx_exclude,
    console=False,                 # GUI operator — tanpa jendela konsol
    disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
    icon=_icon,
)

exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True,
    name='VisionInspect-cli',
    debug=False, bootloader_ignore_signals=False, strip=False,
    upx=True, upx_exclude=_upx_exclude,
    console=True,                  # butuh stdout: --check-db / --encrypt-secret
    disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
    icon=_icon,
)

# One-folder: kedua exe berbagi _internal/ yang sama. COLLECT menggabungkan
# TOC — file dengan tujuan sama otomatis di-dedupe.
coll = COLLECT(
    exe_gui, a_gui.binaries, a_gui.zipfiles, a_gui.datas,
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    strip=False, upx=True, upx_exclude=_upx_exclude,
    name='VisionInspect',
)
