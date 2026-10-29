"""Release helpers. Fail on a failed command, smoke check, or oversized asset."""
from pathlib import Path
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

def run(*args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)

def freeze(entry, name, *extra):
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
        '--name', name, *extra, entry)
    return ROOT / 'dist' / name

def launch_check(exe, *args, timeout=120):
    # Explicit self-test must terminate successfully, not merely stay alive.
    run(exe, *args, timeout=timeout)

def server_check(exe, port, *args, timeout=90):
    process = subprocess.Popen([str(exe), *map(str, args)], cwd=exe.parent)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Server exited early: {process.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2) as r:
                    if r.status == 200 and r.read():
                        return
            except (OSError, TimeoutError):
                time.sleep(.5)
        raise TimeoutError('Frozen server did not become ready')
    finally:
        if process.poll() is None:
            # Kill the tree on Windows (Streamlit/model servers may have children).
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False)
            else:
                process.terminate()
            process.wait(timeout=30)

def archive(folder, asset):
    target = ROOT / 'release' / asset
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(folder).rglob('*')):
            if p.is_file():
                z.write(p, Path(folder).name / p.relative_to(folder))
    if target.stat().st_size >= 2_000_000_000:
        raise RuntimeError(f'{target.name} exceeds the 2 GB release ceiling')
    print(f'{target}: {target.stat().st_size:,} bytes')
    return target
