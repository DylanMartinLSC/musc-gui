"""Only tag events publish. Manual runs build artifacts without a release."""
from pathlib import Path
import os
import subprocess

if os.environ.get('GITHUB_REF', '').startswith('refs/tags/v'):
    tag = os.environ['GITHUB_REF_NAME']
    repo = os.environ['GITHUB_REPOSITORY']
    result = subprocess.run(['gh', 'release', 'view', tag, '-R', repo], capture_output=True)
    if result.returncode:
        subprocess.run(['gh', 'release', 'create', tag, '-R', repo, '--verify-tag',
                        '--title', tag, '--generate-notes'], check=True)
    for asset in Path('release').iterdir():
        if asset.is_file():
            if asset.stat().st_size >= 2_000_000_000:
                raise RuntimeError(f'Oversized asset: {asset}')
            subprocess.run(['gh', 'release', 'upload', tag, str(asset), '-R', repo, '--clobber'], check=True)
