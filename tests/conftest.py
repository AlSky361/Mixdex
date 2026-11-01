"""Use an explicit build, a local wrapper build, or an installed extension."""
import glob
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]


def has_extension(directory):
    return bool(glob.glob(str(directory / 'mixdex*.so')) or glob.glob(str(directory / 'mixdex*.pyd')))


explicit = os.environ.get('MIXDEX_BUILD_DIR')
if explicit:
    build = Path(explicit).resolve()
    if not has_extension(build):
        raise RuntimeError(f'No mixdex extension in MIXDEX_BUILD_DIR={build}')
    sys.path.insert(0, str(build))
else:
    candidates = [_ROOT / 'build', _ROOT / 'build/fair', _ROOT / 'build/windows', _ROOT / 'build/debug']
    build = next((path for path in candidates if has_extension(path)), None)
    if build is not None:
        sys.path.insert(0, str(build))
    else:
        spec = importlib.util.find_spec('mixdex')
        if spec is None or spec.origin is None or not spec.origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            raise RuntimeError('Build mixdex, set MIXDEX_BUILD_DIR, or install the package before testing.')
sys.path.insert(0, str(Path(__file__).resolve().parent))
