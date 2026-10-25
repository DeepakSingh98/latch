"""Standard-library source/wheel integrity check; does not run CUDA or training.

Run from the extracted release: python scripts/check_release.py --wheel ../megakernel_lab-0.0.0-py3-none-any.whl
"""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest((root/name).read_bytes()) == expected, f'source hash mismatch: {name}'
    package = {str(p.relative_to(root/'src')):p.read_bytes() for p in (root/'src/megakernel_lab').rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    for p in root.rglob('*.py'):
        if '.venv' not in p.parts:
            ast.parse(p.read_text(), filename=str(p))
    with zipfile.ZipFile(args.wheel) as wheel:
        names = wheel.namelist()
        assert len(names) == len(set(names)), 'duplicate wheel member'
        shipped = {name:wheel.read(name) for name in names if name.startswith('megakernel_lab/')}
        assert shipped == package, 'wheel/source runtime mismatch'
        metadata = next(n for n in names if n.endswith('.dist-info/METADATA'))
        assert f'Version: {manifest["version"]}\n' in wheel.read(metadata).decode()
        record = next(n for n in names if n.endswith('.dist-info/RECORD'))
        for name, hashed, size in csv.reader(io.StringIO(wheel.read(record).decode())):
            if not hashed:
                assert name == record
                continue
            raw = wheel.read(name)
            actual = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
            assert hashed == actual and int(size) == len(raw), f'wheel RECORD mismatch: {name}'
        for prefix in ('examples/', 'docs/', 'provenance/'):
            for path in (root/prefix).rglob('*'):
                if not path.is_file() or '__pycache__' in path.parts: continue
                relative = str(path.relative_to(root))
                name = next((n for n in names if n.endswith('.data/data/share/latch/'+relative)), None)
                assert name and wheel.read(name) == path.read_bytes(), f'wheel support file missing or changed: {relative}'
        for notice in ('THIRD_PARTY_NOTICES.md','MUON_LICENSE.txt','LLM_RANDOM_LICENSE.txt','PYTORCH_LICENSE.txt'):
            assert any(n.endswith('/'+notice) for n in names), notice
    print(json.dumps({'status':'pass','version':manifest['version'],'source_files':len(manifest['files']),
                      'runtime_files':len(package),'wheel_sha256':digest(args.wheel.read_bytes()),
                      'scope':'package integrity, resources, examples/docs and syntax; no CUDA or training execution'},indent=2))


if __name__ == '__main__':
    main()
