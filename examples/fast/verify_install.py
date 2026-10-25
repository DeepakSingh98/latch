"""Lightweight clean-install check. Run with python -I; no source path is added."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import megakernel_lab.fast_examples as examples
from megakernel_lab.fast_examples import suite, cd_split

package = Path(examples.__file__).resolve().parent
assert 'site-packages' in str(package), examples.__file__
assert Path(cd_split.__file__).resolve().parent == package
assert callable(cd_split.stage_energy) and callable(cd_split.audit)
rows = {}
for family in ('mlp', 'ordinary', 'ffn', 'mixed', 'dynamic'):
    module = suite.fam(family)
    rows[family] = list(module.CASES)
for name in ('vendor.adaptive.p9c', 'vendor.energy.p9d',
             'vendor.energy.c_ops4_b.gpu_tasks', 'vendor.energy.p8_d.gpu_extra',
             'vendor.energy.p9_d.gpu_extra'):
    importlib.import_module('megakernel_lab.fast_examples.' + name)
from megakernel_lab.fast_examples.vendor.adaptive import p8c
assert p8c.policy()['schema']
from megakernel_lab.fast_examples.vendor.energy import cops4b
assert cops4b.config('sd')['chain'] > 0

# Gates remain the original imported policy bytes, independently of updated
# migrated-source hashes and the CPU/GPU observation transport.
historical = json.loads((package / 'vendor/HISTORICAL_MANIFEST.json').read_text())
policies = {}
for row in historical['files']:
    if row['path'].endswith('/POLICY.json'):
        path = package / 'vendor' / row['path']
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == row['sha256'], row['path']
        policies[row['path']] = sha
assert len(policies) == 6, policies

env = dict(os.environ)
env.pop('PYTHONPATH', None)
help_checks = []
for action in ('prepare', 'compare', 'verify', 'train', 'report'):
    text = subprocess.check_output([sys.executable, '-I', '-m', 'megakernel_lab.fast_examples',
                                    action, '--help'], env=env, text=True)
    assert 'adaptive' in text and 'energy' in text
    if action == 'verify':
        assert '--arms' in text and 'cpu' in text
    help_checks.append(action)
text = subprocess.check_output([sys.executable, '-I', '-m', 'megakernel_lab.fast_examples.cd_split',
                                '--help'], env=env, text=True)
assert 'stage-energy' in text and '--root' in text
help_checks.append('cd_split stage-energy')
print(json.dumps(dict(version=importlib.metadata.version('megakernel-lab'),
                     installed_path=examples.__file__, cases=rows,
                     unchanged_policy_sha256=policies, public_help_checked=help_checks), indent=2))
