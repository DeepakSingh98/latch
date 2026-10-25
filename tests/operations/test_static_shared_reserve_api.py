"""Explicit static shared headroom must reach lowering without changing defaults."""
from pathlib import Path

import pytest
import torch

from megakernel_lab.operation_compiler import api, runtime


@pytest.mark.parametrize('value', [False, True, -1, 1.5, '8192', None])
def test_invalid_reserve_refuses_before_capture_or_output(tmp_path, value):
    from megakernel_lab.tinymoe import build
    for call in (
        lambda: api.CapturedModule.build(None, tmp_path / 'captured', cache_dir=tmp_path,
                                        static_shared_reserve_bytes=value),
        lambda: api.prepare_module(None, tmp_path / 'module', cache_dir=tmp_path,
                                   static_shared_reserve_bytes=value),
        lambda: build.prepare('unused', tmp_path / 'tiny', tmp_path, recipe=None, phase='plain',
                              batch_size=1, sequence_length=256, static_shared_reserve_bytes=value),
    ):
        with pytest.raises(ValueError, match='nonnegative integer'):
            call()
    assert not list(tmp_path.iterdir())


def test_public_module_forwards_reserve_only_to_explicit_lowering(tmp_path, monkeypatch):
    from megakernel_lab.operation_compiler import lower as lower_module
    captured = api.capture_module(torch.nn.Linear(3, 2), lambda x: x.square().mean(), (torch.ones(2, 3),))
    real_lower = lower_module.lower
    lowering, building = [], []

    def lower(*args, **kwargs):
        lowering.append(dict(kwargs))
        return real_lower(*args, **kwargs)

    def build(program, **kwargs):
        building.append(dict(kwargs))
        library = tmp_path / 'fixture.so'
        library.write_bytes(b'fixture')
        return library

    monkeypatch.setattr(lower_module, 'lower', lower)
    monkeypatch.setattr(runtime, 'build', build)
    for name, options in [('default', {}), ('zero', {'static_shared_reserve_bytes': 0}),
                          ('reserved', {'static_shared_reserve_bytes': 8192})]:
        captured.build(tmp_path / name, cache_dir=tmp_path / 'cache',
                       execution='compose', compose_selection='common', **options)
    assert lowering[0] == lowering[1]
    assert 'static_shared_reserve_bytes' not in lowering[0]
    assert lowering[2].pop('static_shared_reserve_bytes') == 8192
    assert lowering[0] == lowering[2]
    assert building[0] == building[1] == building[2]


def test_tinymoe_cli_forwards_explicit_shared_reserve(monkeypatch):
    from megakernel_lab.tinymoe import build
    calls = []
    monkeypatch.setattr(build, 'prepare', lambda *args, **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr('sys.argv', ['build', '--model', 'model', '--out', 'artifact', '--cache-dir', 'cache',
                                   '--static-shared-reserve-bytes', '8192'])
    build.main()
    assert calls[0]['static_shared_reserve_bytes'] == 8192
