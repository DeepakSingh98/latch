"""CPU-prepared static batch variants sharing one training and dropout state.

prepare captures/builds batches 2/4/8 on a CPU worker. run loads those exact
artifacts on H100; it cannot capture or compile a missing variant. Each batch
uses a separate static kernel. See dynamic_batch_two_layer.py for one binary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from megakernel_lab.operation_compiler import BatchVariants, SGDConfig


BATCHES = (2, 4, 8)
SEQUENCE = (2, 4, 8, 4, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 16)
        self.dropout = torch.nn.Dropout(.25)
        self.head = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.head(self.dropout(torch.tanh(self.first(x))))


def trainer(directory, device, artifacts=None):
    torch.manual_seed(1818)
    return BatchVariants(Model(), torch.nn.functional.mse_loss,
                         optimizer=SGDConfig(lr=.01, momentum=.9),
                         dropout_seed=1800, batch_sizes=BATCHES,
                         work_dir=directory / 'work', cache_dir=directory / 'cache',
                         device=device, execution='compose', prepared_artifacts=artifacts)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    manager = trainer(directory, 'cpu')
    artifacts = {batch: str(manager.prepare_artifact((torch.zeros(batch, 8),),
                                                    (torch.zeros(batch, 4),)))
                 for batch in BATCHES}
    receipt = {'schema': 'latch-batch-variants-example-v1', 'artifacts': artifacts,
               'separate_static_programs': len(artifacts), 'captures': manager.counters['captures']}
    (directory / 'example.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def run(directory):
    receipt = json.loads((directory / 'example.json').read_text())
    artifacts = {int(batch): Path(path) for batch, path in receipt['artifacts'].items()}
    manager = trainer(directory, 'cuda', artifacts)
    generator = torch.Generator(device='cuda').manual_seed(1819)
    fixtures = [((torch.randn(batch, 8, generator=generator, device='cuda') * .2,),
                 (torch.randn(batch, 4, generator=generator, device='cuda') * .1,))
                for batch in SEQUENCE]
    losses, checkpoint = [], None
    for index, fixture in enumerate(fixtures):
        loss = manager.step(*fixture)
        assert bool(torch.isfinite(loss))
        losses.append(float(loss))
        if index == 1:
            checkpoint = manager.state_dict()
    final = manager.state_dict()
    manager.load_state_dict(checkpoint)
    for fixture in fixtures[2:]:
        manager.step(*fixture)
    restored = manager.state_dict()
    for group in ('parameters', 'optimizer_state', 'state'):
        assert final[group].keys() == restored[group].keys()
        for name, value in final[group].items():
            assert torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                               restored[group][name].contiguous().reshape(-1).view(torch.uint8)), (group, name)
    assert final['updates'] == restored['updates'] == len(SEQUENCE)
    assert manager.counters['captures'] == manager.counters['prepares'] == manager.counters['native_compilations'] == 0
    assert manager.counters['loads'] == len(BATCHES)
    result = {'passed': True, 'sequence': SEQUENCE, 'losses': losses,
              'restart_suffix_bitwise': True, 'gpu_captures': 0, 'gpu_compilations': 0,
              'stats': manager.stats(), 'memory': manager.memory(),
              'scope': 'synthetic installed API/state-sharing example; no numerical or speed claim'}
    (directory / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps((prepare if args.action == 'prepare' else run)(args.directory), indent=2))


if __name__ == '__main__':
    main()
