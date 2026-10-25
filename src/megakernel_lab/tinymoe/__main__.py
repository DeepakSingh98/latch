"""Public TinyMoE workflow entry point."""
import argparse
import importlib
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('data','build','conventional','train','evaluate','benchmark','check-reference','check-grouped'))
    # Parse only the dispatch command so a subcommand's --help reaches that
    # parser rather than being consumed by this small top-level parser.
    args=p.parse_args(sys.argv[1:2])
    remaining=sys.argv[2:]
    module=importlib.import_module('megakernel_lab.tinymoe.'+args.command.replace('-','_'))
    sys.argv=[module.__name__,*remaining]
    module.main()


if __name__=='__main__':main()
