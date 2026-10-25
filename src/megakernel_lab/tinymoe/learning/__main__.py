"""Public TinyMoE token-scheduled experiment tools."""
import argparse
import importlib
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("build", "build-ga", "init-state", "quick-windows", "corpus",
                                       "train", "paired", "observe", "audit", "evaluate-final", "analyze", "confirm"))
    a = p.parse_args(sys.argv[1:2])
    module = importlib.import_module(__package__ + "." + a.command.replace("-", "_"))
    sys.argv = [module.__name__, *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
