"""Command line: readable report plus structured JSON from existing artifacts. No GPU work.

  python -m megakernel_lab.operation_compiler.explain ARTIFACT_DIR|manifest.json [--profile FILE] [--ir FILE] [--diagnostics FILE] [--out DIR] [--stem NAME]
  python -m megakernel_lab.operation_compiler.explain moe4 --artifact artifact.json [--audit assembly-audit.json] [--measurement measurement.json]
                             [--region-manifests DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .moe4 import report_moe4
from .render import render_markdown
from .report import report_artifact, report_manifest, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m megakernel_lab.operation_compiler.explain", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("program", help="report for an artifact directory or a manifest.json")
    p.add_argument("target")
    p.add_argument("--profile", help="profile_step rows, bench native_profile dict, FFN timing.json or native_profile receipt")
    p.add_argument("--ir", help="captured IR json (TrainingIR.to_dict)")
    p.add_argument("--diagnostics", help="collector records json")
    p.add_argument("--out", help="directory for <stem>.json and <stem>.md (default: print markdown)")
    p.add_argument("--stem", default="report")
    p.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    m = sub.add_parser("moe4", help="bounded read-only report for the four-GPU MoE assembly artifact")
    m.add_argument("--artifact", required=True)
    m.add_argument("--audit")
    m.add_argument("--measurement")
    m.add_argument("--region-manifests")
    m.add_argument("--label")
    m.add_argument("--out")
    m.add_argument("--stem", default="report")
    m.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command is None:
        # allow `python -m megakernel_lab.operation_compiler.explain PATH` as shorthand for the program sub-command
        if argv is None:
            argv = sys.argv[1:]
        if argv and not argv[0].startswith("-"):
            return main(["program", *argv])
        parser.print_help()
        return 2
    if args.command == "moe4":
        report = report_moe4(args.artifact, args.audit, args.measurement, region_manifests=args.region_manifests, phase_label=args.label)
    else:
        target = Path(args.target)
        ir = json.loads(Path(args.ir).read_text()) if args.ir else None
        diagnostics = json.loads(Path(args.diagnostics).read_text()) if args.diagnostics else None
        if target.is_dir():
            report = report_artifact(target, profile=args.profile, ir=ir, diagnostics=diagnostics)
        else:
            manifest = json.loads(target.read_text())
            source_path = target.with_name("program.cu")
            report = report_manifest(manifest, source=source_path.read_text() if source_path.exists() else None,
                                     ir=ir, diagnostics=diagnostics, profile=args.profile,
                                     subject={"kind": "manifest", "path": str(target)})
    if args.out:
        json_path, md_path = write_report(report, args.out, stem=args.stem)
        print(json.dumps({"json": str(json_path), "markdown": str(md_path)}))
    elif args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
