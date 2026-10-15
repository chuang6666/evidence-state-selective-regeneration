import argparse
import json
from pathlib import Path

from .replay import replay


def main():
    parser = argparse.ArgumentParser(description="Offline frozen-output reproduction. No API calls or training.")
    parser.add_argument("command", nargs="?", default="replay", choices=("replay", "prepare-references", "rescore", "tables", "statistics"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--mmqa-data", type=Path)
    parser.add_argument("--m3-data", type=Path)
    args = parser.parse_args()
    if args.command == "replay":
        result = replay(args.root)
    else:
        from .offline import rescore, main_tables, prepare_references
        if not args.output_dir:
            parser.error("--output-dir is required and must be outside the repository")
        if args.command == "prepare-references":
            if not args.mmqa_data or not args.m3_data:
                parser.error("--mmqa-data and --m3-data provider JSONL paths are required")
            result = prepare_references(args.root, args.mmqa_data, args.m3_data, args.output_dir)
        elif args.command == "rescore":
            if not args.reference_dir:
                parser.error("--reference-dir is required; reference data are deliberately not bundled")
            result = rescore(args.root, args.reference_dir, args.output_dir)
        elif args.command == "tables":
            result = main_tables(args.root, args.output_dir, args.scores_dir)
        else:
            from .statistics_run import reproduce_statistics
            result = reproduce_statistics(args.root, args.output_dir, args.scores_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
