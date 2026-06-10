import argparse
import json
from pathlib import Path
from typing import Any

from rulex.analysis.rule_type_profile import profile_rule_types, write_rule_type_profile_csv
from rulex.schema.ruleframe import validate_ruleframe


def profile_file(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(input_path)
    rows = profile_rule_types(records)
    write_rule_type_profile_csv(output_path, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write per-rule-type CUAD difficulty features.")
    parser.add_argument("input", type=Path, help="Path to RuleFrame JSONL")
    parser.add_argument("output", type=Path, help="Path for profile CSV output")
    args = parser.parse_args(argv)

    rows = profile_file(args.input, args.output)
    print(f"Wrote {len(rows)} rule-type profile rows to {args.output}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(validate_ruleframe(json.loads(line)))
    return records


if __name__ == "__main__":
    raise SystemExit(main())
