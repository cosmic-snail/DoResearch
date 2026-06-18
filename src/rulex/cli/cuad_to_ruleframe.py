import argparse
import json
from pathlib import Path
from typing import Any

from rulex.data_adapters.cuad_adapter import convert_cuad_to_ruleframes


def convert_file(input_path: Path, output_path: Path) -> int:
    cuad = json.loads(input_path.read_text(encoding="utf-8"))
    records = list(convert_cuad_to_ruleframes(cuad))
    _write_jsonl(output_path, records)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert CUAD JSON to RuleFrame JSONL.")
    parser.add_argument("input", type=Path, help="Path to CUADv1.json")
    parser.add_argument("output", type=Path, help="Path for RuleFrame JSONL output")
    args = parser.parse_args(argv)

    written = convert_file(args.input, args.output)
    print(f"Wrote {written} RuleFrame records to {args.output}")
    return 0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
