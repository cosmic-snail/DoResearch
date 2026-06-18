"""CLI: generate cross-type holdout splits for generalization experiments.

Three holdout strategies:
  random    — randomly sample K clause types
  semantic  — hold out an entire semantic cluster (Licensing / Restrictions / Financial)
  hard      — hold out the K clause types with the lowest training answer rate
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from rulex.schema.ruleframe import validate_ruleframe

logger = logging.getLogger(__name__)

# ── Semantic clusters (41 types → 4 groups) ─────────────────────────────────
# Based on CUAD paper categories + inspection of clause semantics.

SEMANTIC_CLUSTERS: dict[str, list[str]] = {
    "licensing": [
        "License Grant",
        "Non-Transferable License",
        "Affiliate License-Licensor",
        "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Ip Ownership Assignment",
        "Joint Ip Ownership",
        "Source Code Escrow",
    ],
    "restrictions": [
        "Non-Compete",
        "Exclusivity",
        "No-Solicit Of Customers",
        "Competitive Restriction Exception",
        "No-Solicit Of Employees",
        "Non-Disparagement",
        "Most Favored Nation",
        "Rofr/Rofo/Rofn",
        "Change Of Control",
        "Anti-Assignment",
    ],
    "financial": [
        "Revenue/Profit Sharing",
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
        "Cap On Liability",
        "Liquidated Damages",
        "Uncapped Liability",
        "Insurance",
        "Audit Rights",
        "Warranty Duration",
        "Covenant Not To Sue",
    ],
    "administrative": [
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Governing Law",
        "Termination For Convenience",
        "Post-Termination Services",
        "Third Party Beneficiary",
    ],
}


def _all_types() -> list[str]:
    """Return the union of all cluster types (should be all 41)."""
    seen: set[str] = set()
    types: list[str] = []
    for cluster_types in SEMANTIC_CLUSTERS.values():
        for rt in cluster_types:
            if rt not in seen:
                seen.add(rt)
                types.append(rt)
    return types


def _answer_rate_map(records: list[dict[str, Any]]) -> dict[str, float]:
    """Return {rule_type: answer_rate} from *records*."""
    counts: dict[str, list[int]] = {}  # [total, answerable]
    for r in records:
        rt = r["rule_type"]
        if rt not in counts:
            counts[rt] = [0, 0]
        counts[rt][0] += 1
        if r["metadata"]["has_answer"]:
            counts[rt][1] += 1
    return {rt: ans / total for rt, (total, ans) in counts.items() if total > 0}


def _pick_held_out_types(
    args: argparse.Namespace,
    answer_rates: dict[str, float],
) -> tuple[list[str], str]:
    """Return (held_out_types, holdout_label) based on args."""
    all_types = sorted(answer_rates.keys())

    if args.strategy == "random":
        rng = random.Random(args.seed)
        held = sorted(rng.sample(all_types, min(args.holdout_k, len(all_types))))
        label = f"random_k{len(held)}_seed{args.seed}"

    elif args.strategy == "semantic":
        cluster = args.cluster
        if cluster not in SEMANTIC_CLUSTERS:
            raise ValueError(
                f"Unknown cluster {cluster!r}. Available: {list(SEMANTIC_CLUSTERS)}"
            )
        held = sorted(SEMANTIC_CLUSTERS[cluster])
        held = [rt for rt in held if rt in all_types]  # keep only types that exist
        label = f"semantic_{cluster}_k{len(held)}"

    elif args.strategy == "hard":
        sorted_by_rate = sorted(answer_rates.items(), key=lambda x: x[1])
        held = sorted([rt for rt, _ in sorted_by_rate[: args.holdout_k]])
        label = f"hard_k{len(held)}"

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    return held, label


def _filter_and_write(
    records: list[dict[str, Any]],
    output_path: Path,
    keep_types: set[str],
    source_label: str,
) -> int:
    """Write records whose rule_type is in *keep_types*. Return count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for r in records:
            if r["rule_type"] in keep_types:
                fh.write(json.dumps(r, ensure_ascii=True) + "\n")
                written += 1
    logger.info("Wrote %d records → %s (%s)", written, output_path, source_label)
    return written


def run_holdout_split(args: argparse.Namespace) -> dict[str, Any]:
    # Load train + test
    train_records = _read_jsonl(args.train)
    test_records = _read_jsonl(args.test)

    # Compute answer rates from train
    answer_rates = _answer_rate_map(train_records)
    logger.info("Loaded %d train + %d test records", len(train_records), len(test_records))
    logger.info("Found %d rule types with answerable records", len(answer_rates))

    held_out_types, label = _pick_held_out_types(args, answer_rates)
    all_types_set = set(answer_rates.keys())
    seen_types = sorted(all_types_set - set(held_out_types))

    logger.info("Holdout strategy: %s → label=%s", args.strategy, label)
    logger.info("Held-out types (%d): %s", len(held_out_types), ", ".join(held_out_types))
    logger.info("Seen types (%d): %s", len(seen_types), ", ".join(seen_types[:6]) + ("..." if len(seen_types) > 6 else ""))

    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write splits
    _filter_and_write(train_records, out_dir / "train_seen.jsonl", set(seen_types), "train→seen")
    _filter_and_write(train_records, out_dir / "train_heldout.jsonl", set(held_out_types), "train→heldout")
    _filter_and_write(test_records, out_dir / "test_seen.jsonl", set(seen_types), "test→seen")
    _filter_and_write(test_records, out_dir / "test_heldout.jsonl", set(held_out_types), "test→heldout")

    # Compute stats for manifest
    def _counts(path: Path) -> dict[str, int]:
        recs = _read_jsonl(path)
        ans = sum(1 for r in recs if r["metadata"]["has_answer"])
        return {"records": len(recs), "answerable": ans, "no_answer": len(recs) - ans}

    manifest = {
        "strategy": args.strategy,
        "label": label,
        "held_out_types": held_out_types,
        "seen_types": seen_types,
        "train_seen": _counts(out_dir / "train_seen.jsonl"),
        "train_heldout": _counts(out_dir / "train_heldout.jsonl"),
        "test_seen": _counts(out_dir / "test_seen.jsonl"),
        "test_heldout": _counts(out_dir / "test_heldout.jsonl"),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Manifest written → %s", manifest_path)

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return manifest


def _run_with_cached(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    answer_rates: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Like run_holdout_split but uses pre-loaded records (avoids re-reading files)."""
    held_out_types, label = _pick_held_out_types(args, answer_rates)
    all_types_set = set(answer_rates.keys())
    seen_types = sorted(all_types_set - set(held_out_types))

    logger.info("  Holdout: %s → %s", args.strategy, label)
    logger.info("  Held-out (%d): %s", len(held_out_types), ", ".join(held_out_types))

    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    _filter_and_write(train_records, out_dir / "train_seen.jsonl", set(seen_types), "train→seen")
    _filter_and_write(train_records, out_dir / "train_heldout.jsonl", set(held_out_types), "train→heldout")
    _filter_and_write(test_records, out_dir / "test_seen.jsonl", set(seen_types), "test→seen")
    _filter_and_write(test_records, out_dir / "test_heldout.jsonl", set(held_out_types), "test→heldout")

    def _counts(path: Path) -> dict[str, int]:
        recs = _read_jsonl(path)
        ans = sum(1 for r in recs if r["metadata"]["has_answer"])
        return {"records": len(recs), "answerable": ans, "no_answer": len(recs) - ans}

    manifest = {
        "strategy": args.strategy,
        "label": label,
        "held_out_types": held_out_types,
        "seen_types": seen_types,
        "train_seen": _counts(out_dir / "train_seen.jsonl"),
        "train_heldout": _counts(out_dir / "train_heldout.jsonl"),
        "test_seen": _counts(out_dir / "test_seen.jsonl"),
        "test_heldout": _counts(out_dir / "test_heldout.jsonl"),
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


# ── helpers ─────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(validate_ruleframe(json.loads(line)))
    return records


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate cross-type holdout splits for generalization experiments."
    )
    parser.add_argument("--train", type=Path, required=True, help="Training RuleFrame JSONL")
    parser.add_argument("--test", type=Path, required=True, help="Test RuleFrame JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--strategy",
        choices=["random", "semantic", "hard"],
        required=False,
        help="Holdout strategy (required unless --all)",
    )
    parser.add_argument(
        "--holdout-k",
        type=int,
        default=5,
        help="Number of types to hold out (for random/hard strategies)",
    )
    parser.add_argument(
        "--cluster",
        choices=list(SEMANTIC_CLUSTERS),
        help="Semantic cluster to hold out (for semantic strategy)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--all", action="store_true", help="Generate all preset splits")

    args = parser.parse_args(argv)

    if not args.all and not args.strategy:
        parser.error("--strategy is required when --all is not set")

    if args.all:
        # Pre-load data once to avoid re-reading for each preset
        train_records = _read_jsonl(args.train)
        test_records = _read_jsonl(args.test)
        answer_rates = _answer_rate_map(train_records)
        logger.info("Loaded %d train + %d test records for --all presets", len(train_records), len(test_records))

        presets = []
        for k in [5, 10, 15]:
            presets.append(("random", k, None, 42 + k))
            presets.append(("hard", k, None, 42))
        for cluster in ["licensing", "restrictions", "financial"]:
            presets.append(("semantic", 0, cluster, 42))

        results: list[dict[str, Any]] = []
        for strategy, k, cluster, seed in presets:
            logger.info("=== %s k=%d cluster=%s seed=%d ===", strategy, k, cluster, seed)
            args.strategy = strategy
            args.holdout_k = k
            args.cluster = cluster
            args.seed = seed
            # Use cached data directly instead of re-reading files
            results.append(_run_with_cached(train_records, test_records, answer_rates, args))
        return 0

    try:
        run_holdout_split(args)
    except Exception:
        logger.exception("Fatal error in holdout split")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
