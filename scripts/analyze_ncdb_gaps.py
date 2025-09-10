#!/usr/bin/env python3
"""
Analyze NCDB split JSON to compute frame index gap statistics per sequence.

Usage:
  python scripts/analyze_ncdb_gaps.py \
      --split /workspace/data/ncdb-cls/splits/combined_train.json \
      [--top 15] [--show-hist] [--json-out gaps_report.json]

The script is tolerant of a malformed file where a second JSON array (e.g., only dataset_root entries)
was accidentally appended. It extracts the first well‑formed top-level array containing objects
with 'new_filename'.
"""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from collections import defaultdict, Counter


def extract_primary_array_text(raw: str) -> str:
    """Return substring of the first complete JSON array encountered in raw text.
    If standard json.loads succeeds, returns the whole raw.
    """
    raw_stripped = raw.lstrip()
    if not raw_stripped.startswith('['):
        raise ValueError('File does not start with a JSON array.')
    # Fast path
    try:
        json.loads(raw_stripped)
        return raw_stripped
    except Exception:
        pass
    depth = 0
    for i, ch in enumerate(raw_stripped):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                # Include closing bracket
                return raw_stripped[: i + 1]
    raise ValueError('Could not isolate primary JSON array.')


def load_entries(path: Path):
    raw = path.read_text(encoding='utf-8', errors='ignore')
    primary = extract_primary_array_text(raw)
    data = json.loads(primary)
    # Filter only objects with new_filename
    entries = [d for d in data if isinstance(d, dict) and 'new_filename' in d]
    return entries


def numeric_frame(stem: str):
    """Extract trailing integer, or entire digits if all digits, else None."""
    if stem.isdigit():
        return int(stem)
    m = re.search(r'(\d+)$', stem)
    if m:
        return int(m.group(1))
    return None


def analyze(entries, top_n: int, show_hist: bool):
    groups = defaultdict(list)
    for idx, e in enumerate(entries):
        groups[e.get('dataset_root', 'unknown')].append(e)

    seq_stats = []  # (dataset_root, len, min_gap, max_gap, avg_gap, med_gap)
    all_gaps = []
    gap_counter = Counter()
    largest_gap_examples = []  # (gap, dataset_root, prev_frame, next_frame)

    for root, lst in groups.items():
        # sort by numeric frame or lexicographically
        proc = []
        for e in lst:
            num = numeric_frame(e['new_filename'])
            if num is None:
                continue
            proc.append((num, e['new_filename']))
        if len(proc) < 2:
            continue
        proc.sort()
        nums = [p[0] for p in proc]
        gaps = [b - a for a, b in zip(nums, nums[1:]) if b > a]
        if not gaps:
            continue
        for a, b in zip(nums, nums[1:]):
            if b > a:
                gap = b - a
                all_gaps.append(gap)
                gap_counter[gap] += 1
                largest_gap_examples.append((gap, root, a, b))
        seq_stats.append((root, len(nums), min(gaps), max(gaps), mean(gaps), median(gaps)))

    seq_stats.sort(key=lambda x: x[3], reverse=True)  # by max_gap desc
    largest_gap_examples.sort(reverse=True, key=lambda x: x[0])

    summary = {
        'num_sequences': len(seq_stats),
        'total_entries': len(entries),
        'global_min_gap': min(all_gaps) if all_gaps else None,
        'global_max_gap': max(all_gaps) if all_gaps else None,
        'distinct_gaps_count': len(gap_counter),
        'most_common_gaps': gap_counter.most_common(20),
    }

    print('=== NCDB Gap Analysis Summary ===')
    for k, v in summary.items():
        print(f'{k}: {v}')

    print('\nTop sequences by max_gap (largest discontinuities):')
    for root, length, min_gap, max_gap, avg_gap, med_gap in seq_stats[:top_n]:
        print(f'- {root} | frames={length} | min={min_gap} max={max_gap} avg={avg_gap:.2f} med={med_gap:.2f}')

    print(f'\nLargest {top_n} individual gaps examples:')
    for gap, root, a, b in largest_gap_examples[:top_n]:
        print(f'- gap={gap:>4} | {root} | {a:08d} -> {b:08d}')

    if show_hist:
        print('\nGap histogram (sorted by gap):')
        for gap in sorted(gap_counter.keys()):
            print(f' gap {gap:>4}: {gap_counter[gap]}')

    # Recommend stride candidates: top N most common small gaps (<= global median maybe)
    if all_gaps:
        global_med = median(all_gaps)
        candidate_gaps = [g for g, c in gap_counter.most_common() if g <= global_med * 1.5][:15]
        print('\nRecommended stride candidates (frequency-biased, small gaps):')
        print(candidate_gaps)

    return {
        'summary': summary,
        'sequence_stats': seq_stats,
        'gap_hist': gap_counter,
        'largest_gap_examples': largest_gap_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, help='Path to combined split json')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--show-hist', action='store_true')
    ap.add_argument('--json-out', type=str, default='')
    args = ap.parse_args()

    entries = load_entries(Path(args.split))
    result = analyze(entries, top_n=args.top, show_hist=args.show_hist)

    if args.json_out:
        out_path = Path(args.json_out)
        serializable = {
            'summary': result['summary'],
            'sequence_stats': [
                {'dataset_root': r, 'length': l, 'min_gap': mi, 'max_gap': ma, 'avg_gap': av, 'med_gap': md}
                for (r, l, mi, ma, av, md) in result['sequence_stats']
            ],
            'largest_gap_examples': [
                {'gap': g, 'dataset_root': r, 'prev': a, 'next': b} for (g, r, a, b) in result['largest_gap_examples'][:args.top]
            ],
            'gap_hist': dict(result['gap_hist'])
        }
        out_path.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
        print(f'\nSaved JSON report to {out_path}')


if __name__ == '__main__':
    main()
