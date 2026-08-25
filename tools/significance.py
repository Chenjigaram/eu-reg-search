"""Paired comparison of two systems over the same judgements.

Both runs process judgements in the same order within each slice, so the per-query
rows line up and can be paired. Reports a paired t-statistic and a bootstrap interval
on the mean difference, because a large relative gap on a small benchmark is exactly
the situation where an unpaired eyeball is wrong.
"""
import json, math, pathlib, random, statistics, sys

a_name, b_name = sys.argv[1], sys.argv[2]
a = json.load(open(f"reports/ablation-{a_name}.json"))
b = json.load(open(f"reports/ablation-{b_name}.json"))

for label, d in ((a_name, a), (b_name, b)):
    if not d.get("per_query"):
        sys.exit(f"{label}: per_query missing — re-run with the updated CLI")

random.seed(0)
print(f"{'slice':<16}{'n':>5}{a_name:>16}{b_name:>16}{'delta':>9}{'t':>8}{'95% CI':>22}")
print("-" * 92)

for slice_name in sorted(a["per_query"]):
    rows_a = [r[0] for r in a["per_query"][slice_name]]
    rows_b = [r[0] for r in b["per_query"][slice_name]]
    if len(rows_a) != len(rows_b):
        print(f"{slice_name}: length mismatch {len(rows_a)} vs {len(rows_b)}, skipped")
        continue
    diffs = [y - x for x, y in zip(rows_a, rows_b)]
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd else float("inf")
    boot = sorted(statistics.mean(random.choices(diffs, k=n)) for _ in range(2000))
    lo, hi = boot[int(0.025 * 2000)], boot[int(0.975 * 2000)]
    print(f"{slice_name:<16}{n:>5}{statistics.mean(rows_a):>16.4f}{statistics.mean(rows_b):>16.4f}"
          f"{mean:>+9.4f}{t:>8.2f}   [{lo:+.4f}, {hi:+.4f}]")

all_a = [r[0] for s in sorted(a["per_query"]) for r in a["per_query"][s]]
all_b = [r[0] for s in sorted(b["per_query"]) for r in b["per_query"][s]]
diffs = [y - x for x, y in zip(all_a, all_b)]
n = len(diffs)
mean = statistics.mean(diffs)
sd = statistics.stdev(diffs)
t = mean / (sd / math.sqrt(n))
boot = sorted(statistics.mean(random.choices(diffs, k=n)) for _ in range(2000))
lo, hi = boot[int(0.025 * 2000)], boot[int(0.975 * 2000)]
wins = sum(1 for d in diffs if d > 1e-9)
losses = sum(1 for d in diffs if d < -1e-9)
print("-" * 92)
print(f"{'overall':<16}{n:>5}{statistics.mean(all_a):>16.4f}{statistics.mean(all_b):>16.4f}"
      f"{mean:>+9.4f}{t:>8.2f}   [{lo:+.4f}, {hi:+.4f}]")
print(f"\nper-query: {wins} better, {losses} worse, {n - wins - losses} unchanged")
print("significant at 0.05" if lo > 0 or hi < 0 else "NOT significant: interval spans zero")
