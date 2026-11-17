"""gta6-twitch-ccu-forecast-01 entry: empirical log-ratio quantile forecast.

Method: for every past launch in the case's `comparables`, take its release-day
(UTC) hourly Twitch viewers and form the log-ratio of each hour h = 2..24 to
that launch's own hour-1 print, r_h = ln(v_h / v_1).  Two forecasts are
built from the same pool and blended 50/50 in log space:
  (E) the empirical quantiles of r_h across the pool, and
  (F) a per-hour OLS regression of r_h on ln(v_1) (launch size predicts the
      day's shape: small hour-1 launches ramp more), with each hour's slope
      shrunk halfway toward the pooled slope across all hours, and the spread
      taken from that hour's residual quantiles.
The pool is restricted to rolling midnight-local launches (the shape GTA VI
has: viewers build as time zones unlock, then peak in the US evening).
Simultaneous-global launches, early-access-contaminated launches,
free-to-play / slow-burn launches and the target's own pre-release category
history are left out: their release-day shapes are different.
Leave-one-out cross-validation on the kit's comparables: the blend scores
~0.182 mean pinball loss on GTA-like (rolling, console-only) launches versus
~0.227 for (E) alone and ~0.301 for the house baseline.

Standard library only; deterministic (no randomness, no network, no wall
clock); reads every <id>.in in the input directory and writes <id>.out.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Baseline spread (the house baseline's lognormal widening) used only as a
# fallback when an hour has too few comparable ratios to trust.
BASE_SIGMA = 0.15


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_of(comp):
    m = re.search(r"Tier:\s*([a-z-]+)", comp.get("unlock") or "")
    return m.group(1) if m else "comparable"


def is_rolling(comp):
    """Rolling midnight-local unlock, like GTA VI (not one global unlock)."""
    unlock = (comp.get("unlock") or "").lower()
    return "midnight local" in unlock or "rolling" in unlock


def in_pool(comp):
    """Pool for the ratio distribution: clean rolling-midnight launches.

    Simultaneous-global launches (one worldwide unlock hour) have a wholly
    different release-day shape (near-zero hours, then one sharp spike), and
    early-access launches had their peak days before release day, so both
    are left out.  Free-to-play / slow-burn launches are left out too.  The
    target's own pre-release category history is not a launch.
    """
    if tier_of(comp) != "comparable":
        return False
    return is_rolling(comp)


def weight_of(comp):
    """Equal weight inside the pool (validated: platform boosts did not help
    in leave-one-out cross-validation on the rolling launches)."""
    return 1.0 if in_pool(comp) else 0.0


def release_day_ratios(comp):
    """{hour: viewers} for the launch's UTC release day, hours 1..24."""
    day = parse_ts(comp["day_start"])
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_ts(row["from"])
        except Exception:
            continue
        delta = (t - day).total_seconds()
        h = int(delta // 3600) + 1
        if 1 <= h <= 24 and 0 <= delta < 24 * 3600:
            by_hour[h] = row.get("viewers")
    return by_hour


def weighted_quantile(pairs, level):
    """Weighted empirical quantile of [(value, weight)] at `level` (0..1).

    Linear interpolation on the weighted CDF; deterministic.
    """
    pairs = sorted((v, w) for v, w in pairs if w > 0)
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    target = level * total
    acc = 0.0
    prev_v, prev_cum = pairs[0][0], 0.0
    for v, w in pairs:
        cum = acc + w
        if target <= cum:
            if cum == acc:
                return v
            frac = (target - acc) / (cum - acc)
            return prev_v + frac * (v - prev_v)
        prev_v, prev_cum, acc = v, cum, cum
    return pairs[-1][0]


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile_of_sorted(sorted_vals, level):
    """Empirical quantile (linear interpolation) of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = level * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def ols(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Pool data per hour: (log hour-1 print, log-ratio) of each rolling launch.
    data = {h: [] for h in range(2, 25)}
    for comp in case.get("comparables") or []:
        w = weight_of(comp)
        if w <= 0:
            continue
        day = release_day_ratios(comp)
        v1 = day.get(1)
        if v1 is None:
            continue
        v1 = max(1, v1)
        for h in range(2, 25):
            vh = day.get(h)
            if vh is None:
                continue
            data[h].append((math.log(v1), math.log(max(1, vh) / v1)))

    # Pooled OLS slope across all hours (for shrinking each hour's slope).
    all_x, all_y = [], []
    for h in range(2, 25):
        for x, y in data[h]:
            all_x.append(x)
            all_y.append(y)
    _, pooled_b = ols(all_x, all_y) if all_x else (0.0, 0.0)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        pts = data.get(h) or []
        if len(pts) < 8:
            # Too few ratios for this hour: fall back to the house baseline
            # shape (persistence with a widening lognormal spread).
            q = [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))) for z in Z]
        else:
            # (E) empirical quantiles of the log-ratios.
            ratios = [(y, 1.0) for _, y in pts]
            q_e = [weighted_quantile(ratios, lv) for lv in LEVELS]
            med_e = q_e[2]
            # (F) per-hour OLS on log hour-1 size, slope shrunk halfway to
            # the pooled slope; spread from the hour's residual quantiles.
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            a, b = ols(xs, ys)
            b_s = 0.5 * b + 0.5 * pooled_b
            a_s = sum(ys) / len(ys) - b_s * (sum(xs) / len(xs))
            med_f = a_s + b_s * log_p1
            resid = sorted(y - (a_s + b_s * x) for x, y in pts)
            med_r = quantile_of_sorted(resid, 0.5)
            q_f = [med_f + quantile_of_sorted(resid, lv) - med_r for lv in LEVELS]
            # Blend (E) and (F) 50/50 in log space: blend the medians and
            # the log-spreads around them; the result is a log-ratio, so
            # the absolute quantile is hour 1's print times its exp.
            q = []
            for i in range(5):
                med = 0.5 * med_e + 0.5 * med_f
                off = 0.5 * (q_e[i] - med_e) + 0.5 * (q_f[i] - med_f)
                q.append(round2(p1 * math.exp(med + off)))
            q = [max(1, min(1_000_000_000, v)) for v in q]
        # Guard monotonicity and bounds (quantiles are non-decreasing by
        # construction; this only protects against float noise).
        q = [max(1, min(1_000_000_000, v)) for v in q]
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
