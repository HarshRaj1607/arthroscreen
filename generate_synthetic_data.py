"""
ArthroScreen — synthetic training data generator
==================================================

Generates fake but *plausible* per-bend feature vectors for early
pipeline development, BEFORE any real, clinically-labeled data exists.

IMPORTANT — read before you use this for anything beyond pipeline testing:
Every feature's *direction* (does OA push it up or down) is grounded in
real literature we checked. The exact numeric RANGES are NOT — with the
single exception of ROM, no paper we found reports a usable magnitude for
these features, so the ranges below are engineering judgement calls layered
on top of a real direction. See GROUNDING per feature. Do not present a
model trained on this data as "detecting OA" — it only proves your
pipeline code (features -> vector -> split -> train -> classify) runs
end-to-end. Swap this generator out the moment you have real labeled data.

Design choices, and why:
- Each synthetic PERSON gets one underlying `severity` value (0 = no OA,
  ~0..1 = mild..severe OA). All 9 features are derived from that ONE
  number, not sampled independently per feature. Real OA severity shows
  up correlated across sound, motion and pressure together — a mild case
  is mild everywhere. Sampling features independently would destroy that
  correlation and make it impossible to honestly test whether fusing all
  3 modalities beats any single one (the actual hypothesis this project
  is testing).
- Each person contributes MULTIPLE bends (reps), each with its own small
  measurement-noise on top of that person's true severity-driven baseline
  — mimicking real sensor noise / rep-to-rep variation.
- Healthy and OA ranges are made to OVERLAP on purpose (e.g. spectral
  centroid, band_energy). Real biological data is messy; if the synthetic
  classes were cleanly separable, a model trained on it would report
  ~99% accuracy that means nothing once real data shows up.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Feature specs: (healthy_range, oa_range, direction, grounding)
#   direction: "up"   -> OA pushes the value UP as severity increases
#              "down" -> OA pushes the value DOWN as severity increases
# ---------------------------------------------------------------------------
FEATURE_SPECS = {
    "band_energy": dict(
        unit="0-1 normalized", healthy=(0.05, 0.25), oa=(0.20, 0.65), direction="up",
        grounding="DIRECTION ONLY - Karpinski et al. 2025: OA knees emit more/higher-amplitude "
                   "acoustic events. No magnitude reported anywhere.",
    ),
    "spectral_centroid": dict(
        unit="Hz", healthy=(150, 400), oa=(350, 900), direction="up",
        grounding="DIRECTION ONLY - Befrui et al. 2018: OA shifts toward higher relative "
                   "frequency content. No centroid value reported.",
    ),
    "click_count": dict(
        unit="clicks/bend", healthy=(0, 2), oa=(2, 10), direction="up",
        grounding="DIRECTION ONLY - more discrete crepitus events in OA (Karpinski 2025). "
                   "No count data exists in the literature.",
    ),
    "ROM": dict(
        unit="degrees", healthy=(120, 140), oa=(70, 110), direction="down",
        grounding="SOLID - standard clinical goniometry (full flexion ~135-140 deg healthy) "
                   "+ BMC Musculoskelet Disord 2024 confirms OA sit-to-stand ROM is reduced.",
    ),
    "peak_velocity": dict(
        unit="deg/s", healthy=(250, 400), oa=(100, 220), direction="down",
        grounding="DIRECTION ONLY - 2 independent sources (BMC 2024, Research Square 2025 "
                   "preprint) confirm OA movement is slower/more guarded. No raw deg/s found.",
    ),
    "time_to_peak": dict(
        unit="ms", healthy=(400, 700), oa=(600, 1300), direction="up",
        grounding="DIRECTION ONLY - BMC 2024: OA has longer task duration. No ms figures found.",
    ),
    "peak_load": dict(
        unit="device units (uncalibrated FSR)", healthy=(700, 950), oa=(550, 900), direction="down",
        grounding="WEAK / ASSUMPTION - Wipperman et al. 2024 found a *shape* difference "
                   "(flatter mid-stance vGRF curve) in OA, not a clean peak-magnitude difference. "
                   "Treat this as your lowest-confidence, lowest-signal feature.",
    ),
    "load_rate": dict(
        unit="device units/s", healthy=(1400, 2000), oa=(900, 1600), direction="down",
        grounding="ASSUMPTION ONLY, no source - reasoning from the known 'antalgic gait' "
                   "(pain-avoidance loading) compensation pattern, not a measured figure.",
    ),
    "center_of_pressure": dict(
        unit="mm offset from center", healthy=(0, 6), oa=(6, 20), direction="up",
        grounding="DIRECTION ONLY - Chang et al. link foot CoP position to medial knee OA "
                   "loading. No shift magnitude reported.",
    ),
}

FEATURES = list(FEATURE_SPECS.keys())


def _person_baseline(spec: dict, severity: float, rng: np.random.Generator) -> float:
    """One person's TRUE underlying value for one feature, before per-bend noise."""
    h_lo, h_hi = spec["healthy"]
    oa_lo, oa_hi = spec["oa"]
    if severity <= 0:
        return rng.uniform(h_lo, h_hi)
    if spec["direction"] == "up":
        # severity 0 -> near oa_lo (mild, overlaps healthy), severity 1 -> oa_hi (severe)
        return oa_lo + severity * (oa_hi - oa_lo)
    else:
        # severity 0 -> near oa_hi (mild), severity 1 -> oa_lo (severe, most reduced)
        return oa_hi - severity * (oa_hi - oa_lo)


def generate_dataset(
    n_healthy: int = 30,
    n_oa: int = 40,
    bends_per_person: tuple = (3, 6),
    noise_frac: float = 0.06,
    severity_alpha: float = 2.0,
    severity_beta: float = 2.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    n_healthy / n_oa      : how many synthetic people in each class
    bends_per_person      : (min, max) reps per person, sampled per person
    noise_frac            : per-bend measurement noise, as a fraction of each
                             feature's (healthy_hi - healthy_lo) span
    severity_alpha/beta   : Beta distribution shape for OA severity (2,2 = most
                             people mild-to-moderate, few extreme cases either way)
    """
    rng = np.random.default_rng(seed)
    rows = []
    person_id = 0

    for label, n_people in [(0, n_healthy), (1, n_oa)]:
        for _ in range(n_people):
            person_id += 1
            severity = 0.0 if label == 0 else rng.beta(severity_alpha, severity_beta)
            baselines = {f: _person_baseline(spec, severity, rng) for f, spec in FEATURE_SPECS.items()}

            n_bends = rng.integers(bends_per_person[0], bends_per_person[1] + 1)
            for bend_i in range(1, n_bends + 1):
                row = {"person_id": person_id, "bend": bend_i, "label": label,
                       "diagnosis": "OA" if label == 1 else "Healthy",
                       "severity": round(severity, 3)}
                for f, spec in FEATURE_SPECS.items():
                    h_lo, h_hi = spec["healthy"]
                    sigma = noise_frac * (h_hi - h_lo)
                    val = baselines[f] + rng.normal(0, sigma)
                    row[f] = val
                rows.append(row)

    df = pd.DataFrame(rows)

    # physically-sane clipping / rounding
    df["click_count"] = df["click_count"].clip(lower=0).round().astype(int)
    df["ROM"] = df["ROM"].clip(lower=0, upper=150).round(1)
    df["peak_velocity"] = df["peak_velocity"].clip(lower=0).round(1)
    df["time_to_peak"] = df["time_to_peak"].clip(lower=0).round(0)
    df["peak_load"] = df["peak_load"].clip(lower=0).round(0)
    df["load_rate"] = df["load_rate"].clip(lower=0).round(0)
    df["band_energy"] = df["band_energy"].clip(lower=0, upper=1).round(3)
    df["spectral_centroid"] = df["spectral_centroid"].clip(lower=0).round(0)
    df["center_of_pressure"] = df["center_of_pressure"].round(1)

    cols = ["person_id", "bend", "diagnosis", "label", "severity"] + FEATURES
    return df[cols]


def subject_wise_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = RANDOM_SEED):
    """
    Split by PERSON, never by bend — a person's bends all land on the same
    side of the split. This is the rule that matters most in this whole
    pipeline (see our earlier conversation on data leakage).
    """
    rng = np.random.default_rng(seed)
    people = df["person_id"].unique()
    rng.shuffle(people)
    n_test = max(1, int(len(people) * test_frac))
    test_people = set(people[:n_test])
    test_df = df[df["person_id"].isin(test_people)].reset_index(drop=True)
    train_df = df[~df["person_id"].isin(test_people)].reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    df = generate_dataset(n_healthy=30, n_oa=40, bends_per_person=(3, 6))
    df.to_csv("synthetic_arthroscreen_data.csv", index=False)

    train_df, test_df = subject_wise_split(df, test_frac=0.2)
    train_df.to_csv("synthetic_train.csv", index=False)
    test_df.to_csv("synthetic_test.csv", index=False)

    print(f"Generated {len(df)} bend-rows from {df['person_id'].nunique()} people "
          f"({(df.groupby('person_id')['label'].first() == 0).sum()} healthy, "
          f"{(df.groupby('person_id')['label'].first() == 1).sum()} OA)")
    print(f"Train: {len(train_df)} rows / {train_df['person_id'].nunique()} people   "
          f"Test: {len(test_df)} rows / {test_df['person_id'].nunique()} people")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nPer-class feature means (sanity check — should differ but overlap, not separate cleanly):")
    print(df.groupby("diagnosis")[FEATURES].mean().round(2).T)
