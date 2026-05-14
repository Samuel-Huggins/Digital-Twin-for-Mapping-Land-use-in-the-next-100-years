from __future__ import annotations

from pathlib import Path

import pandas as pd


TRANSITION_DIR = Path("results/historical_maps/transitions")
OUT_CSV = TRANSITION_DIR / "stable_transition_probabilities_2013_2019.csv"
OUT_MATRIX_CSV = TRANSITION_DIR / "stable_transition_probability_matrix_2013_2019.csv"

STABLE_FILES = [
    TRANSITION_DIR / "transition_2013_2015.csv",
    TRANSITION_DIR / "transition_2015_2017.csv",
    TRANSITION_DIR / "transition_2017_2019.csv",
]


def main() -> int:
    frames = []

    for path in STABLE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)

    grouped = (
        df.groupby(["from_code", "from_class", "to_code", "to_class"], as_index=False)
        .agg(
            area_km2=("area_km2", "sum"),
            area_ha=("area_ha", "sum"),
            pixels=("pixels", "sum"),
        )
    )

    totals = (
        grouped.groupby(["from_code", "from_class"], as_index=False)
        .agg(total_from_area_km2=("area_km2", "sum"))
    )

    probs = grouped.merge(totals, on=["from_code", "from_class"], how="left")

    probs["transition_probability"] = (
        probs["area_km2"] / probs["total_from_area_km2"]
    )

    probs = probs.sort_values(["from_code", "transition_probability"], ascending=[True, False])

    probs.to_csv(OUT_CSV, index=False)

    matrix = probs.pivot(
        index="from_class",
        columns="to_class",
        values="transition_probability",
    ).fillna(0)

    matrix.to_csv(OUT_MATRIX_CSV)

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MATRIX_CSV}")

    print("\nStable transition probability matrix, 2013–2019:")
    print(matrix.round(3))

    print("\nTop non-stable transitions:")
    non_stable = probs[probs["from_code"] != probs["to_code"]].copy()
    print(
        non_stable.sort_values("transition_probability", ascending=False)
        .head(20)[
            [
                "from_class",
                "to_class",
                "area_km2",
                "transition_probability",
            ]
        ]
        .round(3)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())