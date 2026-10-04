from __future__ import annotations

"""Build the biological dataset-by-method table from canonical cytometry evidence."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/published/v0.2.4/target-discovery-v1/cytometry.csv"
OUTPUT = REPO / "manuscript/oedm2026/final_figures"
STEM = "fig4_biological_validation"
METHODS = [
    "StrataSCAN",
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "SNN-DBSCAN",
    "VDBSCAN-2007",
    "kNN-DBSCAN",
    "kNN+Leiden",
]
STUDIES = ["levine", "mosmann", "nilsson", "samusik"]
HEADERS = ["Levine", "Mosmann (14 markers)", "Nilsson", "Samusik (10 samples)"]
FAST = "#0072B2"
MODERATE = "#B15B00"
SLOW = "#9B2226"
OUTLINE = "#0077B6"


def study(dataset_id: str) -> str:
    return "samusik" if dataset_id.startswith("samusik_") else dataset_id


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame["method"].isin(METHODS)].copy()
    frame["study"] = frame["dataset_id"].map(study)
    frame["ok"] = frame["status"].eq("ok")
    for column in ("macro_target_f1", "runtime_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~frame["ok"], column] = np.nan
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for current_study in STUDIES:
            part = frame.loc[
                frame["method"].eq(method) & frame["study"].eq(current_study)
            ]
            values = part.loc[part["ok"], "macro_target_f1"].dropna()
            runtimes = part.loc[part["ok"], "runtime_seconds"].dropna()
            rows.append(
                {
                    "method": method,
                    "study": current_study,
                    "attempted": int(len(part)),
                    "completed": int(part["ok"].sum()),
                    "target_f1_mean": float(values.mean()) if len(values) else np.nan,
                    "target_f1_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "runtime_median_seconds": float(runtimes.median()) if len(runtimes) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def runtime_text(seconds: float, completed: int, attempted: int) -> tuple[str, str]:
    rounded = int(round(seconds))
    if seconds < 30:
        tier, color = "fast", FAST
    elif seconds <= 120:
        tier, color = "moderate", MODERATE
    else:
        tier, color = "slow", SLOW
    completion = f" - {completed}/{attempted}" if completed != attempted else ""
    return f"{tier} - {rounded} s{completion}", color


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    data = aggregate(pd.read_csv(SOURCE, low_memory=False))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / f"{STEM}-data.csv", index=False)

    nrows, ncols = len(METHODS), len(STUDIES)
    fig, ax = plt.subplots(figsize=(7.16, 3.55))
    ax.set_xlim(-0.9, ncols)
    ax.set_ylim(-0.95, nrows + 0.55)
    ax.axis("off")
    cmap = mpl.colormaps["YlGnBu"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.60)

    # Header and method column.
    ax.text(-0.87, nrows + 0.12, "Method", ha="left", va="center", fontsize=8.4, fontweight="bold")
    for column, header in enumerate(HEADERS):
        ax.add_patch(
            Rectangle((column, nrows - 0.22), 1, 0.78, facecolor="#EEF3F5", edgecolor="#AEB7BC", lw=0.7)
        )
        ax.text(column + 0.5, nrows + 0.16, header, ha="center", va="center", fontsize=7.6, fontweight="bold")

    winners = (
        data.dropna(subset=["target_f1_mean"])
        .groupby("study", observed=True)["target_f1_mean"]
        .max()
        .to_dict()
    )
    for row_index, method in enumerate(METHODS):
        y = nrows - row_index - 1
        ax.add_patch(Rectangle((-0.9, y), 0.9, 1, facecolor="white", edgecolor="#C5CDD1", lw=0.65))
        ax.text(-0.87, y + 0.5, method, ha="left", va="center", fontsize=7.5)
        for column, current_study in enumerate(STUDIES):
            item = data.loc[
                data["method"].eq(method) & data["study"].eq(current_study)
            ].iloc[0]
            value = float(item["target_f1_mean"])
            if np.isfinite(value):
                face = cmap(norm(value))
                edge, linewidth, hatch = "#C5CDD1", 0.65, None
            else:
                face = "#EEF1F3"
                edge, linewidth, hatch = "#C8D0D5", 0.65, "///"
            ax.add_patch(
                Rectangle((column, y), 1, 1, facecolor=face, edgecolor=edge, lw=linewidth, hatch=hatch)
            )
            if not np.isfinite(value):
                ax.text(column + 0.5, y + 0.58, "not completed", ha="center", va="center", fontsize=6.3, color="#7A1F1F")
                ax.text(
                    column + 0.5,
                    y + 0.28,
                    f"{int(item['completed'])}/{int(item['attempted'])}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#7A1F1F",
                )
                continue
            is_winner = abs(value - float(winners[current_study])) <= 1e-12
            if is_winner:
                ax.add_patch(
                    Rectangle((column + 0.01, y + 0.01), 0.98, 0.98, fill=False, edgecolor=OUTLINE, lw=1.9)
                )
            if current_study == "samusik":
                score = f"{value:.3f} +/- {float(item['target_f1_sd']):.3f}"
            else:
                score = f"{value:.3f}"
            text_color = "white" if value >= 0.48 else "#0B172A"
            ax.text(
                column + 0.5,
                y + 0.63,
                score,
                ha="center",
                va="center",
                fontsize=6.8,
                color=text_color,
                fontweight="bold" if is_winner else "normal",
            )
            runtime, runtime_color = runtime_text(
                float(item["runtime_median_seconds"]),
                int(item["completed"]),
                int(item["attempted"]),
            )
            ax.text(column + 0.5, y + 0.28, runtime, ha="center", va="center", fontsize=5.7, color=runtime_color)

    ax.text(
        -0.9,
        -0.43,
        "Cell: target F1; Samusik = mean +/- SD. Runtime: median over successes. Mosmann uses type + state markers.",
        ha="left",
        va="center",
        fontsize=5.8,
    )
    ax.text(1.55, -0.77, "fast <30 s", color=FAST, ha="center", fontsize=5.7)
    ax.text(2.65, -0.77, "moderate 30-120 s", color=MODERATE, ha="center", fontsize=5.7)
    ax.text(3.65, -0.77, "slow >120 s", color=SLOW, ha="center", fontsize=5.7)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.98, bottom=0.04)
    for suffix in ("pdf", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{suffix}", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT / f"{STEM}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(data.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
