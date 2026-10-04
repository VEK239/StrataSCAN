"""Shared, print-safe visual defaults for the OEDM/IEEE figure set."""

from __future__ import annotations

import matplotlib as mpl
import shutil
from pathlib import Path


# Okabe--Ito palette: distinct in colour and still interpretable in grayscale.
IEEE_COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000",
)
IEEE_COLUMN_WIDTH = 3.5
IEEE_TEXT_WIDTH = 7.16


def configure_ieee_style() -> None:
    """Use Times-compatible, vector-safe settings at final printed size."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5, "axes.titlesize": 10.0, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2, "lines.markersize": 4.0,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "figure.dpi": 160, "savefig.dpi": 600,
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.prop_cycle": mpl.cycler(color=IEEE_COLORS),
    })


def save_ieee_figure(fig: object, path: Path, *, dpi: int | None = None) -> None:
    """Write via a fresh sibling path, then replace a possibly locked prior export."""
    temporary = path.with_name(f".{path.stem}.rendering{path.suffix}")
    fig.savefig(temporary, format=path.suffix.lstrip("."), bbox_inches="tight", dpi=dpi)
    shutil.copyfile(temporary, path)
    temporary.unlink()


