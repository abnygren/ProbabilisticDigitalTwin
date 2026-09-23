"""CuI squareness-bin probability row (S/metal vs Cu/V).

Other synthesis features are held at the cubic-training median.
Writes PNG/PDF to outputs_CuI/ and a PNG copy to figures/.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

import config as _cfg

_cfg.TRANSFER_MODE["enabled"] = False
_cfg.TRANSFER_MODE["vary_cu_precursor"] = False
_cfg.TRANSFER_MODE["vary_metal_precursor"] = False
_cfg.CURRENT_PRECURSORS["Cu_precursor"] = "CuI"
_cfg.CURRENT_PRECURSORS["Metal_Precursor"] = "VO(acac)2"
_cfg.ENHANCED_FEATURE_CONFIG["Cu_precursor_hardness"] = False
_cfg.ENHANCED_FEATURE_CONFIG["Cu_hsab_mismatch"] = False

from config import PUBLICATION_STYLE
from selfvalidating import SelfValidatingOptimizer
from visualization import plot_bin_probability_row

plt.rcParams.update(PUBLICATION_STYLE)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT_DIR = PROJECT_DIR / "outputs_CuI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    optimizer = SelfValidatingOptimizer(
        data_dir=PROJECT_DIR / "data_CuI",
        initialize_from_csv=False,
        feature_mode="synthesis",
    )
    fig = plot_bin_probability_row(optimizer)
    if fig is None:
        raise SystemExit("plot_bin_probability_row returned None")

    stem = "bin_probability_row"
    for dest in (OUTPUT_DIR, FIGURES_DIR):
        png = dest / f"{stem}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  wrote {png.relative_to(PROJECT_DIR)}")
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  wrote {pdf.relative_to(PROJECT_DIR)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
