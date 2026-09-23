"""Self-validation figures for the CuI campaign.

Read-only on data_CuI/. Writes PDF and PNG to outputs_CuI/.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm

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

from config import COLORS, PUBLICATION_STYLE
from selfvalidating import SelfValidatingOptimizer
from visualization import _extract_bias_data, _style_ax, _ANN_BOX

plt.rcParams.update(PUBLICATION_STYLE)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT_DIR = PROJECT_DIR / "outputs_CuI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ("Size", "CV", "Squareness")


def save(fig, stem: str):
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  wrote {path.relative_to(PROJECT_DIR)}")
    plt.close(fig)


def snapshot_parity(data):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.patch.set_facecolor("white")
    for ax, label in zip(axes, ORDER):
        d = data[label]
        actuals, preds = d["actuals"], d["base_preds"]
        all_vals = np.concatenate([actuals, preds])
        margin = 0.08 * (all_vals.max() - all_vals.min())
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        _style_ax(ax, title=label,
                  xlabel=f"Actual {d['axis_label']}",
                  ylabel=f"Predicted {d['axis_label']}")
        ax.plot(lims, lims, color="#444444", lw=1.4, ls="--", zorder=2)
        ax.scatter(actuals, preds, s=70, color=COLORS["primary"],
                   edgecolor="white", linewidth=0.8, zorder=4, alpha=0.9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        if label == "Size":
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.text(0.04, 0.96,
                f"$R^2$ = {d['r2_base']:.2f}\nMAE = {d['mae_base']:.3f}",
                transform=ax.transAxes, va="top", fontsize=10, bbox=_ANN_BOX)
    fig.tight_layout(pad=1.2)
    return fig


def loo_grid(data):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    fig.patch.set_facecolor("white")
    rows = [
        ("base_preds", "r2_base", "mae_base", COLORS["warning"]),
        ("corr_preds", "r2_corr", "mae_corr", COLORS["primary"]),
    ]
    for col, label in enumerate(ORDER):
        d = data[label]
        all_vals = np.concatenate([d["actuals"], d["base_preds"], d["corr_preds"]])
        margin = 0.08 * (all_vals.max() - all_vals.min())
        lims = [all_vals.min() - margin, all_vals.max() + margin]
        for row, (pkey, r2k, maek, color) in enumerate(rows):
            ax = axes[row, col]
            _style_ax(
                ax,
                title=label if row == 0 else "",
                xlabel=f"Actual {d['axis_label']}" if row == 1 else "",
                ylabel=f"Predicted {d['axis_label']}" if col == 0 else "",
            )
            ax.plot(lims, lims, color="#444444", lw=1.4, ls="--", zorder=2)
            ax.scatter(d["actuals"], d[pkey], s=70, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4, alpha=0.9)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            if label == "Size":
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.text(0.04, 0.96,
                    f"$R^2$ = {d[r2k]:.2f}\nMAE = {d[maek]:.3f}",
                    transform=ax.transAxes, va="top", fontsize=10, bbox=_ANN_BOX)
    fig.text(0.012, 0.73, "Issued snapshot", fontsize=12, fontweight="bold",
             rotation=90, va="center", color=COLORS["warning"])
    fig.text(0.012, 0.30, "LOO residual GP", fontsize=12, fontweight="bold",
             rotation=90, va="center", color=COLORS["primary"])
    fig.tight_layout(pad=1.3)
    fig.subplots_adjust(left=0.08)
    return fig


def loo_summary(data):
    labels = list(ORDER)
    r2_base = [data[k]["r2_base"] for k in labels]
    r2_corr = [data[k]["r2_corr"] for k in labels]
    mae_pct = [
        100.0 * (data[k]["mae_corr"] - data[k]["mae_base"]) / data[k]["mae_base"]
        for k in labels
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    w = 0.34

    b = ax1.bar(x - w / 2, r2_base, w, label="Issued snapshot",
                color=COLORS["warning"], edgecolor="white", linewidth=0.8, zorder=3)
    c = ax1.bar(x + w / 2, r2_corr, w, label="LOO residual GP",
                color=COLORS["primary"], edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(b, r2_base):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=COLORS["warning"])
    for bar, val in zip(c, r2_corr):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, y + (0.03 if y >= 0 else -0.03),
                 f"{val:.2f}", ha="center", va=va, fontsize=10,
                 fontweight="bold", color=COLORS["primary"])
    _style_ax(ax1, ylabel="$R^2$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax1.set_ylim(min(min(r2_base + r2_corr), 0) - 0.28, 1.18)
    ax1.legend(fontsize=10, frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

    colors = [COLORS["warning"] if v > 0 else COLORS["success"] for v in mae_pct]
    bars = ax2.bar(x, mae_pct, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, mae_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + (1.2 if val >= 0 else -1.2),
                 f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=10, fontweight="bold")
    _style_ax(ax2, ylabel="MAE change vs snapshot (%; positive = worse)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax2.set_ylim(min(mae_pct) - 8, max(mae_pct) + 10)
    fig.tight_layout(pad=1.4)
    return fig


def calibration(optimizer):
    completed = optimizer.rec_store.get_completed()
    z_by = {"Size": [], "CV": [], "Squareness": []}
    cover1, cover2 = [], []
    for prop in ORDER:
        zs, in1, in2 = [], [], []
        for rec in completed:
            err = rec.get("errors") or {}
            z = err.get(f"{prop.lower()}_z_score")
            if z is None:
                continue
            zs.append(z)
            in1.append(abs(z) < 1.0)
            in2.append(abs(z) < 1.96)
        z_by[prop] = np.array(zs)
        cover1.append(100.0 * np.mean(in1) if in1 else 0.0)
        cover2.append(100.0 * np.mean(in2) if in2 else 0.0)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    fig.patch.set_facecolor("white")

    for ax, prop in zip(axes[:3], ORDER):
        zs = z_by[prop]
        _style_ax(ax, title=prop, xlabel="z-score", ylabel="Density" if prop == "Size" else "")
        ax.hist(zs, bins=np.linspace(-3.5, 3.5, 12), density=True,
                color=COLORS["primary"], edgecolor="white", linewidth=0.6, alpha=0.85, zorder=3)
        x = np.linspace(-3.5, 3.5, 200)
        ax.plot(x, norm.pdf(x), color="#444444", lw=1.5, ls="--", zorder=4)
        ax.axvline(-1, color=COLORS["tertiary"], ls=":", lw=1.0)
        ax.axvline(1, color=COLORS["tertiary"], ls=":", lw=1.0)
        ax.set_xlim(-3.5, 3.5)
        rms = float(np.sqrt(np.mean(zs ** 2)))
        ax.text(0.04, 0.96, f"RMS $|z|$ = {rms:.2f}",
                transform=ax.transAxes, va="top", fontsize=9, bbox=_ANN_BOX)

    ax = axes[3]
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2, cover1, w, label="Within $1\\sigma$",
                color=COLORS["primary"], edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w / 2, cover2, w, label="Within $2\\sigma$",
                color=COLORS["secondary"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(68, color=COLORS["primary"], ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax.axhline(95, color=COLORS["secondary"], ls="--", lw=1.0, alpha=0.7, zorder=2)
    _style_ax(ax, title="Coverage", xlabel="", ylabel="Percent of recs")
    ax.set_xticks(x)
    ax.set_xticklabels(list(ORDER))
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9, frameon=True, framealpha=0.95, edgecolor="#CCCCCC", loc="lower right")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig, cover1, cover2, z_by


def main():
    opt = SelfValidatingOptimizer(
        data_dir=PROJECT_DIR / "data_CuI",
        initialize_from_csv=False,
        feature_mode="synthesis",
    )
    data = _extract_bias_data(opt, loo_cv=True)
    if data is None:
        raise RuntimeError("ErrorLearner not fitted; cannot build SI figures.")

    print("Generating SI ErrorLearner figures…")
    save(snapshot_parity(data), "si_snapshot_parity")
    save(loo_grid(data), "si_bias_correction_grid_loo")
    save(loo_summary(data), "si_bias_correction_summary_loo")
    fig_cal, cover1, cover2, z_by = calibration(opt)
    save(fig_cal, "si_uncertainty_calibration")

    rows = []
    for label in ORDER:
        d = data[label]
        zs = z_by[label]
        rows.append({
            "property": label,
            "n": len(d["actuals"]),
            "snapshot_R2": d["r2_base"],
            "snapshot_MAE": d["mae_base"],
            "loo_residualGP_R2": d["r2_corr"],
            "loo_residualGP_MAE": d["mae_corr"],
            "coverage_1sigma_pct": 100.0 * np.mean(np.abs(zs) < 1.0),
            "coverage_2sigma_pct": 100.0 * np.mean(np.abs(zs) < 1.96),
            "rms_z": float(np.sqrt(np.mean(zs ** 2))),
            "mean_bias": opt.error_learner.mean_bias[label],
            "calibration_factor": float(opt.error_learner.calibration_factors[label]),
        })
    csv_path = OUTPUT_DIR / "si_errorlearner_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  wrote {csv_path.relative_to(PROJECT_DIR)}")
    print("\nMetrics:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
