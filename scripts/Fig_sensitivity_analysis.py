#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction en script Python des figures du notebook :
    EGU_sensitivity_analysis.ipynb

Le script lit les CSV de sensibilite :
    Tb_core
    lambda_min
    sigma
    lambda_max
    delta_t

Il produit les figures PNG/PDF/SVG dans OUT_DIR.

Corrections incluses :
  - N_C = len(CASE_IDS) ;
  - I_H force/recalcule avec la definition corrigee :
        I_H = 1 - P10(Amax) / P90(Amax)
  - lecture de cases_one interdite par defaut ;
  - option --allow-cases-one pour autoriser explicitement cases_one ;
  - alias E5 -> IE5 et INIT_E5 -> INIT_IE5 ;
  - verification claire de la colonne case_id ;
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (pas de display X requis)
import matplotlib.pyplot as plt
import math
from netCDF4 import Dataset
from scipy.spatial import ConvexHull, QhullError
from scipy import ndimage as ndi

# =============================================================================
# Configuration par defaut
# =============================================================================

DEFAULT_DATA_PATH = Path(
    "/home/bfildier/analyses/FildierSaba2026/input/sensitivity_analysis"
)

DEFAULT_OUT_DIR = Path("../figures/tests")

# Liste des cas a tracer
CASE_IDS = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16",
    "C17", "C18", "C19",
    "C21", "C22", "C23", "C24", "C25", "C26", "C27",
]
# CASE_IDS = [
#     "C2"
# ]
N_C = len(CASE_IDS)

CMAP = plt.get_cmap("nipy_spectral")
NORM = plt.Normalize(0, max(N_C - 1, 1))


# =============================================================================
# Lecture / harmonisation des donnees
# =============================================================================

def _clean_case_id(value) -> str:
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") else s


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    out = np.full_like(num, np.nan, dtype=float)
    m = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[m] = num[m] / den[m]

    return out


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =============================================================================
# Lecture directe des CSV deja enrichis (pas de recalcul)
# =============================================================================

def load_all_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge directement les CSV "*_enriched.csv" deja calcules,
    sans recalcul d'aucune colonne derivee.

    On ignore pour l'instant lambda_max et delta_t.
    """
    data_path = args.data_path

    varids = ["Tbmin", "lambda_min", "sigma"]

    loaded = {}

    for varid in varids:
        file_name = "%s_enriched.csv" % varid
        file_path = os.path.join(data_path, file_name)

        print(f"[LECTURE] {varid}: {file_path}")

        df = pd.read_csv(file_path)

        if "case_id" in df.columns:
            df["case_id"] = df["case_id"].map(_clean_case_id)

        loaded[varid] = df

    return loaded["Tbmin"], loaded["lambda_min"], loaded["sigma"]


# =============================================================================
# Fonction du notebook : mediane + IQR inter-cas
# =============================================================================

def _get_case_x_values(data: pd.DataFrame, x_parameter: str) -> np.ndarray:
    """
    Version robuste :
    on prend l'union de toutes les valeurs du parametre, pas seulement C1.
    """
    if x_parameter not in data.columns:
        raise KeyError(
            f"Colonne x absente: {x_parameter}. "
            f"Colonnes disponibles: {list(data.columns)}"
        )

    x = np.asarray(data[x_parameter].dropna().unique(), dtype=float)
    return np.sort(x)


def _aligned_case_values(
    data: pd.DataFrame,
    case_id: str,
    x_parameter: str,
    x_ref: np.ndarray,
    y_diagnostic: str,
    y_denominator: Optional[str] = None,
    y_denominator_mean: Optional[str] = None,
) -> np.ndarray:
    if "case_id" not in data.columns:
        raise KeyError("Colonne 'case_id' absente.")

    if x_parameter not in data.columns:
        raise KeyError(f"Colonne x absente: {x_parameter}")

    if y_diagnostic not in data.columns:
        raise KeyError(
            f"Colonne diagnostic absente: {y_diagnostic}\n"
            f"Colonnes disponibles: {list(data.columns)}"
        )

    if y_denominator is not None and y_denominator not in data.columns:
        raise KeyError(f"Colonne denominateur absente: {y_denominator}")

    if y_denominator_mean is not None and y_denominator_mean not in data.columns:
        raise KeyError(f"Colonne denominateur moyen absente: {y_denominator_mean}")

    c_data = data.loc[data["case_id"] == case_id].copy()

    if c_data.empty:
        return np.full(x_ref.size, np.nan, dtype=float)

    c_data = c_data.sort_values(x_parameter)

    y = np.asarray(c_data[y_diagnostic], dtype=float)

    if y_denominator is not None:
        den = np.asarray(c_data[y_denominator], dtype=float)

        if y_denominator_mean is not None:
            den = den + 2.0 * np.asarray(c_data[y_denominator_mean], dtype=float)

        y = _safe_div(y, den)

    tmp = pd.DataFrame(
        {
            "x": np.asarray(c_data[x_parameter], dtype=float),
            "y": y,
        }
    )

    tmp = tmp.groupby("x", as_index=True)["y"].mean()

    y_aligned = tmp.reindex(x_ref).values.astype(float)

    return y_aligned


def _nanpercentile_axis0(y_all: np.ndarray, q: float) -> np.ndarray:
    """
    np.nanpercentile peut emettre des warnings si une colonne est full-NaN.
    Cette version retourne NaN proprement pour ces colonnes.
    """
    y_all = np.asarray(y_all, dtype=float)
    out = np.full(y_all.shape[1], np.nan, dtype=float)

    for j in range(y_all.shape[1]):
        col = y_all[:, j]
        col = col[np.isfinite(col)]

        if col.size > 0:
            out[j] = float(np.percentile(col, q))

    return out


# =============================================================================
# Helpers figures
# =============================================================================

def save_figure(fig, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")

    plt.close(fig)


# =============================================================================
# Focus plots pour toutes les paires IA / IE
# =============================================================================

IA_IE_SUFFIXES = [
    "1", "2", "3", "4", "5", "6",
    "11", "22", "33", "44", "55", "66",
    "111", "222", "333", "444", "555", "666",
]

PARAMETER_TITLES = {
    "sigma": r"$\sigma$",
    "lambda_min": r"$\lambda_{\min}$",
    "lambda_max": r"$\lambda_{\max}$",
    "Tbmin": r"$T_{b,\min}$",
    "delta_t": r"$\Delta t$",
    "all_parameters": "All parameters",
}


TEMPORAL_TITLE_BY_SUFFIX = {
    "1":  r"$\max_{t}$",
    "2":  r"$\max_{t}$",
    "3":  r"$\max_{t}$",
    "4":  r"$\max_{t}$",
    "5":  r"$\max_{t}$",
    "6":  r"$\max_{t}$",

    "11": r"$\sum_t$",
    "22": r"$\sum_t$",
    "33": r"$\sum_t$",
    "44": r"$\sum_t$",
    "55": r"$\sum_t$",
    "66": r"$\sum_t$",

    "111": r"$\langle\cdot\rangle_{t,A_t}$",
    "222": r"$\langle\cdot\rangle_{t,A_t}$",
    "333": r"$\langle\cdot\rangle_{t,A_t}$",
    "444": r"$\langle\cdot\rangle_{t,A_t}$",
    "555": r"$\langle\cdot\rangle_{t,A_t}$",
    "666": r"$\langle\cdot\rangle_{t,A_t}$",
}


TEMPORAL_FILENAME_BY_SUFFIX = {
    "1":  "max_t",
    "2":  "max_t",
    "3":  "max_t",
    "4":  "max_t",
    "5":  "max_t",
    "6":  "max_t",

    "11": "sum_t",
    "22": "sum_t",
    "33": "sum_t",
    "44": "sum_t",
    "55": "sum_t",
    "66": "sum_t",

    "111": "mean_t_area_weighted",
    "222": "mean_t_area_weighted",
    "333": "mean_t_area_weighted",
    "444": "mean_t_area_weighted",
    "555": "mean_t_area_weighted",
    "666": "mean_t_area_weighted",
}


OBJECT_REDUCTION_TITLE_BY_SUFFIX = {
    "1":  r"$\max_a$",
    "2":  r"$Q95_a$",
    "3":  r"$\sum_a \frac{V(a)}{\sum_{a'} V(a')}$",
    "4":  r"$\sum_a \frac{1/V(a)}{\sum_{a'} 1/V(a')}$",
    "5":  r"$\mathrm{mean}_a$",
    "6":  r"$\mathrm{median}_a$",

    "11": r"$\max_a$",
    "22": r"$Q95_a$",
    "33": r"$\sum_a \frac{V(a)}{\sum_{a'} V(a')}$",
    "44": r"$\sum_a \frac{1/V(a)}{\sum_{a'} 1/V(a')}$",
    "55": r"$\mathrm{mean}_a$",
    "66": r"$\mathrm{median}_a$",

    "111": r"$\max_a$",
    "222": r"$Q95_a$",
    "333": r"$\sum_a \frac{V(a)}{\sum_{a'} V(a')}$",
    "444": r"$\sum_a \frac{1/V(a)}{\sum_{a'} 1/V(a')}$",
    "555": r"$\mathrm{mean}_a$",
    "666": r"$\mathrm{median}_a$",
}


OBJECT_REDUCTION_FILENAME_BY_SUFFIX = {
    "1":  "max_a",
    "2":  "q95_a",
    "3":  "w_volume",
    "4":  "w_inv_volume",
    "5":  "mean_a",
    "6":  "median_a",

    "11": "max_a",
    "22": "q95_a",
    "33": "w_volume",
    "44": "w_inv_volume",
    "55": "mean_a",
    "66": "median_a",

    "111": "max_a",
    "222": "q95_a",
    "333": "w_volume",
    "444": "w_inv_volume",
    "555": "mean_a",
    "666": "median_a",
}


OBJECT_AGGREGATION_BY_SUFFIX_EN = {
    "1":  "Maximum over objects",
    "2":  "95th percentile over objects",
    "3":  "Volume-weighted mean",
    "4":  "Inverse-volume-weighted mean",
    "5":  "Mean over objects",
    "6":  "Median over objects",

    "11": "Maximum over objects",
    "22": "95th percentile over objects",
    "33": "Volume-weighted mean",
    "44": "Inverse-volume-weighted mean",
    "55": "Mean over objects",
    "66": "Median over objects",

    "111": "Maximum over objects",
    "222": "95th percentile over objects",
    "333": "Volume-weighted mean",
    "444": "Inverse-volume-weighted mean",
    "555": "Mean over objects",
    "666": "Median over objects",
}


def metric_suffix_from_name(metric: str) -> str:
    metric = str(metric)

    for prefix in ["INIT_IA", "INIT_IE", "INIT_E", "IA", "IE", "E"]:
        if metric.startswith(prefix):
            return metric[len(prefix):]

    return metric


def parameter_display_name(parameter_name: str) -> str:
    return PARAMETER_TITLES.get(parameter_name, parameter_name)


def focus_title_for_pair(parameter_name: str, ia_metric: str, ie_metric: str) -> str:
    suffix = metric_suffix_from_name(ia_metric)

    temporal_title = TEMPORAL_TITLE_BY_SUFFIX.get(suffix, "")
    reduction_title = OBJECT_REDUCTION_TITLE_BY_SUFFIX.get(suffix, "")

    return (
        rf"{temporal_title} temporelle"
        "\n"
        rf"Résumé objets : {reduction_title}"
    )


def metric_kind_from_name(metric: str) -> str:
    metric = str(metric)

    if metric.startswith("IA") or metric.startswith("INIT_IA"):
        return "adjacency"

    if metric.startswith("IE") or metric.startswith("E") or metric.startswith("INIT_IE") or metric.startswith("INIT_E"):
        return "entanglement"

    return "metric"


def metric_axis_label(metric: str, *, language: str = "en", initial: bool = False) -> str:
    suffix = metric_suffix_from_name(metric)
    temporal_op = TEMPORAL_TITLE_BY_SUFFIX.get(suffix, "")
    agg = OBJECT_AGGREGATION_BY_SUFFIX_EN.get(suffix, "")

    kind = metric_kind_from_name(metric)
    segmentation = "initial" if initial else "final"

    if kind == "adjacency":
        quantity = r"adjacency $I_A$"
    elif kind == "entanglement":
        quantity = r"entanglement $I_E$"
    else:
        quantity = "metric"

    return (
        rf"{agg}"
        "\n"
        rf"of {temporal_op} {quantity} ({segmentation})"
    )


def _is_raw_cumulative_suffix(suffix: str) -> bool:
    return suffix in ["11", "22", "33", "44", "55", "66"]


def _plot_median_iqr_curve(
    ax,
    data: pd.DataFrame,
    *,
    x_parameter: str,
    metric: str,
    label: str,
    color,
    linestyle: str = "-",
    linewidth: float = 2.0,
    fill_alpha: float = 0.12,
    show_iqr: bool = True,
) -> bool:
    """
    Trace mediane inter-cas + IQR pour une métrique.
    Retourne False si la colonne est absente ou full-NaN.
    """
    if metric not in data.columns:
        return False

    x = _get_case_x_values(data, x_parameter)
    n_param = len(x)

    y_all = np.full((N_C, n_param), np.nan, dtype=float)

    for i_c, c_name in enumerate(CASE_IDS):
        y_all[i_c, :] = _aligned_case_values(
            data,
            c_name,
            x_parameter,
            x,
            metric,
        )

    if not np.isfinite(y_all).any():
        return False

    y_25 = _nanpercentile_axis0(y_all, 25)
    y_50 = _nanpercentile_axis0(y_all, 50)
    y_75 = _nanpercentile_axis0(y_all, 75)

    if show_iqr:
        ax.fill_between(
            x,
            y_25,
            y_75,
            color=color,
            alpha=fill_alpha,
            edgecolor=None,
        )

    ax.plot(
        x,
        y_50,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    return True


def _style_axis(ax, *, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25)


def _plot_one_parameter_row_for_metric_pair(
    axs_row,
    data: pd.DataFrame,
    *,
    x_parameter: str,
    xlabel: str,
    default_x: float,
    parameter_name: str,
    ia_metric: str,
    ie_metric: str,
    n_ylim=(0, 200),
    metric_ylim=(-0.01, 1.01),
) -> bool:
    """
    Trace une ligne avec 3 colonnes :

      1. Nombre final + fraction 1 - n_i/n_f
      2. Convex hull fraction + Gini(Amax)   [colonnes fusionnees]
      3. IA/IE final + IA/IE initial si disponibles
    """

    required_cols = [
        x_parameter,
        "n_cc",
        ia_metric,
        ie_metric,
    ]

    missing = [c for c in required_cols if c not in data.columns]

    if missing:
        for ax in axs_row:
            ax.axis("off")

        axs_row[0].text(
            0.5,
            0.5,
            f"{parameter_name}\ncolonnes absentes :\n{missing}",
            ha="center",
            va="center",
            fontsize=9,
        )

        print(
            f"[SKIP] {parameter_name} {ia_metric}/{ie_metric}: "
            f"colonnes absentes: {missing}"
        )

        return False

    param_title = parameter_display_name(parameter_name)

    # ==================================================================
    # Colonne 1 : nombre + fraction 1 - n_i/n_f
    # ==================================================================
    ax = axs_row[0]
    ax.axvline(x=default_x, linewidth=1, linestyle=":", c="k")

    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric="n_cc",
        label=r"$n_f$",
        color="black",
        linestyle="-",
        show_iqr=True,
    )

    _style_axis(
        ax,
        xlabel=xlabel,
        ylabel="Number of final aggregates",
        title=rf"{param_title} : number / fraction",
    )

    if n_ylim is not None:
        ax.set_ylim(n_ylim)

    ax2 = ax.twinx()

    plotted_frac = _plot_median_iqr_curve(
        ax2,
        data,
        x_parameter=x_parameter,
        metric="fraction_after_initialization",
        label=r"$1 - n_i/n_f$",
        color="grey",
        linestyle="--",
        show_iqr=True,
    )

    if plotted_frac:
        ax2.set_ylabel("fraction of aggregates\nforming after initialization", color="grey")
        ax2.tick_params(axis="y", colors="grey")
        ax2.set_ylim((-0.01, 1.01))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    # ==================================================================
    # Colonne 2 (fusion des anciennes colonnes 2 et 3) :
    # Convex hull fraction (gauche) + Gini(Amax) (droite)
    # ==================================================================
    ax = axs_row[1]
    ax.axvline(x=default_x, linewidth=1, linestyle=":", c="k")

    plotted_hull = _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric="convex_hull_lambda_fraction",
        label=r"$\max_{t,a} A_{\mathrm{hull}} / (\pi \lambda_{\max}^2)$",
        color="purple",
        linestyle="-.",
        show_iqr=True,
    )

    _style_axis(
        ax,
        xlabel=xlabel,
        ylabel="Convex-hull fraction",
        title=rf"{param_title} : envelope / Gini",
    )

    if plotted_hull:
        ax.set_ylim((-0.01, 1.05))

    ax.set_ylabel("Convex-hull fraction", color="purple")
    ax.tick_params(axis="y", colors="purple")

    ax_gini = ax.twinx()

    gini_color = "navy"
    plotted_gini = _plot_median_iqr_curve(
        ax_gini,
        data,
        x_parameter=x_parameter,
        metric="Amax_gini",
        label=r"$Gini(A_{\max})$",
        color=gini_color,
        linestyle="-",
        show_iqr=True,
    )

    ax_gini.set_ylabel(r"$Gini(A_{\max})$", color=gini_color)
    ax_gini.tick_params(axis="y", colors=gini_color)

    if plotted_gini:
        ax_gini.set_ylim((-0.01, 1.01))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_gini.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    # ==================================================================
    # Colonne 3 : IA/IE final + IA/IE initial
    # ==================================================================
    ax = axs_row[2]
    ax.axvline(x=default_x, linewidth=1, linestyle=":", c="k")

    suffix = metric_suffix_from_name(ia_metric)

    init_ia_metric = f"INIT_IA{suffix}"
    init_ie_metric = f"INIT_IE{suffix}"

    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric=ia_metric,
        label=rf"{ia_metric} final",
        color="blue",
        linestyle="-",
        show_iqr=True,
    )

    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric=init_ia_metric,
        label=rf"{init_ia_metric} initial",
        color="royalblue",
        linestyle="--",
        show_iqr=True,
    )

    _style_axis(
        ax,
        xlabel=xlabel,
        ylabel=metric_axis_label(ia_metric, language="en", initial=False),
        title=rf"{param_title} : final / initial metrics",
    )

    ax2 = ax.twinx()

    _plot_median_iqr_curve(
        ax2,
        data,
        x_parameter=x_parameter,
        metric=ie_metric,
        label=rf"{ie_metric} final",
        color="green",
        linestyle="-",
        show_iqr=True,
    )

    _plot_median_iqr_curve(
        ax2,
        data,
        x_parameter=x_parameter,
        metric=init_ie_metric,
        label=rf"{init_ie_metric} initial",
        color="limegreen",
        linestyle="--",
        show_iqr=True,
    )

    ax.set_ylabel(metric_axis_label(ia_metric, language="en", initial=False), color="blue")
    ax.tick_params(axis="y", colors="blue")
    ax2.set_ylabel(metric_axis_label(ie_metric, language="en", initial=False), color="green")
    ax2.tick_params(axis="y", colors="green")

    if metric_ylim is not None:
        ax.set_ylim(metric_ylim)
        ax2.set_ylim(metric_ylim)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    return True


def plot_focus_grouped_by_metric_pair(
    *,
    ia_metric: str,
    ie_metric: str,
    data_sigma: pd.DataFrame,
    data_lambda_min: pd.DataFrame,
    data_Tbmin: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
):
    """
    Produit une seule figure pour une paire IA/IE.

    Lignes :
      sigma
      lambda_min
      Tbmin

    (lambda_max et delta_t sont ignores pour l'instant)

    Colonnes :
      1. Nombre + fraction 1 - n_i/n_f
      2. Convex hull fraction + Gini (colonnes fusionnees)
      3. IA/IE final + initial
    """

    suffix = metric_suffix_from_name(ia_metric)
    is_raw_cumulative = _is_raw_cumulative_suffix(suffix)

    metric_ylim = None if is_raw_cumulative else (-0.01, 1.01)

    parameter_specs = [
        dict(
            parameter_name="sigma",
            data=data_sigma,
            x_parameter="sigma_km",
            xlabel=r"$\sigma$ (km)",
            default_x=30,
            n_ylim=(0, 200),
            metric_ylim=metric_ylim,
        ),
        dict(
            parameter_name="lambda_min",
            data=data_lambda_min,
            x_parameter="lambda_min_km",
            xlabel=r"$\lambda_{\min}$ (km)",
            default_x=100,
            n_ylim=(0, 500),
            metric_ylim=metric_ylim,
        ),
        dict(
            parameter_name="Tbmin",
            data=data_Tbmin,
            x_parameter="Tb_seed_K",
            xlabel=r"$T_{b,\min}$ (K)",
            default_x=220,
            n_ylim=(0, 200),
            metric_ylim=metric_ylim,
        ),
    ]

    n_rows = len(parameter_specs)

    fig, axs = plt.subplots(
        n_rows,
        3,
        figsize=(14.5, 3.2 * n_rows),
        squeeze=False,
    )

    n_ok = 0

    for i, spec in enumerate(parameter_specs):
        ok = _plot_one_parameter_row_for_metric_pair(
            axs[i, :],
            spec["data"],
            x_parameter=spec["x_parameter"],
            xlabel=spec["xlabel"],
            default_x=spec["default_x"],
            parameter_name=spec["parameter_name"],
            ia_metric=ia_metric,
            ie_metric=ie_metric,
            n_ylim=spec["n_ylim"],
            metric_ylim=spec["metric_ylim"],
        )

        if ok:
            n_ok += 1

    fig.suptitle(
        focus_title_for_pair("all_parameters", ia_metric, ie_metric)
        + "\n"
        + rf"{ia_metric} / {ie_metric}",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    grouped_out_dir = out_dir / "by_metric"
    grouped_out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"fig_by_metric_{ia_metric}_{ie_metric}_all_parameters"

    if n_ok == 0:
        print(f"[SKIP] figure {ia_metric}/{ie_metric}: aucune ligne traçable.")
        plt.close(fig)
        return

    save_figure(fig, grouped_out_dir, stem, formats)


def plot_all_focus_IA_IE_pairs(
    *,
    data_sigma: pd.DataFrame,
    data_lambda_min: pd.DataFrame,
    data_Tbmin: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
):
    """
    Génère une figure par paire IA/IE.

    Sortie :
      OUT_DIR / by_metric / fig_by_metric_IA1_IE1_all_parameters.png
      OUT_DIR / by_metric / fig_by_metric_IA2_IE2_all_parameters.png
      ...
    """

    for suffix in IA_IE_SUFFIXES:
        ia_metric = f"IA{suffix}"
        ie_metric = f"IE{suffix}"

        plot_focus_grouped_by_metric_pair(
            ia_metric=ia_metric,
            ie_metric=ie_metric,
            data_sigma=data_sigma,
            data_lambda_min=data_lambda_min,
            data_Tbmin=data_Tbmin,
            out_dir=out_dir,
            formats=formats,
        )


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduire les figures du notebook EGU_sensitivity_analysis.ipynb."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Dossier contenant les CSV '*_enriched.csv' deja calcules.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Dossier de sortie des figures.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Formats de sortie.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    formats = args.formats

    for old_fig in out_dir.glob("fig_*.*"):
        if old_fig.suffix.lower() in [".png", ".pdf", ".svg"]:
            old_fig.unlink()

    data_Tbmin, data_lambda_min, data_sigma = load_all_data(args)

    plot_all_focus_IA_IE_pairs(
        data_sigma=data_sigma,
        data_lambda_min=data_lambda_min,
        data_Tbmin=data_Tbmin,
        out_dir=out_dir,
        formats=formats,
    )

    print(f"Figures ecrites dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()