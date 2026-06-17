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
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from netCDF4 import Dataset
from scipy.spatial import ConvexHull, QhullError
from scipy import ndimage as ndi

# =============================================================================
# Configuration par defaut
# =============================================================================

DEFAULT_DATA_PATH = Path(
    "../input/sensitivity_analysis"
)

DEFAULT_OUT_DIR = Path("../figures/tests")

# Liste des cas a tracer
# CASE_IDS = [
#     "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
#     "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16",
#     "C17", "C18", "C19",
#     "C21", "C22", "C23", "C24", "C25", "C26", "C27",
# ]
CASE_IDS = [
    "C2"
]
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


def _copy_col_if_absent(df: pd.DataFrame, target: str, candidates: Iterable[str]) -> None:
    if target in df.columns:
        return

    src = _first_existing_col(df, candidates)
    if src is not None:
        df[target] = df[src]


def _compute_ih_aliases(df: pd.DataFrame, prefix: str = "") -> None:
    """
    Force le calcul de :

        I_H = 1 - P10(Amax) / P90(Amax)

    ou, si P10 absent mais P90-P10 disponible :

        I_H = (P90-P10) / P90

    prefix=""         -> IH, Amax_homogeneity
    prefix="INIT_"    -> INIT_IH, INIT_Amax_homogeneity
    prefix="initial_" -> initial_IH, initial_Amax_homogeneity
    prefix="final_"   -> final_IH, final_Amax_homogeneity
    """
    ih_col = f"{prefix}IH"
    hom_col = f"{prefix}Amax_homogeneity"

    p10_candidates = [f"{prefix}Amax_p10_km2"]
    p90_candidates = [f"{prefix}Amax_p90_km2"]
    diff_candidates = [f"{prefix}Amax_p90_minus_p10_km2"]

    # Compatibilite sigma / noms standardises.
    if prefix == "final_":
        p10_candidates += ["Amax_p10_km2"]
        p90_candidates += ["Amax_p90_km2", "final_p90_cluster_size_km2"]
        diff_candidates += ["Amax_p90_minus_p10_km2"]

    if prefix == "":
        p10_candidates += ["final_Amax_p10_km2"]
        p90_candidates += ["final_Amax_p90_km2", "final_p90_cluster_size_km2"]
        diff_candidates += ["final_Amax_p90_minus_p10_km2"]

    p10_col = _first_existing_col(df, p10_candidates)
    p90_col = _first_existing_col(df, p90_candidates)
    diff_col = _first_existing_col(df, diff_candidates)

    computed = None

    if p10_col is not None and p90_col is not None:
        computed = 1.0 - _safe_div(df[p10_col].values, df[p90_col].values)

    elif diff_col is not None and p90_col is not None:
        # P90-P10 sur P90 = 1 - P10/P90.
        computed = _safe_div(df[diff_col].values, df[p90_col].values)

    if computed is not None:
        df[ih_col] = computed
        df[hom_col] = computed
    elif ih_col in df.columns and hom_col not in df.columns:
        df[hom_col] = df[ih_col]


def harmonize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Ajoute des alias pour que les appels du notebook restent valides.
    """
    df = df.copy()

    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].map(_clean_case_id)

    # ------------------------------------------------------------------
    # Nombre de clusters : compatibilite anciennes/nouvelles colonnes
    # ------------------------------------------------------------------
    _copy_col_if_absent(df, "n_cc_final", ["n_cc", "final_n_cc", "n_final_clusters"])
    _copy_col_if_absent(df, "n_cc", ["n_cc_final", "final_n_cc", "n_final_clusters"])
    _copy_col_if_absent(df, "n_final_clusters", ["n_cc_final", "n_cc", "final_n_cc"])
    _copy_col_if_absent(df, "final_n_cc", ["n_cc_final", "n_cc", "n_final_clusters"])

    _copy_col_if_absent(df, "n_cc_initial", ["n_initial_clusters", "INIT_n_cc", "initial_n_cc"])
    _copy_col_if_absent(df, "n_initial_clusters", ["n_cc_initial", "INIT_n_cc", "initial_n_cc"])

    # ------------------------------------------------------------------
    # Amax : compatibilite sigma/final_ et noms standards
    # ------------------------------------------------------------------
    _copy_col_if_absent(df, "Amax_p10_km2", ["final_Amax_p10_km2", "final_p10_cluster_size_km2"])
    _copy_col_if_absent(df, "Amax_p90_km2", ["final_Amax_p90_km2", "final_p90_cluster_size_km2"])
    _copy_col_if_absent(df, "Amax_p90_minus_p10_km2", ["final_Amax_p90_minus_p10_km2"])

    _copy_col_if_absent(df, "final_Amax_p10_km2", ["Amax_p10_km2", "final_p10_cluster_size_km2"])
    _copy_col_if_absent(df, "final_Amax_p90_km2", ["Amax_p90_km2", "final_p90_cluster_size_km2"])
    _copy_col_if_absent(df, "final_Amax_p90_minus_p10_km2", ["Amax_p90_minus_p10_km2"])
    _copy_col_if_absent(df, "final_p90_cluster_size_km2", ["Amax_p90_km2", "final_Amax_p90_km2"])

    # ------------------------------------------------------------------
    # Initial Amax : noms INIT_* <-> initial_*
    # ------------------------------------------------------------------
    for suffix in [
        "n_cc",
        "Amax_median_km2",
        "Amax_p10_km2",
        "Amax_p25_km2",
        "Amax_p75_km2",
        "Amax_p90_km2",
        "Amax_p90_minus_p10_km2",
        "Amax_max_km2",
        "Amax_gini",
    ]:
        _copy_col_if_absent(df, f"INIT_{suffix}", [f"initial_{suffix}"])
        _copy_col_if_absent(df, f"initial_{suffix}", [f"INIT_{suffix}"])

    # ------------------------------------------------------------------
    # Nouvelles metriques E* -> anciens noms IE* attendus par le notebook
    # IMPORTANT :
    # On force IE* = E* si E* existe.
    # Sinon, si une ancienne colonne IE* existe deja dans le CSV,
    # elle pourrait etre tracee par erreur.
    # ------------------------------------------------------------------
    for m in ["1", "2", "3", "4", "5", "6", "11", "22", "33", "44", "55", "66"]:

        # Final entanglement : forcer les nouveaux E* vers les anciens noms IE*
        if f"E{m}" in df.columns:
            df[f"IE{m}"] = df[f"E{m}"]
        else:
            _copy_col_if_absent(df, f"IE{m}", [f"E{m}"])

        # Initial entanglement : forcer les nouveaux INIT_E* vers INIT_IE*
        if f"INIT_E{m}" in df.columns:
            df[f"INIT_IE{m}"] = df[f"INIT_E{m}"]
        else:
            _copy_col_if_absent(df, f"INIT_IE{m}", [f"INIT_E{m}"])

    # ------------------------------------------------------------------
    # Homogeneite / heterogeneite corrigee
    # ------------------------------------------------------------------
    _compute_ih_aliases(df, prefix="")
    _compute_ih_aliases(df, prefix="INIT_")
    _compute_ih_aliases(df, prefix="initial_")
    _compute_ih_aliases(df, prefix="final_")

    return df


def _path_is_forbidden(path: Path) -> bool:
    """
    Interdit explicitement les anciens runs dans cases_one,
    sauf si --allow-cases-one est utilise.
    """
    parts = set(path.resolve().parts)
    return "cases_one" in parts


def read_csv_required(
    path: Path,
    kind: str,
    *,
    allow_cases_one: bool = False,
) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()

    if _path_is_forbidden(path) and not allow_cases_one:
        raise RuntimeError(
            f"\nERREUR: le CSV pour {kind} est dans cases_one, donc ancien run interdit :\n"
            f"{path}\n"
            f"Utilise --allow-cases-one pour l'autoriser explicitement.\n"
        )

    if _path_is_forbidden(path) and allow_cases_one:
        print(
            f"[ATTENTION] {kind}: lecture volontaire depuis cases_one, donc ancien run :\n"
            f"            {path}"
        )

    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable pour {kind}: {path}")

    mtime = pd.Timestamp.fromtimestamp(path.stat().st_mtime)

    print(f"[LECTURE] {kind}")
    print(f"          path : {path}")
    print(f"          date : {mtime}")
    print(f"          size : {path.stat().st_size} bytes")

    df = pd.read_csv(path)
    df = harmonize_columns(df, kind=kind)

    # Diagnostic rapide : verifier les valeurs max des entanglements non cumules.
    for col in ["E1", "E2", "E3", "E4", "E5", "E6", "IE1", "IE2", "IE3", "IE4", "IE5", "IE6"]:
        if col in df.columns:
            vmax = np.nanmax(np.asarray(df[col], dtype=float))
            if np.isfinite(vmax) and vmax > 1.0001:
                print(
                    f"[WARNING] {kind}: {col} max = {vmax:.6f} > 1. "
                    "Pour une metrique non cumulee, verifier que le CSV vient bien du nouveau calcul."
                )

    if "case_id" not in df.columns:
        raise KeyError(
            f"\nERREUR: {kind} ne contient pas la colonne 'case_id'.\n"
            f"Ce script attend un CSV avec une ligne par cas et par valeur de parametre.\n"
            f"Fichier lu : {path}\n"
            f"Colonnes disponibles : {list(df.columns)}\n"
        )

    return df


def _find_csv_strict(
    *,
    cli_path: Optional[str],
    data_path: Path,
    filename: str,
    label: str,
    relative_candidates: list[str],
    allow_cases_one: bool = False,
) -> Path:
    """
    Cherche uniquement dans les chemins attendus.
    Ne fait PAS de recherche recursive.
    """
    data_path = data_path.expanduser().resolve()

    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        print(p)

        if _path_is_forbidden(p) and not allow_cases_one:
            raise RuntimeError(
                f"\nERREUR: chemin interdit pour {label}.\n"
                f"Le fichier est dans cases_one, donc ancien run :\n"
                f"{p}\n"
                f"Utilise --allow-cases-one pour l'autoriser explicitement.\n"
            )

        if not p.exists():
            raise FileNotFoundError(f"CSV introuvable pour {label}: {p}")

        if _path_is_forbidden(p) and allow_cases_one:
            print(f"[ATTENTION] {label}: chemin cases_one autorise explicitement.")

        print(f"[OK] {label}: {p}")
        return p

    checked = []

    for rel in relative_candidates:
        p = (data_path / rel).resolve()
        checked.append(p)

        if _path_is_forbidden(p) and not allow_cases_one:
            continue

        if p.exists():
            if _path_is_forbidden(p) and allow_cases_one:
                print(f"[ATTENTION] {label}: fichier trouve dans cases_one, autorise explicitement.")

            print(f"[OK] {label}: {p}")
            return p

    checked_txt = "\n".join(f"  - {p}" for p in checked)

    cases_one_txt = (
        "cases_one autorise avec --allow-cases-one."
        if allow_cases_one
        else "cases_one interdit par defaut."
    )

    raise FileNotFoundError(
        f"\nCSV introuvable pour {label}: {filename}\n\n"
        f"Chemins testes:\n"
        f"{checked_txt}\n\n"
        f"{cases_one_txt}\n\n"
        f"Commande utile pour verifier manuellement :\n"
        f"find {data_path} -name '{filename}' -print\n"
    )


def load_all_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(args.data_path).expanduser().resolve()
    allow_cases_one = bool(args.allow_cases_one)

    tbmin_csv = _find_csv_strict(
        cli_path=args.tbmin_csv,
        data_path=data_path,
        filename="Tbmin_enriched.csv",
        label="Tbmin",
        relative_candidates=[
            "cases_outputs_tbmin_sensitivity_v2/tbmin_sensitivity_sigma20km_lambdaMin_100km_lambdaMax_1500km_TbAnvil_235K/tbseed_sensitivity_all_cases_summary.csv",
            "cases_outputs_tbmin_sensitivity/tbmin_sensitivity_sigma20km_lambdaMin_100km_lambdaMax_1500km_TbAnvil_235K/tbseed_sensitivity_all_cases_summary.csv",
        ],
    )

    sigma_csv = _find_csv_strict(
        cli_path=args.sigma_csv,
        data_path=data_path,
        filename="sigma_enriched.csv",
        label="sigma",
        relative_candidates=[
            "cases_outputs_sigma_sensitivity_v2/sigma_sensitivity_lambdaMin_100km_lambdaMax_1500km/sigma_sensitivity_all_cases.csv",
            "cases_outputs_sigma_sensitivity/sigma_sensitivity_lambdaMin_100km_lambdaMax_1500km/sigma_sensitivity_all_cases.csv",
        ],
    )

    lambda_max_csv = _find_csv_strict(
        cli_path=args.lambda_max_csv,
        data_path=data_path,
        filename="lambda_max_enriched.csv",
        label="lambda_max",
        relative_candidates=[
            "cases_outputs_lambda_max_sensitivity_v2/lambda_max_sensitivity_sigma50km_lambdaMin_100km_dt_03h/lambda_max_sensitivity_all_cases.csv",
            "cases_outputs_lambda_max_sensitivity/lambda_max_sensitivity_sigma50km_lambdaMin_100km_dt_03h/lambda_max_sensitivity_all_cases.csv",
        ],
    )

    delta_t_csv = _find_csv_strict(
        cli_path=args.delta_t_csv,
        data_path=data_path,
        filename="delta_t_enriched.csv",
        label="delta_t",
        relative_candidates=[
            "cases_outputs_delta_t_sensitivity_v2/delta_t_sensitivity_sigma20km_lambdaMin_100km_lambdaMax_1500km/delta_t_sensitivity_all_cases.csv",
            "cases_outputs_delta_t_sensitivity/delta_t_sensitivity_sigma20km_lambdaMin_100km_lambdaMax_1500km/delta_t_sensitivity_all_cases.csv",
        ],
    )

    lambda_min_csv = _find_csv_strict(
        cli_path=args.lambda_min_csv,
        data_path=data_path,
        filename="lambda_min_enriched.csv",
        label="lambda_min",
        relative_candidates=[
            "cases_outputs_lambda_min_sensitivity_v2/lambda_min_sensitivity_sigma20km_lambdaMax_1500km/lambda_min_sensitivity_all_cases.csv",
            "cases_outputs_lambda_min_sensitivity/lambda_min_sensitivity_sigma20km_lambdaMax_1500km/lambda_min_sensitivity_all_cases.csv",
        ],
    )

    data_Tbmin = read_csv_required(tbmin_csv, "Tbmin")
    data_sigma = read_csv_required(sigma_csv, "sigma")
    data_lambda_max = read_csv_required(lambda_max_csv, "lambda_max")
    data_delta_t = read_csv_required(delta_t_csv, "delta_t")
    data_lambda_min = read_csv_required(
        lambda_min_csv,
        "lambda_min",
        allow_cases_one=allow_cases_one,
    )

    # ------------------------------------------------------------------
    # Colonnes derivees supplementaires :
    #   - 1 - n_i/n_f
    #   - IH25/75
    #   - max(Amax)
    #   - fraction enveloppe convexe / (pi lambda_max^2) si disponible
    # ------------------------------------------------------------------
    data_Tbmin = enrich_derived_columns(
        data_Tbmin,
        kind="Tbmin",
        x_parameter="Tb_seed_K",
    )

    data_sigma = enrich_derived_columns(
        data_sigma,
        kind="sigma",
        x_parameter="sigma_km",
    )

    data_lambda_max = enrich_derived_columns(
        data_lambda_max,
        kind="lambda_max",
        x_parameter="lambda_max_km",
    )

    data_delta_t = enrich_derived_columns(
        data_delta_t,
        kind="delta_t",
        x_parameter="delta_t_requested_h",
    )

    data_lambda_min = enrich_derived_columns(
        data_lambda_min,
        kind="lambda_min",
        x_parameter="lambda_min_km",
    )

    # ------------------------------------------------------------------
    # Ajout des métriques cumulées bornées :
    # IA111..IA666, IE111..IE666,
    # et INIT_IA111/INIT_IE111 si CSV initiaux disponibles.
    # ------------------------------------------------------------------
    data_Tbmin = add_bounded_cumulative_metrics_from_label_csvs(
        data_Tbmin,
        all_cases_csv_path=tbmin_csv,
        x_parameter="Tb_seed_K",
        kind="Tbmin",
    )

    data_sigma = add_bounded_cumulative_metrics_from_label_csvs(
        data_sigma,
        all_cases_csv_path=sigma_csv,
        x_parameter="sigma_km",
        kind="sigma",
    )

    data_lambda_max = add_bounded_cumulative_metrics_from_label_csvs(
        data_lambda_max,
        all_cases_csv_path=lambda_max_csv,
        x_parameter="lambda_max_km",
        kind="lambda_max",
    )

    data_delta_t = add_bounded_cumulative_metrics_from_label_csvs(
        data_delta_t,
        all_cases_csv_path=delta_t_csv,
        x_parameter="delta_t_requested_h",
        kind="delta_t",
    )

    data_lambda_min = add_bounded_cumulative_metrics_from_label_csvs(
        data_lambda_min,
        all_cases_csv_path=lambda_min_csv,
        x_parameter="lambda_min_km",
        kind="lambda_min",
    )

    return data_Tbmin, data_lambda_min, data_sigma, data_lambda_max, data_delta_t

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


def subplotAllData(
    ax,
    data: pd.DataFrame,
    x_parameter: str,
    xlabel: str,
    y_diagnostic: str,
    ylabel: str,
    y_denominator: Optional[str] = None,
    y_denominator_mean: Optional[str] = None,
    col_diag="k",
    col_ylabel="k",
    one_minus: bool = False,
    exp_transform: bool = False,
    yscale: str = "linear",
    show_all_cases: bool = True,
    **kwargs,
):
    x = _get_case_x_values(data, x_parameter)
    n_param = len(x)

    y_all = np.full((N_C, n_param), np.nan, dtype=float)

    for i_c, c_name in enumerate(CASE_IDS):
        y = _aligned_case_values(
            data,
            c_name,
            x_parameter,
            x,
            y_diagnostic,
            y_denominator=y_denominator,
            y_denominator_mean=y_denominator_mean,
        )

        if one_minus:
            y = 1.0 - y
        elif exp_transform:
            y = 1.0 - np.exp(1.0 - y)

        y_all[i_c, :] = y

        if show_all_cases and np.isfinite(y).any():
            ax.plot(
                x,
                y,
                c=CMAP(NORM(i_c)),
                linewidth=1,
                alpha=0.5,
                **kwargs,
            )

    y_25 = _nanpercentile_axis0(y_all, 25)
    y_50 = _nanpercentile_axis0(y_all, 50)
    y_75 = _nanpercentile_axis0(y_all, 75)

    ax.fill_between(x, y_25, y_75, color=col_diag, alpha=0.5, edgecolor=None)
    ax.plot(x, y_50, color=col_diag, linewidth=2, alpha=1, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, c=col_ylabel)
    ax.set_yscale(yscale)


# =============================================================================
# Helpers figures
# =============================================================================

def save_figure(fig, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")

    plt.close(fig)


def heterogeneity_kwargs(
    df: pd.DataFrame,
    diagnostic_prefix: str = "",
    legacy: bool = False,
):
    """
    Retourne les arguments pour tracer l'homogeneite / heterogeneite.

    legacy=False :
        utilise I_H = 1 - P10/P90 si disponible/calcule.

    legacy=True :
        reproduit l'ancienne logique du notebook :
            1 - (P90-P10)/P90 = P10/P90
        donc l'inverse de la nouvelle heterogeneite.
    """
    if not legacy:
        if diagnostic_prefix == "final_" and "final_IH" in df.columns:
            return dict(y_diagnostic="final_IH", y_denominator=None, one_minus=False)

        if diagnostic_prefix == "INIT_" and "INIT_IH" in df.columns:
            return dict(y_diagnostic="INIT_IH", y_denominator=None, one_minus=False)

        if diagnostic_prefix == "initial_" and "initial_IH" in df.columns:
            return dict(y_diagnostic="initial_IH", y_denominator=None, one_minus=False)

        if "IH" in df.columns:
            return dict(y_diagnostic="IH", y_denominator=None, one_minus=False)

    # Fallback legacy.
    if diagnostic_prefix == "final_":
        return dict(
            y_diagnostic="final_Amax_p90_minus_p10_km2",
            y_denominator="final_Amax_p90_km2",
            one_minus=True,
        )

    if diagnostic_prefix == "INIT_":
        return dict(
            y_diagnostic="INIT_Amax_p90_minus_p10_km2",
            y_denominator="INIT_Amax_p90_km2",
            one_minus=True,
        )

    if diagnostic_prefix == "initial_":
        return dict(
            y_diagnostic="initial_Amax_p90_minus_p10_km2",
            y_denominator="initial_Amax_p90_km2",
            one_minus=True,
        )

    return dict(
        y_diagnostic="Amax_p90_minus_p10_km2",
        y_denominator="Amax_p90_km2",
        one_minus=True,
    )

# =============================================================================
# Colonnes derivees supplementaires : fraction initiale, IH25/75,
# max(Amax), fraction enveloppe convexe / (pi lambda_max^2)
# =============================================================================

FIXED_LAMBDA_MAX_BY_KIND = {
    "Tbmin": 1500.0,
    "sigma": 1500.0,
    "lambda_min": 1500.0,
    "delta_t": 1500.0,
    # lambda_max est variable : on prend la colonne lambda_max_km.
}


CONVEX_HULL_AREA_CANDIDATES = [
    # Noms possibles si cette metrique est deja sauvegardé dans CSV.
    "convex_hull_Amax_max_km2",
    "convex_hull_max_km2",
    "Aconvex_max_km2",
    "Aconv_max_km2",
    "A_hull_max_km2",
    "hull_Amax_max_km2",
    "max_hull_area_km2",
    "max_convex_hull_km2",
    "max_convex_envelope_km2",
    "final_convex_hull_Amax_max_km2",
    "final_convex_hull_max_km2",
    "final_Aconvex_max_km2",
    "final_max_hull_area_km2",
    "final_max_convex_hull_km2",
]


def _copy_common_amax_aliases(df: pd.DataFrame) -> None:
    """
    Complete les alias Amax finaux utiles pour les nouveaux plots.
    """
    _copy_col_if_absent(df, "Amax_p25_km2", ["final_Amax_p25_km2"])
    _copy_col_if_absent(df, "Amax_p75_km2", ["final_Amax_p75_km2"])
    _copy_col_if_absent(df, "Amax_max_km2", ["final_Amax_max_km2"])

    _copy_col_if_absent(df, "final_Amax_p25_km2", ["Amax_p25_km2"])
    _copy_col_if_absent(df, "final_Amax_p75_km2", ["Amax_p75_km2"])
    _copy_col_if_absent(df, "final_Amax_max_km2", ["Amax_max_km2"])

    _copy_col_if_absent(df, "final_Amax_gini", ["Amax_gini"])
    _copy_col_if_absent(df, "Amax_gini", ["final_Amax_gini"])


def _compute_fraction_after_initialization(df: pd.DataFrame) -> None:
    """
    Calcule :

        fraction_after_initialization = 1 - n_i / n_f

    avec :
      n_i = nombre de clusters initiaux
      n_f = nombre de clusters finaux
    """
    if "n_cc_initial" in df.columns and "n_cc_final" in df.columns:
        df["fraction_after_initialization"] = (
            1.0 - _safe_div(df["n_cc_initial"].values, df["n_cc_final"].values)
        )

    elif "n_initial_clusters" in df.columns and "n_cc" in df.columns:
        df["fraction_after_initialization"] = (
            1.0 - _safe_div(df["n_initial_clusters"].values, df["n_cc"].values)
        )


def _compute_ih25_75_aliases(df: pd.DataFrame, prefix: str = "") -> None:
    """
    Calcule :

        I_H^{25/75} = 1 - P25(Amax) / P75(Amax)

    prefix=""         -> IH25_75, Amax_homogeneity_25_75
    prefix="INIT_"    -> INIT_IH25_75, INIT_Amax_homogeneity_25_75
    prefix="initial_" -> initial_IH25_75, initial_Amax_homogeneity_25_75
    prefix="final_"   -> final_IH25_75, final_Amax_homogeneity_25_75
    """
    ih_col = f"{prefix}IH25_75"
    hom_col = f"{prefix}Amax_homogeneity_25_75"

    p25_candidates = [f"{prefix}Amax_p25_km2"]
    p75_candidates = [f"{prefix}Amax_p75_km2"]

    if prefix == "final_":
        p25_candidates += ["Amax_p25_km2"]
        p75_candidates += ["Amax_p75_km2"]

    if prefix == "":
        p25_candidates += ["final_Amax_p25_km2"]
        p75_candidates += ["final_Amax_p75_km2"]

    p25_col = _first_existing_col(df, p25_candidates)
    p75_col = _first_existing_col(df, p75_candidates)

    if p25_col is not None and p75_col is not None:
        computed = 1.0 - _safe_div(df[p25_col].values, df[p75_col].values)
        df[ih_col] = computed
        df[hom_col] = computed


def _infer_lambda_max_for_normalization(
    df: pd.DataFrame,
    *,
    kind: str,
    x_parameter: str,
) -> np.ndarray:
    """
    Retourne lambda_max ligne par ligne pour normaliser :

        A_hull / (pi lambda_max^2)

    Pour lambda_max sensitivity : lambda_max varie.
    Pour les autres sensibilites : lambda_max est fixe a 1500 km.
    """
    if x_parameter == "lambda_max_km" and "lambda_max_km" in df.columns:
        return np.asarray(df["lambda_max_km"], dtype=float)

    if "lambda_max_km" in df.columns and kind == "lambda_max":
        return np.asarray(df["lambda_max_km"], dtype=float)

    fixed = FIXED_LAMBDA_MAX_BY_KIND.get(kind, np.nan)
    return np.full(len(df), fixed, dtype=float)


def _compute_convex_hull_lambda_fraction(
    df: pd.DataFrame,
    *,
    kind: str,
    x_parameter: str,
) -> None:
    """
    Calcule si possible :

        max_{t,a} A_hull(a,t) / (pi lambda_max^2)

    IMPORTANT :
    cette metrique ne peut etre tracee que si une colonne d'aire
    d'enveloppe convexe existe deja dans le CSV global.

    Si aucune colonne candidate n'existe, on cree une colonne NaN.
    """
    hull_col = _first_existing_col(df, CONVEX_HULL_AREA_CANDIDATES)

    out_col = "convex_hull_lambda_fraction"

    if hull_col is None:
        df[out_col] = np.nan
        print(
            f"[INFO] {kind}: aucune colonne d'enveloppe convexe trouvee. "
            f"{out_col} restera NaN."
        )
        return

    lambda_max_km = _infer_lambda_max_for_normalization(
        df,
        kind=kind,
        x_parameter=x_parameter,
    )

    denom = np.pi * lambda_max_km**2
    df[out_col] = _safe_div(df[hull_col].values, denom)

    print(
        f"[OK] {kind}: {out_col} calculee depuis la colonne {hull_col}."
    )


def enrich_derived_columns(
    df: pd.DataFrame,
    *,
    kind: str,
    x_parameter: str,
) -> pd.DataFrame:
    """
    Ajoute toutes les colonnes derivees utiles aux figures enrichies.
    """
    df = df.copy()

    _copy_common_amax_aliases(df)

    _compute_fraction_after_initialization(df)

    _compute_ih25_75_aliases(df, prefix="")
    _compute_ih25_75_aliases(df, prefix="final_")
    _compute_ih25_75_aliases(df, prefix="INIT_")
    _compute_ih25_75_aliases(df, prefix="initial_")

    _compute_convex_hull_lambda_fraction(
        df,
        kind=kind,
        x_parameter=x_parameter,
    )

    return df

# =============================================================================
# Ajout des métriques cumulées bornées IA111..IA666 / IE111..IE666
# depuis les CSV par label
# =============================================================================

BOUNDED_CUM_SUFFIXES = ["111", "222", "333", "444", "555", "666"]

BOUNDED_CUM_MAP = {
    "111": "m1",  # max_a
    "222": "m2",  # Q95_a
    "333": "m3",  # pondération volume
    "444": "m4",  # pondération 1/volume
    "555": "m5",  # mean_a
    "666": "m6",  # median_a
}


def _summarize_label_metric(values, volumes) -> dict[str, float]:
    """
    Résume une métrique label par label avec les mêmes agrégations
    que IA1..IA6 / IA11..IA66.

    m1 = max
    m2 = Q95
    m3 = moyenne pondérée par V
    m4 = moyenne pondérée par 1/V
    m5 = moyenne
    m6 = médiane
    """
    out = {k: np.nan for k in ("m1", "m2", "m3", "m4", "m5", "m6")}

    x = np.asarray(values, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(v) & (v > 0.0)

    if not m.any():
        return out

    x = x[m]
    v = v[m]

    out["m1"] = float(np.nanmax(x))
    out["m2"] = float(np.nanpercentile(x, 95))

    if np.isfinite(v.sum()) and v.sum() > 0.0:
        w = v / v.sum()
        out["m3"] = float(np.nansum(w * x))

    wi = 1.0 / v
    if np.isfinite(wi.sum()) and wi.sum() > 0.0:
        wi = wi / wi.sum()
        out["m4"] = float(np.nansum(wi * x))

    out["m5"] = float(np.nanmean(x))
    out["m6"] = float(np.nanmedian(x))

    return out


def _bounded_time_mean_values_from_label_df(
    df_label: pd.DataFrame,
    *,
    raw_sum_col: str,
    old_sum_norm_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule, label par label :

        bounded_sum(a) = sum_t A_t(a) I(a,t) / sum_t A_t(a)

    Dans les CSV label :
      volume_vox       = sum_t A_t(a)
      raw_sum_col      = sum_t A_t(a) I(a,t)

    Si raw_sum_col est absent mais old_sum_norm_col et Amax_px existent :
      old_sum_norm_col = sum_t A_t(a) I(a,t) / Amax(a)
      donc :
      raw_sum = old_sum_norm_col * Amax_px
    """
    if "volume_vox" not in df_label.columns:
        raise KeyError("Colonne absente dans CSV label : volume_vox")

    vol = np.asarray(df_label["volume_vox"], dtype=np.float64)

    if raw_sum_col in df_label.columns:
        raw_sum = np.asarray(df_label[raw_sum_col], dtype=np.float64)

    elif old_sum_norm_col in df_label.columns and "Amax_px" in df_label.columns:
        old_sum_norm = np.asarray(df_label[old_sum_norm_col], dtype=np.float64)
        amax = np.asarray(df_label["Amax_px"], dtype=np.float64)
        raw_sum = old_sum_norm * amax

    else:
        raise KeyError(
            f"Impossible de calculer la métrique bornée. "
            f"Colonnes attendues : {raw_sum_col} ou ({old_sum_norm_col} + Amax_px). "
            f"Colonnes disponibles : {list(df_label.columns)}"
        )

    values = _safe_div(raw_sum, vol)

    return values, vol

# =============================================================================
# Enveloppe convexe normalisée :
# max_{t,a} |H_t(a)| / (pi lambda_max^2)
# =============================================================================

R_EARTH_KM = 6371.0088

FIXED_LAMBDA_MAX_BY_KIND = {
    "Tbmin": 1500.0,
    "sigma": 1500.0,
    "lambda_min": 1500.0,
    "delta_t": 1500.0,
    # lambda_max varie avec la colonne lambda_max_km.
}


def haversine_km_vec(lat0, lon0, lat1, lon1):
    lat0r = np.deg2rad(np.asarray(lat0, dtype="f8"))
    lat1r = np.deg2rad(np.asarray(lat1, dtype="f8"))

    dphi = lat1r - lat0r
    dlmb = np.deg2rad(
        (
            np.asarray(lon1, dtype="f8")
            - np.asarray(lon0, dtype="f8")
            + 180.0
        )
        % 360.0
        - 180.0
    )

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(lat0r) * np.cos(lat1r) * np.sin(dlmb / 2.0) ** 2
    )

    return (
        2.0
        * R_EARTH_KM
        * np.arctan2(
            np.sqrt(np.clip(a, 0.0, 1.0)),
            np.sqrt(np.clip(1.0 - a, 0.0, 1.0)),
        )
    )


def infer_xy_km_from_coords(latitudes, longitudes):
    lat = np.asarray(latitudes, dtype="f8")
    lon = np.asarray(longitudes, dtype="f8")

    # Cas lat/lon 2D : on essaie d'extraire un axe latitude et un axe longitude.
    if lat.ndim == 2:
        lat = lat[:, 0]

    if lon.ndim == 2:
        lon = lon[0, :]

    lat = lat.ravel()
    lon = lon.ravel()

    fallback = 0.04 * 111.32

    if lat.size < 2 or lon.size < 2:
        return float(fallback), float(fallback)

    lat_mean = float(np.nanmean(lat))
    lon_mean = float(np.nanmean(lon))

    dy = haversine_km_vec(lat[0], lon_mean, lat[1], lon_mean)
    dx = haversine_km_vec(lat_mean, lon[0], lat_mean, lon[1])

    if not (np.isfinite(dx) and dx > 0.0):
        dx = fallback * max(0.05, math.cos(math.radians(lat_mean)))

    if not (np.isfinite(dy) and dy > 0.0):
        dy = fallback

    return float(dx), float(dy)


def choose_label_var_from_netcdf(nc: Dataset) -> Optional[str]:
    preferred = [
        "labels_3d_algo_reconstitution",
        "labels_3d_global_recon",
        "labels_3d_global_iter",
        "labels",
        "label",
    ]

    for name in preferred:
        if name in nc.variables:
            v = nc.variables[name]
            if v.ndim == 3 and str(v.dtype).startswith(("i", "u")):
                return name

    for name, v in nc.variables.items():
        if v.ndim == 3 and str(v.dtype).startswith(("i", "u")):
            return name

    return None


def _read_lat_lon_from_labels_nc(nc: Dataset):
    lat_candidates = ["latitude", "lat", "Latitude", "LAT"]
    lon_candidates = ["longitude", "lon", "Longitude", "LON"]

    lat_name = None
    lon_name = None

    for name in lat_candidates:
        if name in nc.variables:
            lat_name = name
            break

    for name in lon_candidates:
        if name in nc.variables:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        raise KeyError(
            "Impossible de trouver les coordonnées latitude/longitude "
            "dans CCS_final_labels.nc."
        )

    latitudes = np.asarray(nc.variables[lat_name][:], dtype=np.float64)
    longitudes = np.asarray(nc.variables[lon_name][:], dtype=np.float64)

    return latitudes, longitudes


def _signed_polygon_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return float(
        0.5
        * (
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )


def _polygon_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0

    return float(abs(_signed_polygon_area(poly)))


def _convex_hull_polygon(points):
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)

    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if points.shape[0] < 3:
        xmn, ymn = points.min(axis=0)
        xmx, ymx = points.max(axis=0)

        poly = np.array(
            [
                [xmn, ymn],
                [xmx, ymn],
                [xmx, ymx],
                [xmn, ymx],
            ],
            dtype=np.float64,
        )

        return poly[::-1] if _signed_polygon_area(poly) < 0.0 else poly

    try:
        hull = ConvexHull(points, qhull_options="QJ")
        poly = points[hull.vertices]

        return poly[::-1] if _signed_polygon_area(poly) < 0.0 else poly

    except QhullError:
        xmn, ymn = points.min(axis=0)
        xmx, ymx = points.max(axis=0)

        poly = np.array(
            [
                [xmn, ymn],
                [xmx, ymn],
                [xmx, ymx],
                [xmn, ymx],
            ],
            dtype=np.float64,
        )

        return poly[::-1] if _signed_polygon_area(poly) < 0.0 else poly



def _lambda_max_for_row(row: pd.Series, *, kind: str, x_parameter: str) -> float:
    """
    Retourne lambda_max pour normaliser par pi lambda_max^2.

    - Pour lambda_max sensitivity : lambda_max est la valeur de la ligne.
    - Pour les autres sensibilités : lambda_max fixe = 1500 km.
    """
    if x_parameter == "lambda_max_km" and "lambda_max_km" in row.index:
        return float(row["lambda_max_km"])

    if kind == "lambda_max" and "lambda_max_km" in row.index:
        return float(row["lambda_max_km"])

    return float(FIXED_LAMBDA_MAX_BY_KIND.get(kind, 1500.0))


def _convex_hull_area_from_boundary_pixels(
    *,
    ys_global: np.ndarray,
    xs_global: np.ndarray,
    dy_km: float,
    dx_km: float,
) -> float:
    """
    Calcule l'aire de l'enveloppe convexe à partir des pixels de bord seulement.

    Les pixels intérieurs n'influencent pas l'enveloppe convexe, donc cela évite
    de construire un très grand nuage de points.
    """
    ys_global = np.asarray(ys_global, dtype=np.float64)
    xs_global = np.asarray(xs_global, dtype=np.float64)

    if ys_global.size == 0:
        return 0.0

    corners = np.column_stack(
        [
            np.r_[
                xs_global * dx_km - 0.5 * dx_km,
                xs_global * dx_km - 0.5 * dx_km,
                xs_global * dx_km + 0.5 * dx_km,
                xs_global * dx_km + 0.5 * dx_km,
            ],
            np.r_[
                ys_global * dy_km - 0.5 * dy_km,
                ys_global * dy_km + 0.5 * dy_km,
                ys_global * dy_km - 0.5 * dy_km,
                ys_global * dy_km + 0.5 * dy_km,
            ],
        ]
    )

    poly = _convex_hull_polygon(corners)
    return _polygon_area(poly)


def _boundary_pixels_from_label_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne les pixels de bord d'un masque booléen.

    On utilise une érosion 8-connexe :
      boundary = mask - erosion(mask)

    Les pixels de bord suffisent pour l'enveloppe convexe.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0 or not mask.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Cas très petits : inutile d'éroder.
    if mask.shape[0] <= 2 or mask.shape[1] <= 2:
        return np.where(mask)

    eroded = ndi.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )

    boundary = mask & (~eroded)

    return np.where(boundary)


def _compute_max_convex_hull_area_from_labels_nc(
    labels_nc_path: Path,
    *,
    block_t: int = 20,
) -> float:
    """
    Version rapide et exacte de :

        max_{t,a} |H_t(a)|

    Optimisations :
      1. lecture par blocs temporels ;
      2. boîte englobante comme borne supérieure ;
      3. si bbox_area <= max courant, on saute ;
      4. ConvexHull calculé seulement sur les pixels de bord.
    """
    labels_nc_path = Path(labels_nc_path)

    if not labels_nc_path.exists():
        raise FileNotFoundError(f"labels nc introuvable : {labels_nc_path}")

    with Dataset(labels_nc_path, "r") as nc:
        var_name = choose_label_var_from_netcdf(nc)

        if var_name is None:
            raise KeyError(
                f"Aucune variable de labels 3D trouvée dans {labels_nc_path}"
            )

        latitudes, longitudes = _read_lat_lon_from_labels_nc(nc)
        dx_km, dy_km = infer_xy_km_from_coords(latitudes, longitudes)

        var = nc.variables[var_name]
        nt = int(var.shape[0])

        current_max = 0.0
        n_hull_computed = 0
        n_bbox_skipped = 0
        n_empty_skipped = 0

        for t0 in range(0, nt, block_t):
            t1 = min(nt, t0 + block_t)
            block = np.asarray(var[t0:t1, :, :], dtype=np.int32)

            for k in range(block.shape[0]):
                arr = block[k]

                if not np.any(arr > 0):
                    n_empty_skipped += 1
                    continue

                max_label = int(arr.max())

                if max_label <= 0:
                    n_empty_skipped += 1
                    continue

                # find_objects donne les bounding boxes des labels.
                # L'index 0 correspond au label 1.
                objects = ndi.find_objects(arr, max_label=max_label)

                for label_minus_1, slc in enumerate(objects):
                    if slc is None:
                        continue

                    lbl = label_minus_1 + 1

                    sy, sx = slc
                    ny = int(sy.stop - sy.start)
                    nx = int(sx.stop - sx.start)

                    if ny <= 0 or nx <= 0:
                        continue

                    # Borne supérieure : l'enveloppe convexe ne peut pas dépasser la bbox.
                    bbox_area_km2 = float(ny * dy_km) * float(nx * dx_km)

                    if bbox_area_km2 <= current_max:
                        n_bbox_skipped += 1
                        continue

                    sub = arr[sy, sx]
                    mask = sub == lbl

                    if not mask.any():
                        continue

                    yy_local, xx_local = _boundary_pixels_from_label_mask(mask)

                    if yy_local.size == 0:
                        continue

                    yy_global = yy_local + sy.start
                    xx_global = xx_local + sx.start

                    area_hull = _convex_hull_area_from_boundary_pixels(
                        ys_global=yy_global,
                        xs_global=xx_global,
                        dy_km=dy_km,
                        dx_km=dx_km,
                    )

                    n_hull_computed += 1

                    if np.isfinite(area_hull) and area_hull > current_max:
                        current_max = float(area_hull)

            del block

        print(
            f"[HULL] {labels_nc_path.parent}: "
            f"max_hull={current_max:.3f} km2 | "
            f"ConvexHull calculés={n_hull_computed} | "
            f"bbox skip={n_bbox_skipped} | "
            f"empty skip={n_empty_skipped}"
        )

    return float(current_max)


def _compute_convex_hull_lambda_fraction_for_run(
    run_dir: Path,
    *,
    lambda_max_km: float,
    block_t: int = 20,
) -> tuple[float, float]:
    """
    Calcule et met en cache :

        max_convex_hull_area_km2
        convex_hull_lambda_fraction =
            max_convex_hull_area_km2 / (pi lambda_max_km^2)

    Cache écrit dans :
        run_dir / convex_hull_lambda_fraction_summary.csv
    """
    run_dir = Path(run_dir)
    cache_csv = run_dir / "convex_hull_lambda_fraction_summary.csv"

    if cache_csv.exists():
        try:
            cache = pd.read_csv(cache_csv)

            if (
                "lambda_max_km" in cache.columns
                and "max_convex_hull_area_km2" in cache.columns
                and "convex_hull_lambda_fraction" in cache.columns
                and len(cache) > 0
            ):
                cached_lambda = float(cache["lambda_max_km"].iloc[0])

                if np.isclose(cached_lambda, float(lambda_max_km)):
                    return (
                        float(cache["max_convex_hull_area_km2"].iloc[0]),
                        float(cache["convex_hull_lambda_fraction"].iloc[0]),
                    )

        except Exception:
            pass

    labels_nc = run_dir / "CCS_final_labels.nc"

    max_hull_area_km2 = _compute_max_convex_hull_area_from_labels_nc(
        labels_nc,
        block_t=block_t,
    )

    denom = math.pi * float(lambda_max_km) ** 2
    fraction = max_hull_area_km2 / denom if denom > 0.0 else np.nan

    out = pd.DataFrame(
        [
            {
                "lambda_max_km": float(lambda_max_km),
                "max_convex_hull_area_km2": float(max_hull_area_km2),
                "convex_hull_lambda_fraction": float(fraction),
            }
        ]
    )

    out.to_csv(cache_csv, index=False)

    return float(max_hull_area_km2), float(fraction)

# =============================================================================
# Ajout des métriques cumulées bornées IA111..IA666 / IE111..IE666
# depuis les CSV par label
# =============================================================================

BOUNDED_CUM_SUFFIXES = ["111", "222", "333", "444", "555", "666"]

BOUNDED_CUM_MAP = {
    "111": "m1",  # max_a
    "222": "m2",  # Q95_a
    "333": "m3",  # pondération volume
    "444": "m4",  # pondération 1/volume
    "555": "m5",  # mean_a
    "666": "m6",  # median_a
}


def _summarize_label_metric(values, volumes) -> dict[str, float]:
    """
    Résume une métrique label par label.

    m1 = max
    m2 = Q95
    m3 = moyenne pondérée par V
    m4 = moyenne pondérée par 1/V
    m5 = moyenne
    m6 = médiane
    """
    out = {k: np.nan for k in ("m1", "m2", "m3", "m4", "m5", "m6")}

    x = np.asarray(values, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(v) & (v > 0.0)

    if not m.any():
        return out

    x = x[m]
    v = v[m]

    out["m1"] = float(np.nanmax(x))
    out["m2"] = float(np.nanpercentile(x, 95))

    if np.isfinite(v.sum()) and v.sum() > 0.0:
        w = v / v.sum()
        out["m3"] = float(np.nansum(w * x))

    wi = 1.0 / v
    if np.isfinite(wi.sum()) and wi.sum() > 0.0:
        wi = wi / wi.sum()
        out["m4"] = float(np.nansum(wi * x))

    out["m5"] = float(np.nanmean(x))
    out["m6"] = float(np.nanmedian(x))

    return out


def _bounded_time_mean_values_from_label_df(
    df_label: pd.DataFrame,
    *,
    raw_sum_col: str,
    old_sum_norm_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule, label par label :

        bounded_sum(a) = sum_t A_t(a) I(a,t) / sum_t A_t(a)

    Dans les CSV label :
      volume_vox       = sum_t A_t(a)
      raw_sum_col      = sum_t A_t(a) I(a,t)

    Si raw_sum_col est absent mais old_sum_norm_col et Amax_px existent :
      old_sum_norm_col = sum_t A_t(a) I(a,t) / Amax(a)
      donc :
      raw_sum = old_sum_norm_col * Amax_px
    """
    if "volume_vox" not in df_label.columns:
        raise KeyError("Colonne absente dans CSV label : volume_vox")

    vol = np.asarray(df_label["volume_vox"], dtype=np.float64)

    if raw_sum_col in df_label.columns:
        raw_sum = np.asarray(df_label[raw_sum_col], dtype=np.float64)

    elif old_sum_norm_col in df_label.columns and "Amax_px" in df_label.columns:
        old_sum_norm = np.asarray(df_label[old_sum_norm_col], dtype=np.float64)
        amax = np.asarray(df_label["Amax_px"], dtype=np.float64)
        raw_sum = old_sum_norm * amax

    else:
        raise KeyError(
            f"Impossible de calculer la métrique bornée. "
            f"Colonnes attendues : {raw_sum_col} ou ({old_sum_norm_col} + Amax_px). "
            f"Colonnes disponibles : {list(df_label.columns)}"
        )

    values = _safe_div(raw_sum, vol)

    return values, vol


def _add_bounded_summary_to_out(
    out: dict[str, float],
    *,
    df_label: pd.DataFrame,
    prefix: str,
    raw_sum_col: str,
    old_sum_norm_col: str,
) -> None:
    """
    Ajoute prefix111..prefix666 dans out.

    Exemples :
      prefix="IA"      -> IA111..IA666
      prefix="IE"      -> IE111..IE666
      prefix="INIT_IA" -> INIT_IA111..INIT_IA666
      prefix="INIT_IE" -> INIT_IE111..INIT_IE666
    """
    values, volumes = _bounded_time_mean_values_from_label_df(
        df_label,
        raw_sum_col=raw_sum_col,
        old_sum_norm_col=old_sum_norm_col,
    )

    summary = _summarize_label_metric(values, volumes)

    for suffix, key in BOUNDED_CUM_MAP.items():
        out[f"{prefix}{suffix}"] = summary[key]


def _compute_bounded_metrics_from_label_csvs(
    adj_csv: Optional[Path],
    ent_csv: Optional[Path],
    init_adj_csv: Optional[Path] = None,
    init_ent_csv: Optional[Path] = None,
) -> dict[str, float]:
    """
    Calcule si possible :

      final :
        IA111..IA666
        IE111..IE666

      initial :
        INIT_IA111..INIT_IA666 si initial_adjacency_label_scores.csv existe
        INIT_IE111..INIT_IE666 si initial_entanglement_label_scores.csv existe
    """
    out = {}

    # ------------------------------------------------------------------
    # Final adjacency : IA111..IA666
    # ------------------------------------------------------------------
    if adj_csv is not None and adj_csv.exists():
        adj_df = pd.read_csv(adj_csv)

        _add_bounded_summary_to_out(
            out,
            df_label=adj_df,
            prefix="IA",
            raw_sum_col="adj_AIt_sum_raw",
            old_sum_norm_col="IA_sum_norm",
        )

    # ------------------------------------------------------------------
    # Final entanglement : IE111..IE666
    # ------------------------------------------------------------------
    if ent_csv is not None and ent_csv.exists():
        ent_df = pd.read_csv(ent_csv)

        _add_bounded_summary_to_out(
            out,
            df_label=ent_df,
            prefix="IE",
            raw_sum_col="ent_AIt_sum_raw",
            old_sum_norm_col="E_sum_norm",
        )

    # ------------------------------------------------------------------
    # Initial adjacency : INIT_IA111..INIT_IA666
    # seulement si le CSV existe.
    # ------------------------------------------------------------------
    if init_adj_csv is not None and init_adj_csv.exists():
        init_adj_df = pd.read_csv(init_adj_csv)

        _add_bounded_summary_to_out(
            out,
            df_label=init_adj_df,
            prefix="INIT_IA",
            raw_sum_col="adj_AIt_sum_raw",
            old_sum_norm_col="IA_sum_norm",
        )

    # ------------------------------------------------------------------
    # Initial entanglement : INIT_IE111..INIT_IE666
    # ------------------------------------------------------------------
    if init_ent_csv is not None and init_ent_csv.exists():
        init_ent_df = pd.read_csv(init_ent_csv)

        _add_bounded_summary_to_out(
            out,
            df_label=init_ent_df,
            prefix="INIT_IE",
            raw_sum_col="ent_AIt_sum_raw",
            old_sum_norm_col="E_sum_norm",
        )

    # Diagnostic léger : les métriques 111..666 doivent être dans [0,1].
    for k, v in out.items():
        if np.isfinite(v) and (v < -1e-6 or v > 1.000001):
            print(
                f"[WARNING] {k} = {v:.6f} hors [0,1]. "
                "Vérifier les CSV label et la définition instantanée."
            )

    return out


def _format_int_parameter(x) -> int:
    return int(round(float(x)))


def _candidate_run_dir_names(x_parameter: str, x_value) -> list[str]:
    """
    Génère des noms possibles de dossiers de run selon le paramètre.
    """
    iv = _format_int_parameter(x_value)

    if x_parameter == "lambda_min_km":
        return [
            f"lambda_min_{iv:03d}km",
            f"lambda_min_{iv}km",
        ]

    if x_parameter == "lambda_max_km":
        return [
            f"lambda_max_{iv:04d}km",
            f"lambda_max_{iv}km",
        ]

    if x_parameter == "delta_t_requested_h":
        return [
            f"delta_t_{iv}h",
            f"delta_t_{iv:01d}h",
        ]

    if x_parameter == "sigma_km":
        return [
            f"sigma_{iv:02d}km",
            f"sigma_{iv:03d}km",
            f"sigma_{iv}km",
            f"g1_sigma_{iv:02d}km",
            f"g1_sigma_{iv:03d}km",
            f"g1_sigma_{iv}km",
        ]

    if x_parameter == "Tb_seed_K":
        return [
            f"Tb_seed_{iv:03d}K",
            f"Tb_seed_{iv}K",
            f"Tbmin_{iv:03d}K",
            f"Tbmin_{iv}K",
            f"tbseed_{iv:03d}K",
            f"tbseed_{iv}K",
            f"tbmin_{iv:03d}K",
            f"tbmin_{iv}K",
        ]

    return [
        f"{x_parameter}_{iv}",
    ]


def _run_dir_name_matches_parameter(name: str, x_parameter: str, x_value) -> bool:
    """
    Fallback robuste si le nom exact du dossier n'est pas dans les candidats.
    """
    iv = _format_int_parameter(x_value)
    s = str(name)

    if x_parameter == "lambda_min_km":
        return ("lambda_min" in s) and (str(iv) in s)

    if x_parameter == "lambda_max_km":
        return ("lambda_max" in s) and (str(iv) in s)

    if x_parameter == "delta_t_requested_h":
        return ("delta_t" in s) and (str(iv) in s)

    if x_parameter == "sigma_km":
        return ("sigma" in s.lower()) and (str(iv) in s)

    if x_parameter == "Tb_seed_K":
        sl = s.lower()
        return (("tb" in sl) or ("seed" in sl)) and (str(iv) in s)

    return str(iv) in s


def _find_label_score_csvs_for_row(
    *,
    experiment_root: Path,
    case_id: str,
    x_parameter: str,
    x_value,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Retrouve les CSV label pour une ligne du CSV global.

    Retourne :
      adjacency_label_scores.csv
      entanglement_label_scores.csv
      initial_adjacency_label_scores.csv si existe
      initial_entanglement_label_scores.csv si existe
    """
    case_dir = experiment_root / str(case_id)

    if not case_dir.exists():
        return None, None, None, None

    # Essai 1 : noms de dossiers attendus.
    for dirname in _candidate_run_dir_names(x_parameter, x_value):
        run_dir = case_dir / dirname

        adj_csv = run_dir / "adjacency_label_scores.csv"
        ent_csv = run_dir / "entanglement_label_scores.csv"

        init_adj_csv = run_dir / "initial_adjacency_label_scores.csv"
        init_ent_csv = run_dir / "initial_entanglement_label_scores.csv"

        if adj_csv.exists() and ent_csv.exists():
            return (
                adj_csv,
                ent_csv,
                init_adj_csv if init_adj_csv.exists() else None,
                init_ent_csv if init_ent_csv.exists() else None,
            )

    # Essai 2 : fallback par recherche dans le dossier du cas.
    for adj_csv in case_dir.rglob("adjacency_label_scores.csv"):
        run_dir = adj_csv.parent
        ent_csv = run_dir / "entanglement_label_scores.csv"

        if not ent_csv.exists():
            continue

        if _run_dir_name_matches_parameter(run_dir.name, x_parameter, x_value):
            init_adj_csv = run_dir / "initial_adjacency_label_scores.csv"
            init_ent_csv = run_dir / "initial_entanglement_label_scores.csv"

            return (
                adj_csv,
                ent_csv,
                init_adj_csv if init_adj_csv.exists() else None,
                init_ent_csv if init_ent_csv.exists() else None,
            )

    return None, None, None, None


def add_bounded_cumulative_metrics_from_label_csvs(
    df: pd.DataFrame,
    *,
    all_cases_csv_path: Path,
    x_parameter: str,
    kind: str,
) -> pd.DataFrame:
    """
    Ajoute au DataFrame global :

      final :
        IA111..IA666
        IE111..IE666

      initial si CSV disponibles :
        INIT_IA111..INIT_IA666
        INIT_IE111..INIT_IE666
    """
    df = df.copy()

    all_cases_csv_path = Path(all_cases_csv_path).expanduser().resolve()
    experiment_root = all_cases_csv_path.parent

    new_cols = [
        "IA111", "IA222", "IA333", "IA444", "IA555", "IA666",
        "IE111", "IE222", "IE333", "IE444", "IE555", "IE666",

        "INIT_IA111", "INIT_IA222", "INIT_IA333",
        "INIT_IA444", "INIT_IA555", "INIT_IA666",

        "INIT_IE111", "INIT_IE222", "INIT_IE333",
        "INIT_IE444", "INIT_IE555", "INIT_IE666",

        # Nouvelle metrique enveloppe convexe
        "max_convex_hull_area_km2",
        "convex_hull_lambda_fraction",
    ]

    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "case_id" not in df.columns:
        print(f"[SKIP] {kind}: colonne case_id absente, impossible d'ajouter IA111/IE111.")
        return df

    if x_parameter not in df.columns:
        print(f"[SKIP] {kind}: colonne {x_parameter} absente, impossible d'ajouter IA111/IE111.")
        return df

    n_ok = 0
    n_missing = 0
    n_error = 0

    for idx, row in df.iterrows():
        case_id = _clean_case_id(row["case_id"])
        x_value = row[x_parameter]

        adj_csv, ent_csv, init_adj_csv, init_ent_csv = _find_label_score_csvs_for_row(
            experiment_root=experiment_root,
            case_id=case_id,
            x_parameter=x_parameter,
            x_value=x_value,
        )

        if adj_csv is None or ent_csv is None:
            n_missing += 1
            continue

        try:
            metrics = _compute_bounded_metrics_from_label_csvs(
                adj_csv=adj_csv,
                ent_csv=ent_csv,
                init_adj_csv=init_adj_csv,
                init_ent_csv=init_ent_csv,
            )

            # ----------------------------------------------------------
            # Nouvelle metrique :
            # max_{t,a} |H_t(a)| / (pi lambda_max^2)
            # ----------------------------------------------------------
            run_dir = Path(adj_csv).parent

            lambda_max_km = _lambda_max_for_row(
                row,
                kind=kind,
                x_parameter=x_parameter,
            )

            max_hull_area_km2, hull_fraction = (
                _compute_convex_hull_lambda_fraction_for_run(
                    run_dir,
                    lambda_max_km=lambda_max_km,
                    block_t=20,
                )
            )

            metrics["max_convex_hull_area_km2"] = max_hull_area_km2
            metrics["convex_hull_lambda_fraction"] = hull_fraction

            for col, value in metrics.items():
                df.loc[idx, col] = value

            n_ok += 1

        except Exception as exc:
            n_error += 1
            print(
                f"[WARNING] {kind}: échec métriques enrichies pour "
                f"{case_id}, {x_parameter}={x_value}: {exc}"
            )

    print(
        f"[BOUNDED] {kind}: IA111..IA666 / IE111..IE666 ajoutées "
        f"pour {n_ok} lignes. Missing={n_missing}, errors={n_error}."
    )

    return df

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
    "1":  r"$\underset{t}{\max}$",
    "2":  r"$\underset{t}{\max}$",
    "3":  r"$\underset{t}{\max}$",
    "4":  r"$\underset{t}{\max}$",
    "5":  r"$\underset{t}{\max}$",
    "6":  r"$\underset{t}{\max}$",

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


def focus_filename_for_pair(parameter_name: str, ia_metric: str, ie_metric: str) -> str:
    suffix = metric_suffix_from_name(ia_metric)

    temporal = TEMPORAL_FILENAME_BY_SUFFIX.get(suffix, "unknown_time")
    reduction = OBJECT_REDUCTION_FILENAME_BY_SUFFIX.get(suffix, "unknown_reduction")

    return f"fig_focus_{parameter_name}_{temporal}_{reduction}_{ia_metric}_{ie_metric}"


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


def _metric_has_finite_values(data: pd.DataFrame, metric: str) -> bool:
    if metric not in data.columns:
        return False

    vals = np.asarray(data[metric], dtype=float)
    return bool(np.isfinite(vals).any())


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
    legacy_heterogeneity: bool,
    n_ylim=(0, 200),
    amax_ylim=(0, 1e6),
    metric_ylim=(-0.01, 1.01),
) -> bool:
    """
    Trace une ligne avec 4 colonnes :

      1. Nombre final + fraction 1 - n_i/n_f
      2. Amax90 + max(Amax) + fraction enveloppe convexe
      3. IH10/90 + IH25/75 + Gini
      4. IA/IE final + IA/IE initial si disponibles
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
    # Colonne 2 : Amax90 + max(Amax) + enveloppe convexe normalisee
    # ==================================================================
    ax = axs_row[1]
    ax.axvline(x=default_x, linewidth=1, linestyle=":", c="k")

    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric="Amax_p90_km2",
        label=r"$A_{\max}^{90}$",
        color="brown",
        linestyle="-",
        show_iqr=True,
    )

    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric="Amax_max_km2",
        label=r"$\max(A_{\max})$",
        color="saddlebrown",
        linestyle="--",
        show_iqr=True,
    )

    _style_axis(
        ax,
        xlabel=xlabel,
        ylabel=r"Aggregate size $A_{\max}$" "\n" r"(km$^2$) in final segmentation",
        title=rf"{param_title} : size / envelope",
    )

    if amax_ylim is not None:
        ax.set_ylim(amax_ylim)
    ax.set_ylabel(r"Aggregate size $A_{\max}$" "\n" r"(km$^2$) in final segmentation", color="brown")
    ax.tick_params(axis="y", colors="brown")

    ax2 = ax.twinx()

    plotted_hull = _plot_median_iqr_curve(
        ax2,
        data,
        x_parameter=x_parameter,
        metric="convex_hull_lambda_fraction",
        label=r"$\max_{t,a} A_{\mathrm{hull}} / (\pi \lambda_{\max}^2)$",
        color="purple",
        linestyle="-.",
        show_iqr=True,
    )

    if plotted_hull:
        ax2.set_ylabel(
            r"Convex-hull fraction",
            color="purple",
        )
        ax2.tick_params(axis="y", colors="purple")
        ax2.set_ylim((-0.01, 1.05))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    # ==================================================================
    # Colonne 3 : IH10/90 + IH25/75 + Gini
    # ==================================================================
    ax = axs_row[2]
    ax.axvline(x=default_x, linewidth=1, linestyle=":", c="k")

    # Axe gauche : hétérogénéités IH
    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric="IH",
        label=r"$I_H^{10/90}=1-P10/P90$",
        color="orange",
        linestyle="-",
        show_iqr=True,
    )

    _plot_median_iqr_curve(
        ax,
        data,
        x_parameter=x_parameter,
        metric="IH25_75",
        label=r"$I_H^{25/75}=1-P25/P75$",
        color="red",
        linestyle="--",
        show_iqr=True,
    )

    _style_axis(
        ax,
        xlabel=xlabel,
        ylabel=r"Heterogeneity $I_H$",
        title=rf"{param_title} : heterogeneity / Gini",
    )

    ax.set_ylim((-0.01, 1.01))
    ax.set_ylabel(r"Heterogeneity $I_H$", color="orange")
    ax.tick_params(axis="y", colors="orange")
    ax.spines["left"].set_color("orange")

    # Axe droit : Gini(Amax)
    ax_gini = ax.twinx()

    gini_color = "navy"
    _plot_median_iqr_curve(
        ax_gini,
        data,
        x_parameter=x_parameter,
        metric="Amax_gini",
        label=r"$Gini(A_{\max})$",
        color=gini_color,
        linestyle="-.",
        show_iqr=True,
    )

    ax_gini.set_ylabel(r"$Gini(A_{\max})$", color=gini_color)
    ax_gini.tick_params(axis="y", colors=gini_color)
    ax_gini.spines["right"].set_color(gini_color)
    ax_gini.set_ylim((-0.01, 1.01))

    # Légende combinée gauche + droite
    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax_gini.get_legend_handles_labels()

    ax.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        fontsize=7,
        loc="best",
    )
    # ==================================================================
    # Colonne 4 : IA/IE final + IA/IE initial
    # ==================================================================
    ax = axs_row[3]
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
    data_lambda_max: pd.DataFrame,
    data_delta_t: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
    legacy_heterogeneity: bool,
):
    """
    Produit une seule figure pour une paire IA/IE.

    Lignes :
      sigma
      lambda_min
      Tbmin
      lambda_max
      delta_t

    Colonnes :
      1. Nombre + fraction 1 - n_i/n_f
      2. Amax90 + max(Amax) + enveloppe convexe normalisee
      3. IH10/90 + IH25/75 + Gini
      4. IA/IE final + initial
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
            amax_ylim=(0, 1e6),
            metric_ylim=metric_ylim,
        ),
        dict(
            parameter_name="lambda_min",
            data=data_lambda_min,
            x_parameter="lambda_min_km",
            xlabel=r"$\lambda_{\min}$ (km)",
            default_x=100,
            n_ylim=(0, 500),
            amax_ylim=(0, 1e6),
            metric_ylim=metric_ylim,
        ),
        dict(
            parameter_name="Tbmin",
            data=data_Tbmin,
            x_parameter="Tb_seed_K",
            xlabel=r"$T_{b,\min}$ (K)",
            default_x=220,
            n_ylim=(0, 200),
            amax_ylim=(0, 1e6),
            metric_ylim=metric_ylim,
        ),
        dict(
            parameter_name="lambda_max",
            data=data_lambda_max,
            x_parameter="lambda_max_km",
            xlabel=r"$\lambda_{\max}$ (km)",
            default_x=1500,
            n_ylim=(0, 200),
            amax_ylim=(0, 1e6),
            metric_ylim=metric_ylim,
        ),
        dict(
            parameter_name="delta_t",
            data=data_delta_t,
            x_parameter="delta_t_requested_h",
            xlabel=r"$\Delta t$ (h)",
            default_x=3,
            n_ylim=(0, 200),
            amax_ylim=(0, 1e6),
            metric_ylim=metric_ylim,
        ),
    ]

    n_rows = len(parameter_specs)

    fig, axs = plt.subplots(
        n_rows,
        4,
        figsize=(18.5, 3.2 * n_rows),
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
            legacy_heterogeneity=legacy_heterogeneity,
            n_ylim=spec["n_ylim"],
            amax_ylim=spec["amax_ylim"],
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

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

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
    data_lambda_max: pd.DataFrame,
    data_delta_t: pd.DataFrame,
    out_dir: Path,
    formats: Iterable[str],
    legacy_heterogeneity: bool,
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
            data_lambda_max=data_lambda_max,
            data_delta_t=data_delta_t,
            out_dir=out_dir,
            formats=formats,
            legacy_heterogeneity=legacy_heterogeneity,
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
        help="Dossier contenant les sous-dossiers de sensibilite.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Dossier de sortie des figures.",
    )

    parser.add_argument(
        "--tbmin-csv",
        type=str,
        default=None,
        help="Chemin direct vers le CSV Tbmin.",
    )

    parser.add_argument(
        "--lambda-min-csv",
        type=str,
        default=None,
        help="Chemin direct vers le CSV lambda_min.",
    )

    parser.add_argument(
        "--sigma-csv",
        type=str,
        default=None,
        help="Chemin direct vers le CSV sigma.",
    )

    parser.add_argument(
        "--lambda-max-csv",
        type=str,
        default=None,
        help="Chemin direct vers le CSV lambda_max.",
    )

    parser.add_argument(
        "--delta-t-csv",
        type=str,
        default=None,
        help="Chemin direct vers le CSV delta_t.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Formats de sortie.",
    )

    parser.add_argument(
        "--legacy-heterogeneity",
        action="store_true",
        help=(
            "Utiliser exactement l'expression heterogeneite du notebook : "
            "1 - (P90-P10)/P90 = P10/P90."
        ),
    )

    parser.add_argument(
        "--allow-cases-one",
        action="store_true",
        help="Autoriser explicitement la lecture des anciens runs dans cases_one.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    formats = args.formats

    # Nettoyer uniquement les anciennes figures produites par ce script.
    # Ne touche pas aux CSV.
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_fig in out_dir.glob("fig_*.*"):
        if old_fig.suffix.lower() in [".png", ".pdf", ".svg"]:
            old_fig.unlink()

    (
        data_Tbmin,
        data_lambda_min,
        data_sigma,
        data_lambda_max,
        data_delta_t,
    ) = load_all_data(args)

    enriched_dir = out_dir / "enriched_csv"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    data_Tbmin.to_csv(enriched_dir / "Tbmin_enriched.csv", index=False)
    data_lambda_min.to_csv(enriched_dir / "lambda_min_enriched.csv", index=False)
    data_sigma.to_csv(enriched_dir / "sigma_enriched.csv", index=False)
    data_lambda_max.to_csv(enriched_dir / "lambda_max_enriched.csv", index=False)
    data_delta_t.to_csv(enriched_dir / "delta_t_enriched.csv", index=False)

    print(f"CSV enrichis écrits dans : {enriched_dir.resolve()}")
    
    plot_all_focus_IA_IE_pairs(
        data_sigma=data_sigma,
        data_lambda_min=data_lambda_min,
        data_Tbmin=data_Tbmin,
        data_lambda_max=data_lambda_max,
        data_delta_t=data_delta_t,
        out_dir=out_dir,
        formats=formats,
        legacy_heterogeneity=args.legacy_heterogeneity,
    )

    print(f"Figures ecrites dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()