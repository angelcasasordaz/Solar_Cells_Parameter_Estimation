import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import get_optimizer_by_class

from algorithm_acronym_list import resolve_optimizer_name
from dbo_optimizer import DBOOptimizer
from dsade_optimizer import DSADE
from macro_de_optimizer import MaCRO_DE

# This module will be created in the next step.
from solar_objective import (
    AVAILABLE_SOLAR_PROBLEMS,
    build_mealpy_problem,
    evaluate_solution_metrics,
    get_fit_series,
    get_parameter_names,
)


DEFAULT_EPOCHS = 500
DEFAULT_RUNS = 15
CONVERGENCE_CACHE_VERSION = 3

DEFAULT_PROBLEMS = [
    "SingleDiode",
    "DoubleDiode",
    "TripleDiode",
]

DEFAULT_OPTIMIZERS = [
    "DE",
    "PSO",
    "GA",
    "SADE",
    "JADE",
    "SHADE",
    "GWO",
    "WOA",
    "HHO",
    "BRO",
    "RUN",
    "BBOA",
    "FOX",
    "RIME",
]


@dataclass
class Paths:
    exp_tag: str
    fig_dir: str
    res_dir: str
    cache_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar Cells + MEALPY Benchmark Framework")
    parser.add_argument("--exp-id", type=int, default=3, help="Numeric experiment identifier")
    parser.add_argument("--output-root", default=".", help="Root directory for Figures/Results")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cache if available")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS), choices=list(AVAILABLE_SOLAR_PROBLEMS.keys()), help="Solar-cell models to execute")
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS), help="List of optimizers")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum optimization iterations")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Independent runs per optimizer")
    parser.add_argument("--seed-base", type=int, default=1234, help="Base random seed")
    parser.add_argument("--parallel", default="yes", choices=["yes", "no"], help="Execute runs in parallel")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--convergence-extra-scale", default="none", choices=["none", "auto", "log", "symlog", "exp"], help="Save an additional convergence plot with the selected y-axis scale or transformation")
    parser.add_argument("--dsade-beta-min", type=float, default=0.2, help="Minimum adaptive beta")
    parser.add_argument("--dsade-beta-max", type=float, default=0.8, help="Maximum adaptive beta")
    parser.add_argument("--dsade-pcr", type=float, default=0.2, help="Crossover probability")
    parser.add_argument("--dsade-mahal-q", type=float, default=0.68, help="Mahalanobis threshold")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.pop_size <= 1:
        parser.error("--pop-size must be greater than one")
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")

    if args.n_workers is None:
        available_workers = max(1, (os.cpu_count() or 1) - 1)
        args.n_workers = min(available_workers, max(1, args.runs))

    return args

def make_paths(args: argparse.Namespace) -> Paths:
    exp_tag = f"EXP{args.exp_id:03d}"

    fig_dir = os.path.join(
        args.output_root,
        "Figures",
        exp_tag,
    )
    res_dir = os.path.join(
        args.output_root,
        "Results",
        exp_tag,
    )
    cache_dir = os.path.join(
        res_dir,
        "cache",
    )

    for path in (fig_dir, res_dir, cache_dir):
        os.makedirs(path, exist_ok=True)

    return Paths(
        exp_tag=exp_tag,
        fig_dir=fig_dir,
        res_dir=res_dir,
        cache_dir=cache_dir,
    )


def normalize_optimizer_name(name: str) -> str:
    return "".join(
        char.lower()
        for char in str(name)
        if char.isalnum()
    )


def display_optimizer_name(name: str) -> str:
    label = str(name)

    if label.startswith("Original"):
        return label[len("Original"):]

    return label


def is_macro_de_name(name: str) -> bool:
    return normalize_optimizer_name(name) == "macrode"


CONVERGENCE_CMAP = "Set1"
# Qualitative:
# "tab10"
# "tab20"
# "Set1"
# "Set2"
# "Set3"
# "Dark2"
# "Paired"
# "Accent"


def custom_optimizer_kwargs(args: argparse.Namespace) -> dict:
    return {
        "epoch": args.epochs,
        "pop_size": args.pop_size,
        "beta_min": args.dsade_beta_min,
        "beta_max": args.dsade_beta_max,
        "pcr": args.dsade_pcr,
        "mahalanobis_q": args.dsade_mahal_q,
    }


def build_optimizer(
    name: str,
    args: argparse.Namespace,
):
    resolved_name = resolve_optimizer_name(name)
    optimizer_key = normalize_optimizer_name(resolved_name)

    if optimizer_key in ("dsade", "dsadeawad"):
        optimizer_class = DSADE
        optimizer_kwargs = custom_optimizer_kwargs(args)

    elif optimizer_key == "macrode":
        optimizer_class = MaCRO_DE
        optimizer_kwargs = custom_optimizer_kwargs(args)

    elif optimizer_key == "dbo":
        optimizer_class = DBOOptimizer
        optimizer_kwargs = {
            "epoch": args.epochs,
            "pop_size": args.pop_size,
        }

    else:
        optimizer_class = get_optimizer_by_class(resolved_name)
        optimizer_kwargs = {
            "epoch": args.epochs,
            "pop_size": args.pop_size,
        }

    return optimizer_class(**optimizer_kwargs)


def build_cache_signature(args: argparse.Namespace) -> str:
    problem_bounds = {}

    for problem_name in args.problems:
        problem_class = AVAILABLE_SOLAR_PROBLEMS[problem_name]
        problem_bounds[problem_name] = {
            "lb": np.asarray(problem_class.lb, dtype=float).tolist(),
            "ub": np.asarray(problem_class.ub, dtype=float).tolist(),
        }

    payload = {
        "problems": args.problems,
        "problem_bounds": problem_bounds,
        "optimizers": args.optimizers,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "runs": args.runs,
        "seed_base": args.seed_base,
        "beta_min": args.dsade_beta_min,
        "beta_max": args.dsade_beta_max,
        "pcr": args.dsade_pcr,
        "mahalanobis_q": args.dsade_mahal_q,
        "convergence_cache_version": CONVERGENCE_CACHE_VERSION,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha1(encoded).hexdigest()[:10]


def safe_path_component(value) -> str:
    component = "".join(
        char
        if char.isalnum() or char in ("-", "_")
        else "_"
        for char in str(value)
    )

    return component or "item"


def run_checkpoint_path(
    paths: Paths,
    cache_signature: str,
    problem_name: str,
    optimizer_name: str,
    run: int,
) -> str:
    optimizer_tag = (
        f"{safe_path_component(optimizer_name)}_"
        f"{hashlib.sha1(str(optimizer_name).encode('utf-8')).hexdigest()[:8]}"
    )

    checkpoint_dir = os.path.join(
        paths.cache_dir,
        cache_signature,
        safe_path_component(problem_name),
        optimizer_tag,
    )

    return os.path.join(
        checkpoint_dir,
        f"run_{run + 1:03d}.pkl",
    )


def checkpoint_metadata(
    args: argparse.Namespace,
    cache_signature: str,
    problem_name: str,
    optimizer_name: str,
    run: int,
    seed: int,
) -> dict:
    return {
        "cache_signature": cache_signature,
        "problem_name": problem_name,
        "optimizer_name": optimizer_name,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "run": run,
        "seed": seed,
        "convergence_cache_version": CONVERGENCE_CACHE_VERSION,
    }


def save_run_checkpoint(
    checkpoint_path: str,
    metadata: dict,
    output: dict,
) -> None:
    os.makedirs(
        os.path.dirname(checkpoint_path),
        exist_ok=True,
    )

    tmp_path = (
        f"{checkpoint_path}.tmp."
        f"{os.getpid()}."
        f"{time.time_ns()}"
    )

    with open(tmp_path, "wb") as file:
        pickle.dump(
            {
                "metadata": metadata,
                "output": output,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        tmp_path,
        checkpoint_path,
    )


def load_run_checkpoint(
    checkpoint_path: str,
    expected_metadata: dict,
):
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, "rb") as file:
            payload = pickle.load(file)

    except Exception as exc:
        print_status(
            f"CACHE INVALID | path={checkpoint_path} | reason={exc}"
        )
        return None

    if payload.get("metadata") != expected_metadata:
        print_status(
            f"CACHE MISMATCH | path={checkpoint_path}"
        )
        return None

    output = payload.get("output")

    required_keys = {
        "best_fitness",
        "best_solution",
        "runtime",
        "curve",
        "metrics",
    }

    if (
        not isinstance(output, dict)
        or not required_keys.issubset(output)
    ):
        print_status(
            f"CACHE INVALID | path={checkpoint_path} | "
            "reason=missing keys"
        )
        return None

    try:
        output["curve"] = validate_convergence_curve(
            output["curve"],
            int(expected_metadata["epochs"]),
            expected_metadata["optimizer_name"],
            int(expected_metadata["run"]) + 1,
            "cache",
            best_fitness=float(output["best_fitness"]),
        ).copy()

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    return output


def print_status(message: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


def validate_convergence_curve(
    curve,
    expected_length: int,
    optimizer_name: str,
    run_label,
    stage: str,
    best_fitness=None,
) -> np.ndarray:
    curve = np.asarray(curve, dtype=float).reshape(-1).copy()

    if curve.size == 0:
        raise ValueError(
            "Missing convergence history for "
            f"{optimizer_name}, run {run_label}, stage {stage}"
        )

    if expected_length > 1 and curve.size == 1:
        raise ValueError(
            "Single-point convergence history for "
            f"{optimizer_name}, run {run_label}, stage {stage}; "
            f"refusing to expand one value to {expected_length} iterations"
        )

    if curve.size != expected_length:
        raise ValueError(
            "Unexpected convergence length for "
            f"{optimizer_name}, run {run_label}, stage {stage}: "
            f"{curve.size} != {expected_length}"
        )

    if not np.all(np.isfinite(curve)):
        raise ValueError(
            "Convergence history has non-finite values for "
            f"{optimizer_name}, run {run_label}, stage {stage}"
        )

    if best_fitness is not None:
        if not bool(
            np.isclose(
                curve[-1],
                float(best_fitness),
                rtol=1e-9,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Convergence final value does not match best fitness for "
                f"{optimizer_name}, run {run_label}, stage {stage}"
            )

    if np.any(
        np.diff(curve) > 1e-12
    ):
        raise ValueError(
            "Convergence history is not best-so-far for "
            f"{optimizer_name}, run {run_label}, stage {stage}"
        )

    return curve


def run_single(
    problem_name: str,
    optimizer_name: str,
    args: argparse.Namespace,
    seed: int,
    run_index=None,
    total_runs=None,
) -> dict:
    logging.disable(logging.INFO)
    np.random.seed(seed)

    run_label = (
        f"run {run_index + 1}/{total_runs}"
        if run_index is not None and total_runs is not None
        else "run"
    )

    mealpy_problem, solar_problem = build_mealpy_problem(
        problem_name
    )

    print_status(
        "START | "
        f"problem={problem_name} | "
        f"optimizer={optimizer_name} | "
        f"{run_label} | "
        f"dims={solar_problem.dims} | "
        f"epochs={args.epochs} | "
        f"pop={args.pop_size} | "
        f"seed={seed}"
    )

    optimizer = build_optimizer(
        optimizer_name,
        args,
    )

    start_time = time.time()
    result = optimizer.solve(
        mealpy_problem,
        seed=seed,
    )
    runtime = time.time() - start_time

    best_solution = np.asarray(
        result.solution,
        dtype=float,
    )

    metrics = evaluate_solution_metrics(
        problem_name,
        best_solution,
    )

    convergence = validate_convergence_curve(
        np.asarray(
            optimizer.history.list_global_best_fit,
            dtype=float,
        ).copy(),
        args.epochs,
        optimizer_name,
        run_index + 1 if run_index is not None else "NA",
        "history",
        best_fitness=float(result.target.fitness),
    )

    print_status(
        "DONE  | "
        f"problem={problem_name} | "
        f"optimizer={optimizer_name} | "
        f"{run_label} | "
        f"rmse={metrics['RMSE']:.6e} | "
        f"nrmse={metrics['NRMSE']:.6e} | "
        f"time={runtime:.2f}s"
    )

    return {
        "best_fitness": float(result.target.fitness),
        "best_solution": best_solution,
        "runtime": float(runtime),
        "curve": convergence.copy(),
        "metrics": metrics,
    }


def run_parallel_task(task: dict):
    output = run_single(
        task["problem_name"],
        task["optimizer_name"],
        task["args"],
        task["seed"],
        task["run"],
        task["total_runs"],
    )

    save_run_checkpoint(
        task["checkpoint_path"],
        task["metadata"],
        output,
    )

    return task["run"], output


def resolve_convergence_scale(
    curves_dict: dict,
    requested_scale: str,
):
    if requested_scale == "none":
        return None

    if requested_scale in ("symlog", "exp"):
        return requested_scale

    finite_values = []

    for curve in curves_dict.values():
        curve = np.asarray(curve, dtype=float)

        finite_values.extend(
            curve[np.isfinite(curve)]
        )

    if len(finite_values) == 0:
        return None

    min_value = np.min(finite_values)

    if requested_scale == "auto":
        return "log" if min_value > 0 else "symlog"

    if requested_scale == "log" and min_value <= 0:
        print_status(
            "CONVERGENCE SCALE | log requires positive values; "
            "using symlog"
        )
        return "symlog"

    return requested_scale


def plot_convergence(
    curves_dict: dict,
    title: str,
    out_path: str,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(
        figsize=(10, 5),
        facecolor="white",
    )

    plot_items = list(
        curves_dict.items()
    )
    macro_idx = next(
        (
            index
            for index, (optimizer_name, _) in enumerate(plot_items)
            if is_macro_de_name(optimizer_name)
        ),
        None,
    )

    if macro_idx is not None:
        macro_item = plot_items.pop(macro_idx)
        plot_items.append(macro_item)

    n_algorithms = len(plot_items)
    colormap = plt.get_cmap(
        CONVERGENCE_CMAP,
        max(n_algorithms, 1),
    )
    colors = colormap(
        np.arange(n_algorithms)
    )

    for (optimizer_name, curve), color in zip(
        plot_items,
        colors,
    ):
        curve = validate_convergence_curve(
            curve,
            len(curve),
            optimizer_name,
            "Mean",
            "before_plot",
        )

        if yscale == "exp":
            plot_curve = np.where(
                np.isfinite(curve),
                np.exp(
                    np.clip(
                        curve,
                        -745.0,
                        709.0,
                    )
                ),
                np.nan,
            )

        elif yscale == "log":
            plot_curve = np.where(
                np.isfinite(curve) & (curve > 0.0),
                curve,
                np.nan,
            )

        else:
            plot_curve = np.asarray(
                curve,
                dtype=float,
            )

        iterations = np.arange(
            1,
            len(plot_curve) + 1,
        )

        is_macro_de = is_macro_de_name(
            optimizer_name
        )

        if is_macro_de:
            ax.plot(
                iterations,
                plot_curve,
                linewidth=4.4,
                label="_nolegend_",
                color=color,
                solid_capstyle="round",
            )

        ax.plot(
            iterations,
            plot_curve,
            linewidth=(
                3.0
                if is_macro_de
                else 2.2
            ),
            label=display_optimizer_name(
                optimizer_name
            ),
            color=color,
            solid_capstyle="round",
        )

    if yscale in ("log", "symlog"):
        ax.set_yscale(yscale)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "exp(RMSE)"
        if yscale == "exp"
        else "RMSE"
    )
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_problem_convergence_plots(
    curves_dict: dict,
    problem_name: str,
    paths: Paths,
    requested_extra_scale: str,
) -> None:
    plot_convergence(
        curves_dict,
        f"Convergence Curve - {problem_name}",
        os.path.join(
            paths.fig_dir,
            (
                f"{paths.exp_tag}_"
                f"{problem_name}_"
                "convergence.png"
            ),
        ),
        yscale="linear",
    )

    extra_scale = resolve_convergence_scale(
        curves_dict,
        requested_extra_scale,
    )

    if extra_scale is None:
        return

    plot_convergence(
        curves_dict,
        (
            f"Convergence Curve - {problem_name} "
            f"({extra_scale.upper()} Scale)"
        ),
        os.path.join(
            paths.fig_dir,
            (
                f"{paths.exp_tag}_"
                f"{problem_name}_"
                f"convergence_{extra_scale}.png"
            ),
        ),
        yscale=extra_scale,
    )


def plot_fit_check(
    problem_name: str,
    optimizer_data: dict,
    paths: Paths,
) -> None:
    n_optimizers = len(optimizer_data)

    if n_optimizers == 0:
        return

    n_cols = min(3, n_optimizers)
    n_rows = int(np.ceil(n_optimizers / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 3.6 * n_rows),
        squeeze=False,
        facecolor="white",
    )

    ordered_items = list(
        optimizer_data.items()
    )
    macro_idx = next(
        (
            index
            for index, (optimizer_name, _) in enumerate(ordered_items)
            if is_macro_de_name(optimizer_name)
        ),
        None,
    )

    if macro_idx is not None:
        macro_item = ordered_items.pop(macro_idx)
        ordered_items.append(macro_item)

    for ax, (optimizer_name, data) in zip(axes.flat, ordered_items):
        best_run_index = int(np.argmin(data["rmse_runs"]))
        best_solution = data["best_solutions"][best_run_index]
        series = get_fit_series(problem_name, best_solution)

        voltage = series["voltage"]
        measured_current = series["measured_current"]
        estimated_current = series["estimated_current"]
        measured_power = series["measured_power"]
        estimated_power = series["estimated_power"]

        ax.scatter(
            voltage,
            measured_current,
            s=24,
            color="red",
            marker="o",
            label="Measured I",
            zorder=3,
        )
        ax.plot(
            voltage,
            estimated_current,
            color="blue",
            linewidth=2.8,
            label="Estimated I",
        )
        ax.scatter(
            voltage,
            measured_power,
            s=28,
            color="black",
            marker="^",
            label="Measured P",
            zorder=3,
        )
        ax.plot(
            voltage,
            estimated_power,
            color="#00cfd1",
            linestyle="--",
            linewidth=2.8,
            label="Estimated P",
        )

        ax.set_title(
            f"{display_optimizer_name(optimizer_name)} | "
            f"RMSE={np.min(data['rmse_runs']):.3e}"
        )
        ax.set_xlabel("Voltage")
        ax.set_ylabel("Current & Power")
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(ordered_items):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"I-V / P-V Fit Check - {problem_name}",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    out_path = os.path.join(
        paths.fig_dir,
        f"{paths.exp_tag}_{problem_name}_iv_pv_check.png",
    )
    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def summarize_array(values) -> dict:
    values = np.asarray(values, dtype=float)

    return {
        "Best": float(np.min(values)),
        "Mean": float(np.mean(values)),
        "Median": float(np.median(values)),
        "Std": float(np.std(values)),
    }


def is_saturation_current_column(column_name: str) -> bool:
    normalized_name = str(column_name)

    if normalized_name.startswith("Best_"):
        normalized_name = normalized_name[len("Best_"):]

    return normalized_name in {
        "Isd",
        "Isd1",
        "Isd2",
        "Isd3",
    }


def write_excel_with_parameter_formats(
    dataframe: pd.DataFrame,
    path: str,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dataframe.to_excel(
            writer,
            index=False,
        )

        worksheet = writer.sheets["Sheet1"]

        for column_index, column_name in enumerate(
            dataframe.columns,
            start=1,
        ):
            if not is_saturation_current_column(column_name):
                continue

            for row in worksheet.iter_rows(
                min_row=2,
                min_col=column_index,
                max_col=column_index,
            ):
                row[0].number_format = "0.000000E+00"


def convergence_filename(problem_name: str) -> str:
    problem_filenames = {
        "SingleDiode": "convergence_values_single_diode.xlsx",
        "DoubleDiode": "convergence_values_double_diode.xlsx",
        "TripleDiode": "convergence_values_triple_diode.xlsx",
    }

    return problem_filenames.get(
        problem_name,
        f"convergence_values_{safe_path_component(problem_name).lower()}.xlsx",
    )


def unique_extra_excel_path(path: str) -> str:
    if not os.path.exists(path):
        return path

    base_path, extension = os.path.splitext(path)
    suffix = 1

    while True:
        candidate_path = f"{base_path}_{suffix:03d}{extension}"

        if not os.path.exists(candidate_path):
            return candidate_path

        suffix += 1


def excel_sheet_name(name: str, used_names: set) -> str:
    invalid_characters = set(r"[]:*?/\\")
    clean_name = "".join(
        "_" if char in invalid_characters else char
        for char in str(name)
    ).strip()
    clean_name = clean_name or "Optimizer"
    clean_name = clean_name[:31]

    sheet_name = clean_name
    suffix = 1

    while sheet_name in used_names:
        suffix_text = f"_{suffix}"
        sheet_name = f"{clean_name[:31 - len(suffix_text)]}{suffix_text}"
        suffix += 1

    used_names.add(sheet_name)

    return sheet_name


def export_convergence_values(
    results_struct: dict,
    paths: Paths,
) -> list:
    convergence_paths = []

    for problem_name, optimizer_data in results_struct.items():
        if not optimizer_data:
            continue

        convergence_path = os.path.join(
            paths.res_dir,
            convergence_filename(problem_name),
        )
        convergence_path = unique_extra_excel_path(
            convergence_path
        )

        used_sheet_names = set()

        with pd.ExcelWriter(convergence_path, engine="openpyxl") as writer:
            for optimizer_name, data in optimizer_data.items():
                curves_by_run = data["curves_by_run"]
                mean_curve = np.asarray(
                    data["curve"],
                    dtype=float,
                ).reshape(-1)

                convergence_data = {
                    "Iteration": np.arange(
                        1,
                        mean_curve.size + 1,
                    ),
                }

                for run_number in sorted(curves_by_run):
                    convergence_data[f"Run_{run_number}"] = np.asarray(
                        curves_by_run[run_number],
                        dtype=float,
                    )

                convergence_data["Mean"] = mean_curve

                pd.DataFrame(convergence_data).to_excel(
                    writer,
                    sheet_name=excel_sheet_name(
                        optimizer_name,
                        used_sheet_names,
                    ),
                    index=False,
                )

        convergence_paths.append(convergence_path)

    return convergence_paths


def export_results(
    results_struct: dict,
    summary_path: str,
    runs_path: str,
):
    summary_rows = []
    run_rows = []

    for problem_name, optimizer_data in results_struct.items():
        parameter_names = get_parameter_names(problem_name)

        for optimizer_name, data in optimizer_data.items():
            rmse_stats = summarize_array(
                data["rmse_runs"]
            )
            nrmse_stats = summarize_array(
                data["nrmse_runs"]
            )
            ae_stats = summarize_array(
                data["ae_runs"]
            )
            mae_stats = summarize_array(
                data["mae_runs"]
            )
            runtime_stats = summarize_array(
                data["runtime_runs"]
            )

            best_run_index = int(
                np.argmin(data["rmse_runs"])
            )

            summary_row = {
                "Problem": problem_name,
                "Dimensions": len(parameter_names),
                "Optimizer": optimizer_name,
                "RMSE_Best": rmse_stats["Best"],
                "RMSE_Mean": rmse_stats["Mean"],
                "RMSE_Median": rmse_stats["Median"],
                "RMSE_Std": rmse_stats["Std"],
                "NRMSE_Best": nrmse_stats["Best"],
                "NRMSE_Mean": nrmse_stats["Mean"],
                "AE_Mean": ae_stats["Mean"],
                "MAE_Mean": mae_stats["Mean"],
                "Runtime_Mean": runtime_stats["Mean"],
                "Best_Run": best_run_index + 1,
            }

            best_solution = data[
                "best_solutions"
            ][best_run_index]

            for parameter_name, parameter_value in zip(
                parameter_names,
                best_solution,
            ):
                summary_row[
                    f"Best_{parameter_name}"
                ] = float(parameter_value)

            summary_rows.append(summary_row)

            for run_index in range(
                len(data["rmse_runs"])
            ):
                run_row = {
                    "Problem": problem_name,
                    "Dimensions": len(parameter_names),
                    "Optimizer": optimizer_name,
                    "Run": run_index + 1,
                    "RMSE": float(
                        data["rmse_runs"][run_index]
                    ),
                    "NRMSE": float(
                        data["nrmse_runs"][run_index]
                    ),
                    "AE": float(
                        data["ae_runs"][run_index]
                    ),
                    "MAE": float(
                        data["mae_runs"][run_index]
                    ),
                    "Runtime": float(
                        data["runtime_runs"][run_index]
                    ),
                }

                solution = data[
                    "best_solutions"
                ][run_index]

                for parameter_name, parameter_value in zip(
                    parameter_names,
                    solution,
                ):
                    run_row[parameter_name] = float(
                        parameter_value
                    )

                run_rows.append(run_row)

    summary_df = pd.DataFrame(summary_rows)
    runs_df = pd.DataFrame(run_rows)

    write_excel_with_parameter_formats(
        summary_df,
        summary_path,
    )

    write_excel_with_parameter_formats(
        runs_df,
        runs_path,
    )

    return summary_df, runs_df


def main() -> None:
    args = parse_args()

    logging.disable(logging.INFO)

    paths = make_paths(args)

    cache_signature = build_cache_signature(args)

    selected_problems = list(args.problems)

    print("=" * 65)
    print("SOLAR-CELL PARAMETER ESTIMATION FRAMEWORK")
    print("=" * 65)
    print(f"Experiment      : {paths.exp_tag}")
    print(f"Problems        : {selected_problems}")
    print(f"Optimizers      : {args.optimizers}")
    print(f"Epochs          : {args.epochs}")
    print(f"Population      : {args.pop_size}")
    print(f"Runs            : {args.runs}")
    print(f"Parallel        : {args.parallel}")
    print(f"Workers         : {args.n_workers}")
    print(f"Extra scale     : {args.convergence_extra_scale}")
    print(f"Cache signature : {cache_signature}")

    results_struct = {}

    for problem_index, problem_name in enumerate(
        selected_problems,
        start=1,
    ):
        print("\n" + "=" * 65)
        print(
            f"PROBLEM {problem_index}/{len(selected_problems)}: "
            f"{problem_name}",
            flush=True,
        )
        print("=" * 65)

        results_struct[problem_name] = {}
        convergence_data = {}

        for optimizer_index, optimizer_name in enumerate(
            args.optimizers,
            start=1,
        ):
            try:
                print_status(
                    f"OPTIMIZER "
                    f"{optimizer_index}/{len(args.optimizers)} | "
                    f"problem={problem_name} | "
                    f"optimizer={optimizer_name}"
                )

                completed = []
                pending_runs = []
                checkpoint_records = {}

                for run in range(args.runs):
                    seed = args.seed_base + run

                    checkpoint_path = run_checkpoint_path(
                        paths,
                        cache_signature,
                        problem_name,
                        optimizer_name,
                        run,
                    )

                    metadata = checkpoint_metadata(
                        args,
                        cache_signature,
                        problem_name,
                        optimizer_name,
                        run,
                        seed,
                    )

                    checkpoint_records[run] = (
                        checkpoint_path,
                        metadata,
                    )

                    cached_output = None

                    if args.reuse_cache:
                        cached_output = load_run_checkpoint(
                            checkpoint_path,
                            metadata,
                        )

                    if cached_output is None:
                        pending_runs.append(run)

                    else:
                        completed.append(
                            (run, cached_output)
                        )

                        print_status(
                            f"CACHE HIT | "
                            f"problem={problem_name} | "
                            f"optimizer={optimizer_name} | "
                            f"run={run + 1}/{args.runs}"
                        )

                if len(pending_runs) == 0:
                    print_status(
                        f"CACHE COMPLETE | "
                        f"problem={problem_name} | "
                        f"optimizer={optimizer_name} | "
                        f"runs={args.runs}/{args.runs}"
                    )

                if (
                    args.parallel == "yes"
                    and len(pending_runs) > 1
                ):
                    tasks = []

                    for run in pending_runs:
                        checkpoint_path, metadata = (
                            checkpoint_records[run]
                        )

                        tasks.append({
                            "run": run,
                            "problem_name": problem_name,
                            "optimizer_name": optimizer_name,
                            "args": args,
                            "seed": args.seed_base + run,
                            "total_runs": args.runs,
                            "checkpoint_path": checkpoint_path,
                            "metadata": metadata,
                        })

                    print_status(
                        f"SUBMITTED | "
                        f"problem={problem_name} | "
                        f"optimizer={optimizer_name} | "
                        f"runs={len(tasks)} | "
                        f"workers={args.n_workers}"
                    )

                    with ProcessPoolExecutor(
                        max_workers=args.n_workers
                    ) as executor:
                        futures = [
                            executor.submit(
                                run_parallel_task,
                                task,
                            )
                            for task in tasks
                        ]

                        for future in as_completed(futures):
                            completed.append(
                                future.result()
                            )

                            print_status(
                                f"PROGRESS | "
                                f"problem={problem_name} | "
                                f"optimizer={optimizer_name} | "
                                f"completed_runs="
                                f"{len(completed)}/{args.runs}"
                            )

                else:
                    for run in pending_runs:
                        checkpoint_path, metadata = (
                            checkpoint_records[run]
                        )

                        output = run_single(
                            problem_name,
                            optimizer_name,
                            args,
                            args.seed_base + run,
                            run,
                            args.runs,
                        )

                        save_run_checkpoint(
                            checkpoint_path,
                            metadata,
                            output,
                        )

                        print_status(
                            f"CHECKPOINT SAVED | "
                            f"problem={problem_name} | "
                            f"optimizer={optimizer_name} | "
                            f"run={run + 1}/{args.runs}"
                        )

                        completed.append(
                            (run, output)
                        )

                completed = sorted(
                    completed,
                    key=lambda item: item[0],
                )

                rmse_runs = []
                nrmse_runs = []
                ae_runs = []
                mae_runs = []
                runtime_runs = []
                best_solutions = []
                curves = []
                curves_by_run = {}

                for run, output in completed:
                    metrics = output["metrics"]
                    curve = validate_convergence_curve(
                        output["curve"],
                        args.epochs,
                        optimizer_name,
                        run + 1,
                        "before_mean",
                        best_fitness=float(output["best_fitness"]),
                    ).copy()

                    rmse_runs.append(
                        metrics["RMSE"]
                    )
                    nrmse_runs.append(
                        metrics["NRMSE"]
                    )
                    ae_runs.append(
                        metrics["AE"]
                    )
                    mae_runs.append(
                        metrics["MAE"]
                    )
                    runtime_runs.append(
                        output["runtime"]
                    )
                    best_solutions.append(
                        output["best_solution"]
                    )
                    curves.append(
                        np.asarray(
                            curve,
                            dtype=float,
                        ).copy()
                    )
                    curves_by_run[run + 1] = curve.copy()

                    print(
                        f"Run {run + 1:02d} | "
                        f"RMSE = {metrics['RMSE']:.6e} | "
                        f"NRMSE = {metrics['NRMSE']:.6e} | "
                        f"MAE = {metrics['MAE']:.6e} | "
                        f"Time = {output['runtime']:.2f}s"
                    )

                if not curves:
                    raise ValueError(
                        f"No convergence curves found for {optimizer_name}"
                    )

                curve_matrix = np.stack(
                    curves,
                    axis=0,
                ).copy()

                mean_curve = np.mean(
                    curve_matrix,
                    axis=0,
                ).copy()

                convergence_data[optimizer_name] = {
                    "runs": {
                        run_number: run_curve.copy()
                        for run_number, run_curve in curves_by_run.items()
                    },
                    "mean": mean_curve.copy(),
                }

                results_struct[
                    problem_name
                ][optimizer_name] = {
                    "rmse_runs": np.asarray(
                        rmse_runs,
                        dtype=float,
                    ),
                    "nrmse_runs": np.asarray(
                        nrmse_runs,
                        dtype=float,
                    ),
                    "ae_runs": np.asarray(
                        ae_runs,
                        dtype=float,
                    ),
                    "mae_runs": np.asarray(
                        mae_runs,
                        dtype=float,
                    ),
                    "runtime_runs": np.asarray(
                        runtime_runs,
                        dtype=float,
                    ),
                    "best_solutions": np.asarray(
                        best_solutions,
                        dtype=float,
                    ),
                    "curves_by_run": {
                        run_number: run_curve.copy()
                        for run_number, run_curve in curves_by_run.items()
                    },
                    "curves": curve_matrix.copy(),
                    "curve": mean_curve.copy(),
                }

                print("-" * 55)
                print(
                    f"RMSE Mean   : "
                    f"{np.mean(rmse_runs):.6e}"
                )
                print(
                    f"RMSE Median : "
                    f"{np.median(rmse_runs):.6e}"
                )
                print(
                    f"RMSE Std    : "
                    f"{np.std(rmse_runs):.6e}"
                )
                print(
                    f"RMSE Best   : "
                    f"{np.min(rmse_runs):.6e}"
                )
                print("-" * 55)

            except Exception as exc:
                print(
                    f"[SKIPPED OPTIMIZER] "
                    f"{optimizer_name} on {problem_name}"
                )
                print(f"Reason: {exc}")
                continue

        if convergence_data:
            curves_plot = {
                optimizer_name: data["mean"].copy()
                for optimizer_name, data in convergence_data.items()
            }

            save_problem_convergence_plots(
                curves_plot,
                problem_name,
                paths,
                args.convergence_extra_scale,
            )
            plot_fit_check(
                problem_name,
                results_struct[problem_name],
                paths,
            )

    summary_path = os.path.join(
        paths.res_dir,
        f"Global_Results_{paths.exp_tag}.xlsx",
    )

    runs_path = os.path.join(
        paths.res_dir,
        f"Run_Details_{paths.exp_tag}.xlsx",
    )

    export_results(
        results_struct,
        summary_path,
        runs_path,
    )

    convergence_paths = export_convergence_values(
        results_struct,
        paths,
    )

    print("\n" + "=" * 65)
    print("COMPLETED")
    print("=" * 65)
    print(f"Figures : {paths.fig_dir}")
    print(f"Summary : {summary_path}")
    print(f"Runs    : {runs_path}")
    for convergence_path in convergence_paths:
        print(f"Convergence values : {convergence_path}")


if __name__ == "__main__":
    main()
