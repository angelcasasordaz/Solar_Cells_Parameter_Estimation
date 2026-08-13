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
from mealpy.swarm_based.DMOA import OriginalDMOA

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

CHART_PALETTE = {
    "DSADE": "#3266ad",
    "DSADE_AWAD": "#d1495b",
    "MaCRO-DE": "#3266ad",
    "DBO": "#00a6a6",
    "OriginalGWO": "#e06c00",
    "OriginalWOA": "#2a9d5c",
    "OriginalCA": "#c44569",
    "OriginalPSO": "#9b59b6",
    "OriginalDE": "#6a4c93",
    "JADE": "#2d6a4f",
    "SADE": "#f4a261",
    "OriginalSHADE": "#264653",
    "OriginalFOX": "#1b9aaa",
    "OriginalRIME": "#e76f51",
    "OriginalBRO": "#577590",
    "OriginalDMOA": "#90be6d",
    "OriginalMGO": "#f9844a",
    "OriginalHHO": "#4d4d4d",
    "OriginalGOA": "#8a5a44",
    "BRO": "#577590",
    "DE": "#6a4c93",
    "DMO": "#90be6d",
    "GWO": "#e06c00",
    "HHO": "#4d4d4d",
    "MFO": "#8a5a44",
    "MGO": "#f9844a",
    "PSO": "#9b59b6",
    "SHADE": "#264653",
    "WOA": "#2a9d5c",
}

class SafeOriginalDMOA(OriginalDMOA):
    """Original DMOA with the zero-division in MEALPY 3.0.2 removed."""

    def evolve(self, epoch):
        cf = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.pop], dtype=float)
        mean_cost = np.mean(fit_list)

        if np.isfinite(mean_cost) and abs(mean_cost) > self.EPSILON:
            fi = np.exp(-fit_list / mean_cost)
        else:
            fi = np.ones(self.pop_size, dtype=float)

        for idx in range(self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = self.generator.choice(list(set(range(self.pop_size)) - {idx, alpha}))
            phi = (self.peep / 2.0) * self.generator.uniform(
                -1.0,
                1.0,
                self.problem.n_dims,
            )
            new_pos = self.pop[alpha].solution + phi * (
                self.pop[alpha].solution - self.pop[k].solution
            )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        sm = np.zeros(self.pop_size, dtype=float)

        for idx in range(self.pop_size):
            k = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
            phi = (self.peep / 2.0) * self.generator.uniform(
                -1.0,
                1.0,
                self.problem.n_dims,
            )
            new_pos = self.pop[idx].solution + phi * (
                self.pop[idx].solution - self.pop[k].solution
            )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)

            denominator = max(
                abs(float(agent.target.fitness)),
                abs(float(self.pop[idx].target.fitness)),
                self.EPSILON,
            )
            sm[idx] = (
                float(agent.target.fitness) - float(self.pop[idx].target.fitness)
            ) / denominator

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        for idx in range(self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.generate_agent()
                self.C[idx] = 0

        new_tau = np.mean(sm)

        for idx in range(self.pop_size):
            m_value = np.full(self.problem.n_dims, sm[idx], dtype=float)
            phi = (self.peep / 2.0) * self.generator.uniform(
                -1.0,
                1.0,
                self.problem.n_dims,
            )

            if new_tau > self.tau:
                new_pos = self.pop[idx].solution - cf * phi * self.generator.random() * (
                    self.pop[idx].solution - m_value
                )
            else:
                new_pos = self.pop[idx].solution + cf * phi * self.generator.random() * (
                    self.pop[idx].solution - m_value
                )

            self.tau = new_tau
            new_pos = self.correct_solution(new_pos)
            self.pop[idx] = self.generate_agent(new_pos)


@dataclass
class Paths:
    exp_tag: str
    fig_dir: str
    res_dir: str
    cache_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar Cells + MEALPY Benchmark Framework")
    parser.add_argument("--exp-id", type=int, default=4, help="Numeric experiment identifier")
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
    optimizer_key = normalize_optimizer_name(name)

    if optimizer_key in ("dmo", "dmoa", "originaldmoa"):
        optimizer_class = SafeOriginalDMOA
        optimizer_kwargs = {
            "epoch": args.epochs,
            "pop_size": args.pop_size,
        }
        return optimizer_class(**optimizer_kwargs)

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

    return output


def print_status(message: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


def normalize_curve_length(
    curve,
    expected_length: int,
) -> np.ndarray:
    curve = np.asarray(curve, dtype=float).reshape(-1)

    if curve.size == expected_length:
        return curve

    if curve.size == 0:
        return np.full(
            expected_length,
            np.nan,
            dtype=float,
        )

    if curve.size > expected_length:
        return curve[:expected_length]

    padding = np.full(
        expected_length - curve.size,
        curve[-1],
        dtype=float,
    )

    return np.concatenate([curve, padding])


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
    result = optimizer.solve(mealpy_problem)
    runtime = time.time() - start_time

    best_solution = np.asarray(
        result.solution,
        dtype=float,
    )

    metrics = evaluate_solution_metrics(
        problem_name,
        best_solution,
    )

    convergence = normalize_curve_length(
        optimizer.history.list_global_best_fit,
        args.epochs,
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
        "curve": convergence,
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

    plot_items = [
        item
        for item in curves_dict.items()
        if item[0] != "MaCRO-DE"
    ]

    if "MaCRO-DE" in curves_dict:
        plot_items.append(
            (
                "MaCRO-DE",
                curves_dict["MaCRO-DE"],
            )
        )

    for optimizer_name, curve in plot_items:
        curve = np.asarray(
            curve,
            dtype=float,
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
            plot_curve = curve

        color = CHART_PALETTE.get(
            optimizer_name,
            None,
        )

        if optimizer_name == "MaCRO-DE":
            ax.plot(
                plot_curve,
                linewidth=4.4,
                label="_nolegend_",
                color="black",
                solid_capstyle="round",
            )

        ax.plot(
            plot_curve,
            linewidth=(
                3.0
                if optimizer_name == "MaCRO-DE"
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

    ordered_items = [
        item
        for item in optimizer_data.items()
        if item[0] != "MaCRO-DE"
    ]

    if "MaCRO-DE" in optimizer_data:
        ordered_items.append(
            (
                "MaCRO-DE",
                optimizer_data["MaCRO-DE"],
            )
        )

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
        curves_plot = {}

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

                for run, output in completed:
                    metrics = output["metrics"]

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
                        output["curve"]
                    )

                    print(
                        f"Run {run + 1:02d} | "
                        f"RMSE = {metrics['RMSE']:.6e} | "
                        f"NRMSE = {metrics['NRMSE']:.6e} | "
                        f"MAE = {metrics['MAE']:.6e} | "
                        f"Time = {output['runtime']:.2f}s"
                    )

                curves_array = np.asarray(
                    curves,
                    dtype=float,
                )

                mean_curve = np.nanmean(
                    curves_array,
                    axis=0,
                )

                curves_plot[
                    optimizer_name
                ] = mean_curve

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
                    "curve": mean_curve,
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

        if curves_plot:
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

    print("\n" + "=" * 65)
    print("COMPLETED")
    print("=" * 65)
    print(f"Figures : {paths.fig_dir}")
    print(f"Summary : {summary_path}")
    print(f"Runs    : {runs_path}")


if __name__ == "__main__":
    main()
