from __future__ import annotations

from typing import Any

import numpy as np
from mealpy import FloatVar

# These three files will be created in the next stage.
from single_diode import SingleDiode
from double_diode import DoubleDiode
from triple_diode import TripleDiode


PENALTY_VALUE = 1e300


AVAILABLE_SOLAR_PROBLEMS = {
    "SingleDiode": SingleDiode,
    "DoubleDiode": DoubleDiode,
    "TripleDiode": TripleDiode,
}


def get_solar_problem(problem_name: str):
    """
    Creates and returns the selected solar-cell model.
    """
    try:
        problem_class = AVAILABLE_SOLAR_PROBLEMS[problem_name]

    except KeyError as exc:
        available = ", ".join(
            AVAILABLE_SOLAR_PROBLEMS.keys()
        )

        raise ValueError(
            f"Unknown solar-cell problem: {problem_name}. "
            f"Available problems: {available}"
        ) from exc

    return problem_class()


def get_parameter_names(problem_name: str) -> tuple[str, ...]:
    """
    Returns the ordered parameter names for a solar-cell model.
    """
    solar_problem = get_solar_problem(problem_name)

    return tuple(
        solar_problem.parameter_names
    )


def validate_solution(
    solar_problem: Any,
    solution,
) -> np.ndarray:
    """
    Converts a candidate solution into a valid one-dimensional NumPy array.
    """
    candidate = np.asarray(
        solution,
        dtype=float,
    ).reshape(-1)

    if candidate.size != solar_problem.dims:
        raise ValueError(
            f"{solar_problem.name} requires "
            f"{solar_problem.dims} parameters, "
            f"but received {candidate.size}"
        )

    return candidate


def is_within_bounds(
    solar_problem: Any,
    solution,
) -> bool:
    """
    Checks whether every decision variable is inside its search interval.
    """
    candidate = validate_solution(
        solar_problem,
        solution,
    )

    lower_bounds = np.asarray(
        solar_problem.lb,
        dtype=float,
    )

    upper_bounds = np.asarray(
        solar_problem.ub,
        dtype=float,
    )

    return bool(
        np.all(candidate >= lower_bounds)
        and np.all(candidate <= upper_bounds)
    )


def safe_objective(
    solar_problem: Any,
    solution,
) -> float:
    """
    Evaluates RMSE safely.

    Invalid dimensions, out-of-bound solutions, overflows, divisions by zero,
    NaN values, and infinite values receive a large minimization penalty.
    """
    try:
        candidate = validate_solution(
            solar_problem,
            solution,
        )

        if not is_within_bounds(
            solar_problem,
            candidate,
        ):
            return PENALTY_VALUE

        fitness = float(
            solar_problem.evaluate(candidate)
        )

    except (
        ValueError,
        TypeError,
        FloatingPointError,
        OverflowError,
        ZeroDivisionError,
    ):
        return PENALTY_VALUE

    if not np.isfinite(fitness):
        return PENALTY_VALUE

    return fitness


def build_mealpy_problem(
    problem_name: str,
):
    """
    Builds the problem dictionary required by MEALPY.

    Returns
    -------
    mealpy_problem:
        Dictionary passed directly to optimizer.solve().
    solar_problem:
        Solar model instance containing dimensions, bounds, equations,
        parameter names, and metric calculations.
    """
    solar_problem = get_solar_problem(
        problem_name
    )

    lower_bounds = np.asarray(
        solar_problem.lb,
        dtype=float,
    )

    upper_bounds = np.asarray(
        solar_problem.ub,
        dtype=float,
    )

    if lower_bounds.size != solar_problem.dims:
        raise ValueError(
            f"Invalid lower-bound length for "
            f"{solar_problem.name}"
        )

    if upper_bounds.size != solar_problem.dims:
        raise ValueError(
            f"Invalid upper-bound length for "
            f"{solar_problem.name}"
        )

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError(
            f"Every lower bound must be smaller "
            f"than its upper bound in {solar_problem.name}"
        )

    bounds = FloatVar(
        lb=lower_bounds,
        ub=upper_bounds,
        name="parameters",
    )

    def objective_function(solution) -> float:
        return safe_objective(
            solar_problem,
            solution,
        )

    mealpy_problem = {
        "bounds": bounds,
        "minmax": "min",
        "obj_func": objective_function,
        "name": solar_problem.name,
    }

    return mealpy_problem, solar_problem


def evaluate_solution_metrics(
    problem_name: str,
    solution,
) -> dict[str, float]:
    """
    Calculates all final metrics for the best candidate returned by an optimizer.

    Required model output:
        RMSE
        NRMSE
        AE
        MAE
    """
    solar_problem = get_solar_problem(
        problem_name
    )

    candidate = validate_solution(
        solar_problem,
        solution,
    )

    if not is_within_bounds(
        solar_problem,
        candidate,
    ):
        return {
            "RMSE": PENALTY_VALUE,
            "NRMSE": PENALTY_VALUE,
            "AE": PENALTY_VALUE,
            "MAE": PENALTY_VALUE,
        }

    try:
        raw_metrics = solar_problem.calculate_metrics(
            candidate
        )

    except (
        ValueError,
        TypeError,
        FloatingPointError,
        OverflowError,
        ZeroDivisionError,
    ):
        return {
            "RMSE": PENALTY_VALUE,
            "NRMSE": PENALTY_VALUE,
            "AE": PENALTY_VALUE,
            "MAE": PENALTY_VALUE,
        }

    required_metrics = (
        "RMSE",
        "NRMSE",
        "AE",
        "MAE",
    )

    metrics = {}

    for metric_name in required_metrics:
        if metric_name not in raw_metrics:
            raise KeyError(
                f"{solar_problem.name}.calculate_metrics() "
                f"did not return {metric_name}"
            )

        metric_value = float(
            raw_metrics[metric_name]
        )

        if not np.isfinite(metric_value):
            metric_value = PENALTY_VALUE

        metrics[metric_name] = metric_value

    return metrics


def get_fit_series(
    problem_name: str,
    solution,
) -> dict[str, np.ndarray]:
    """
    Returns measured and estimated I-V/P-V series for visual fit checks.
    """
    solar_problem = get_solar_problem(
        problem_name
    )

    candidate = validate_solution(
        solar_problem,
        solution,
    )

    voltage = np.asarray(
        solar_problem.voltage_measured,
        dtype=float,
    )

    measured_current = np.asarray(
        solar_problem.current_measured,
        dtype=float,
    )

    estimated_current = np.asarray(
        solar_problem.predict_current(candidate),
        dtype=float,
    )

    return {
        "voltage": voltage,
        "measured_current": measured_current,
        "estimated_current": estimated_current,
        "measured_power": voltage * measured_current,
        "estimated_power": voltage * estimated_current,
    }
