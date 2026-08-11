from __future__ import annotations

import numpy as np


class SingleDiode:
    """
    Single-diode solar-cell parameter estimation problem.

    Decision variables
    ------------------
    Rs  : Series resistance
    Rsh : Shunt resistance
    Iph : Photogenerated current
    Isd : Diode saturation current
    n   : Diode ideality factor
    """

    name = "SingleDiode"

    parameter_names = (
        "Rs",
        "Rsh",
        "Iph",
        "Isd",
        "n",
    )

    dims = 5

    lb = np.array(
        [
            0.0,
            1e-6,
            0.0,
            1e-12,
            1.0,
        ],
        dtype=float,
    )

    ub = np.array(
        [
            0.5,
            100.0,
            1.0,
            1e-6,
            2.0,
        ],
        dtype=float,
    )

    voltage_measured = np.array(
        [
            -0.2057,
            -0.1291,
            -0.0588,
            0.0057,
            0.0646,
            0.1185,
            0.1678,
            0.2132,
            0.2545,
            0.2924,
            0.3269,
            0.3585,
            0.3873,
            0.4137,
            0.4373,
            0.4590,
            0.4784,
            0.4960,
            0.5119,
            0.5265,
            0.5398,
            0.5521,
            0.5633,
            0.5736,
            0.5833,
            0.5900,
        ],
        dtype=float,
    )

    current_measured = np.array(
        [
            0.7640,
            0.7620,
            0.7605,
            0.7605,
            0.7600,
            0.7590,
            0.7570,
            0.7570,
            0.7555,
            0.7540,
            0.7505,
            0.7465,
            0.7385,
            0.7280,
            0.7065,
            0.6755,
            0.6320,
            0.5730,
            0.4990,
            0.4130,
            0.3165,
            0.2120,
            0.1035,
            -0.0100,
            -0.1230,
            -0.2100,
        ],
        dtype=float,
    )

    boltzmann_constant = 1.380649e-23
    electron_charge = 1.602e-19
    temperature = 306.15

    def evaluate(self, solution) -> float:
        """
        Returns RMSE, which is the optimization objective.
        """
        metrics = self.calculate_metrics(solution)
        return metrics["RMSE"]

    def predict_current(self, solution) -> np.ndarray:
        candidate = np.asarray(
            solution,
            dtype=float,
        ).reshape(-1)

        if candidate.size != self.dims:
            raise ValueError(
                f"{self.name} requires {self.dims} parameters"
            )

        rs, rsh, iph, isd, ideality = candidate

        if rsh <= 0.0 or ideality <= 0.0:
            raise ValueError(
                "Rsh and n must be greater than zero"
            )

        voltage_term = self.voltage_measured + (
            rs * self.current_measured
        )

        exponent = (
            self.electron_charge
            * voltage_term
            / (
                ideality
                * self.boltzmann_constant
                * self.temperature
            )
        )

        exponent = np.clip(
            exponent,
            -700.0,
            700.0,
        )

        diode_current = isd * (
            np.exp(exponent) - 1.0
        )

        shunt_current = voltage_term / rsh

        return (
            iph
            - diode_current
            - shunt_current
        )

    def calculate_metrics(self, solution) -> dict[str, float]:
        """
        Calculates RMSE, normalized RMSE, absolute error and mean absolute error.
        """
        candidate = np.asarray(
            solution,
            dtype=float,
        ).reshape(-1)

        if candidate.size != self.dims:
            raise ValueError(
                f"{self.name} requires {self.dims} parameters"
            )

        rs, rsh, iph, isd, ideality = candidate

        if rsh <= 0.0 or ideality <= 0.0:
            raise ValueError(
                "Rsh and n must be greater than zero"
            )

        measured_current = self.current_measured
        estimated_current = self.predict_current(candidate)

        residuals = (
            measured_current
            - estimated_current
        )

        squared_error = residuals**2

        rmse = float(
            np.sqrt(
                np.mean(squared_error)
            )
        )

        current_range = float(
            np.max(estimated_current)
            - np.min(estimated_current)
        )

        if current_range <= 0.0:
            nrmse = np.inf
        else:
            nrmse = float(
                rmse / current_range
            )

        absolute_errors = np.abs(
            residuals
        )

        ae = float(
            np.sum(
                100.0 * absolute_errors
            )
        )

        mae = float(
            np.mean(
                absolute_errors
            )
        )

        if not all(
            np.isfinite(value)
            for value in (
                rmse,
                nrmse,
                ae,
                mae,
            )
        ):
            raise FloatingPointError(
                "The calculated metrics are not finite"
            )

        return {
            "RMSE": rmse,
            "NRMSE": nrmse,
            "AE": ae,
            "MAE": mae,
        }
