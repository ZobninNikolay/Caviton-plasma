#!/usr/bin/env python3
"""Level-2 electrostatic dispersion model for Langmuir and ion-acoustic waves.

The real frequency and collisionless Landau damping are obtained from complex
roots of the multi-species Maxwellian longitudinal dielectric function. Neutral
drag is added as a separately reported conservative closure.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import atomic_mass as AMU
from scipy.constants import e, epsilon_0, m_e, pi
from scipy.optimize import root
from scipy.special import wofz

from plasma_level1 import debye_length, electron_ion_collision_frequency


@dataclass(frozen=True)
class Species:
    name: str
    density_m3: float
    temperature_eV: float
    mass_kg: float
    charge_state: int
    neutral_collision_s1: float = 0.0

    @property
    def thermal_speed_ms(self) -> float:
        """One-dimensional thermal speed sqrt(T/m) used in zeta."""
        return math.sqrt(e * self.temperature_eV / self.mass_kg)

    @property
    def debye_length_m(self) -> float:
        return math.sqrt(
            epsilon_0 * self.temperature_eV
            / (self.density_m3 * self.charge_state**2 * e)
        )

    @property
    def plasma_frequency_rad_s(self) -> float:
        return math.sqrt(
            self.density_m3 * self.charge_state**2 * e**2
            / (epsilon_0 * self.mass_kg)
        )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def plasma_dispersion(zeta: complex) -> complex:
    return 1j * math.sqrt(pi) * wofz(zeta)


def susceptibility(omega: complex, k_m1: float, species: Species) -> complex:
    zeta = omega / (math.sqrt(2.0) * k_m1 * species.thermal_speed_ms)
    return (1.0 + zeta * plasma_dispersion(zeta)) / (k_m1 * species.debye_length_m) ** 2


def longitudinal_dielectric(omega: complex, k_m1: float, species: Sequence[Species]) -> complex:
    return 1.0 + sum(susceptibility(omega, k_m1, item) for item in species)


def build_species(background: dict, Ti_eV: float, composition: dict) -> tuple[Species, list[Species]]:
    ne = float(background["electron_density_m3"])
    fractions = [float(item["fraction"]) for item in composition["ions"]]
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"Ion fractions in {composition['name']} must sum to one")
    electron = Species("e-", ne, float(background["electron_temperature_eV"]), m_e, -1)
    reference_nu_in = float(background["nu_in_scaled_s-1"])
    ions = [
        Species(
            name=item["name"],
            density_m3=ne * float(item["fraction"]),
            temperature_eV=Ti_eV,
            mass_kg=float(item["mass_u"]) * AMU,
            charge_state=1,
            neutral_collision_s1=reference_nu_in * float(item["neutral_collision_scale"]),
        )
        for item in composition["ions"]
    ]
    return electron, ions


def solve_scaled_root(
    dielectric: Callable[[complex], complex],
    omega_scale: float,
    guesses: Iterable[complex],
    tolerance: float,
    residual_tolerance: float,
) -> tuple[complex, float]:
    candidates: list[tuple[float, complex]] = []
    for guess in guesses:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            solution = root(
                lambda pair: (
                    dielectric(omega_scale * complex(pair[0], pair[1])).real,
                    dielectric(omega_scale * complex(pair[0], pair[1])).imag,
                ),
                [guess.real, guess.imag],
                method="hybr",
                tol=tolerance,
            )
        omega = omega_scale * complex(solution.x[0], solution.x[1])
        if not solution.success or not np.isfinite(omega.real) or not np.isfinite(omega.imag):
            continue
        residual = abs(dielectric(omega))
        if omega.real > 0.0 and omega.imag <= 0.0 and residual <= residual_tolerance:
            candidates.append((residual, omega))
    if not candidates:
        raise RuntimeError("Complex dispersion root did not converge")
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1], candidates[0][0]


def langmuir_initial(k_m1: float, electron: Species) -> float:
    return math.sqrt(
        electron.plasma_frequency_rad_s**2
        + 3.0 * k_m1**2 * electron.thermal_speed_ms**2
    )


def langmuir_long_wavelength_asymptotic(
    k_m1: float,
    electron: Species,
) -> tuple[complex, float]:
    """Stable Maxwellian Langmuir root for k*lambda_De << 1.

    Direct evaluation of 1 + zeta*Z(zeta) loses relative precision for the
    very large electron zeta encountered in the extrema search. The retained
    real-frequency expansion is O(kappa^4); the exponentially small Landau
    decrement is the matching large-zeta result. The returned second value is
    a truncation indicator rather than a directly evaluated dielectric
    residual.
    """
    kappa = k_m1 * electron.debye_length_m
    omega_pe = electron.plasma_frequency_rad_s
    omega_real = omega_pe * math.sqrt(1.0 + 3.0 * kappa**2 + 6.0 * kappa**4)
    gamma = -math.sqrt(pi / 8.0) * omega_pe / kappa**3 * math.exp(
        -1.0 / (2.0 * kappa**2) - 1.5
    )
    return complex(omega_real, gamma), kappa**6


def fast_acoustic_initial(k_m1: float, electron: Species, ions: Sequence[Species]) -> float:
    ne = electron.density_m3
    harmonic_mass_factor = sum(item.density_m3 / ne / item.mass_kg for item in ions)
    mean_ti = sum(item.density_m3 / ne * item.temperature_eV for item in ions)
    kappa = k_m1 * electron.debye_length_m
    speed2 = e * (electron.temperature_eV / (1.0 + kappa**2) + 3.0 * mean_ti) * harmonic_mass_factor
    return k_m1 * math.sqrt(speed2)


def solve_kinetic_mode(
    mode: str,
    k_m1: float,
    electron: Species,
    ions: Sequence[Species],
    solver_cfg: dict,
) -> tuple[complex, float]:
    species = [electron, *ions]
    dielectric = lambda omega: longitudinal_dielectric(omega, k_m1, species)
    if mode == "langmuir":
        if k_m1 * electron.debye_length_m < 0.04:
            return langmuir_long_wavelength_asymptotic(k_m1, electron)
        scale = langmuir_initial(k_m1, electron)
        guesses = [complex(1.0, damping) for damping in (-1.0e-5, -1.0e-4, -1.0e-3, -1.0e-2)]
    elif mode == "ion_acoustic_fast":
        scale = fast_acoustic_initial(k_m1, electron, ions)
        guesses = [
            complex(frequency_scale, damping)
            for frequency_scale in (0.75, 0.9, 1.0, 1.1, 1.3)
            for damping in (-1.0e-3, -1.0e-2, -5.0e-2, -1.5e-1)
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return solve_scaled_root(
        dielectric,
        scale,
        guesses,
        float(solver_cfg["root_relative_tolerance"]),
        float(solver_cfg["dielectric_residual_tolerance"]),
    )


def find_least_damped_slow_candidate(
    background: dict,
    Ti_eV: float,
    kappa: float,
    composition: dict,
    solver_cfg: dict,
) -> dict | None:
    """Search the low-phase-speed half-plane for a distinct multi-ion root.

    Strongly damped Vlasov roots are numerous. This diagnostic returns only the
    least-damped root below 90% of the fast-mode phase velocity and does not
    promote it to a propagating branch unless it passes the configured Q test.
    """
    if len(composition["ions"]) < 2:
        return None
    electron, ions = build_species(background, Ti_eV, composition)
    k_m1 = kappa / electron.debye_length_m
    species = [electron, *ions]
    dielectric = lambda omega: longitudinal_dielectric(omega, k_m1, species)
    fast_root, _ = solve_kinetic_mode("ion_acoustic_fast", k_m1, electron, ions, solver_cfg)
    fast_phase = fast_root.real / k_m1
    min_thermal = min(item.thermal_speed_ms for item in ions)
    phase_guesses = np.geomspace(max(200.0, 0.35 * min_thermal), 0.9 * fast_phase, 36)
    roots: list[tuple[complex, float]] = []
    for phase_guess in phase_guesses:
        omega_scale = k_m1 * phase_guess
        for damping in (-0.02, -0.08, -0.20, -0.40, -0.70, -1.0):
            try:
                candidate, residual = solve_scaled_root(
                    dielectric,
                    omega_scale,
                    [complex(1.0, damping)],
                    float(solver_cfg["root_relative_tolerance"]),
                    float(solver_cfg["dielectric_residual_tolerance"]),
                )
            except RuntimeError:
                continue
            phase = candidate.real / k_m1
            if not (0.25 * min_thermal <= phase <= 0.9 * fast_phase):
                continue
            if any(abs(candidate - prior) / abs(prior) < 1.0e-3 for prior, _ in roots):
                continue
            roots.append((candidate, residual))
    if not roots:
        return None
    gamma_collision, meta = collision_correction("ion_acoustic_fast", kappa, background, electron, ions)
    roots.sort(key=lambda pair: pair[0].real / (2.0 * abs(pair[0].imag + gamma_collision)), reverse=True)
    candidate, residual = roots[0]
    gamma_total = candidate.imag + gamma_collision
    quality = candidate.real / (2.0 * abs(gamma_total))
    return {
        "mode": "ion_acoustic_slow_candidate",
        "composition": composition["name"],
        "ion_fractions": ";".join(f"{item.name}:{item.density_m3/electron.density_m3:.3f}" for item in ions),
        "pressure_Pa": float(background["pressure_Pa"]),
        "absorbed_power_W": float(background["absorbed_power_W"]),
        "electron_density_m3": electron.density_m3,
        "electron_temperature_eV": electron.temperature_eV,
        "ion_temperature_eV": Ti_eV,
        "k_lambda_De": kappa,
        "k_m-1": k_m1,
        "wavelength_m": 2.0 * pi / k_m1,
        "frequency_Hz": candidate.real / (2.0 * pi),
        "phase_velocity_ms": candidate.real / k_m1,
        "gamma_kinetic_s-1": candidate.imag,
        "gamma_collision_s-1": gamma_collision,
        "gamma_total_s-1": gamma_total,
        "quality_factor": quality,
        "dielectric_residual": residual,
        "propagating_Q_ge_threshold": quality >= float(solver_cfg["propagating_min_Q"]),
        **meta,
    }


def collision_correction(
    mode: str,
    kappa: float,
    background: dict,
    electron: Species,
    ions: Sequence[Species],
) -> tuple[float, dict]:
    nu_en = float(background["nu_en_scaled_s-1"])
    nu_ei = electron_ion_collision_frequency(electron.density_m3, electron.temperature_eV)
    if mode == "langmuir":
        gamma = -0.5 * (nu_en + nu_ei)
        return gamma, {"nu_en_s-1": nu_en, "nu_ei_s-1": nu_ei, "nu_in_effective_s-1": 0.0}
    weights_raw = np.array([item.density_m3 / item.mass_kg for item in ions])
    weights = weights_raw / weights_raw.sum()
    nu_in_effective = float(sum(weight * item.neutral_collision_s1 for weight, item in zip(weights, ions)))
    electron_polarization_weight = kappa**2 / (1.0 + kappa**2)
    gamma = -0.5 * (nu_in_effective + electron_polarization_weight * nu_en)
    return gamma, {
        "nu_en_s-1": nu_en,
        "nu_ei_s-1": nu_ei,
        "nu_in_effective_s-1": nu_in_effective,
    }


def solve_mode_point(
    mode: str,
    background: dict,
    Ti_eV: float,
    kappa: float,
    composition: dict,
    solver_cfg: dict,
    calculate_group_velocity: bool = True,
) -> dict:
    electron, ions = build_species(background, Ti_eV, composition)
    k_m1 = kappa / electron.debye_length_m
    kinetic_root, residual = solve_kinetic_mode(mode, k_m1, electron, ions, solver_cfg)
    gamma_collision, collision_meta = collision_correction(mode, kappa, background, electron, ions)
    gamma_total = kinetic_root.imag + gamma_collision

    if calculate_group_velocity:
        delta = float(solver_cfg["group_derivative_fraction"])
        roots = []
        for factor in (1.0 - delta, 1.0 + delta):
            shifted_root, _ = solve_kinetic_mode(mode, k_m1 * factor, electron, ions, solver_cfg)
            roots.append(shifted_root.real)
        group_velocity = (roots[1] - roots[0]) / (2.0 * delta * k_m1)
    else:
        group_velocity = float("nan")

    phase_velocity = kinetic_root.real / k_m1
    wavelength = 2.0 * pi / k_m1
    damping_time = 1.0 / abs(gamma_total)
    damping_length_amplitude = abs(group_velocity) * damping_time
    damping_length_energy = 0.5 * damping_length_amplitude
    quality = kinetic_root.real / (2.0 * abs(gamma_total))
    propagating = quality >= float(solver_cfg["propagating_min_Q"])
    fractions = ";".join(f"{item.name}:{item.density_m3/electron.density_m3:.3f}" for item in ions)
    return {
        "mode": mode,
        "composition": composition["name"],
        "ion_fractions": fractions,
        "pressure_Pa": float(background["pressure_Pa"]),
        "absorbed_power_W": float(background["absorbed_power_W"]),
        "electron_density_m3": electron.density_m3,
        "electron_temperature_eV": electron.temperature_eV,
        "ion_temperature_eV": Ti_eV,
        "lambda_De_m": electron.debye_length_m,
        "k_lambda_De": kappa,
        "k_m-1": k_m1,
        "wavelength_m": wavelength,
        "omega_real_rad_s": kinetic_root.real,
        "frequency_Hz": kinetic_root.real / (2.0 * pi),
        "phase_velocity_ms": phase_velocity,
        "group_velocity_ms": group_velocity,
        "gamma_kinetic_s-1": kinetic_root.imag,
        "gamma_collision_s-1": gamma_collision,
        "gamma_total_s-1": gamma_total,
        "damping_time_s": damping_time,
        "L_damp_amplitude_m": damping_length_amplitude,
        "L_damp_energy_m": damping_length_energy,
        "quality_factor": quality,
        "dielectric_residual": residual,
        "propagating_Q_ge_threshold": propagating,
        **collision_meta,
    }


def load_backgrounds(root_dir: Path, cfg: dict) -> pd.DataFrame:
    path = root_dir / cfg["input"]["background_csv"]
    frame = pd.read_csv(path)
    frame = frame[frame.scenario == cfg["input"]["background_scenario"]].copy()
    frame = frame.sort_values(["pressure_Pa", "absorbed_power_W"])
    return frame.drop_duplicates(["pressure_Pa", "absorbed_power_W"])


def select_background(backgrounds: pd.DataFrame, pressure: float, power: float) -> dict:
    mask = np.isclose(backgrounds.pressure_Pa, pressure) & np.isclose(backgrounds.absorbed_power_W, power)
    matches = backgrounds[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected one background for p={pressure}, Pabs={power}; found {len(matches)}")
    return matches.iloc[0].to_dict()


def calculate_core(root_dir: Path, cfg: dict) -> pd.DataFrame:
    backgrounds = load_backgrounds(root_dir, cfg)
    core = cfg["core_window"]
    rows = []
    for pressure in core["pressure_Pa"]:
        for power in core["absorbed_power_W"]:
            background = select_background(backgrounds, pressure, power)
            for Ti in core["ion_temperature_eV"]:
                for kappa in core["k_lambda_De"]:
                    for composition in cfg["composition_scenarios"]:
                        for mode in ("langmuir", "ion_acoustic_fast"):
                            rows.append(solve_mode_point(mode, background, Ti, kappa, composition, cfg["solver"]))
    frame = pd.DataFrame(rows)
    acceptance = cfg["acceptance"]
    frame["required_spatial_length_m"] = np.where(
        frame["mode"] == "langmuir",
        float(acceptance["langmuir_min_amplitude_length_m"]),
        float(acceptance["acoustic_min_amplitude_length_in_wavelengths"]) * frame["wavelength_m"],
    )
    frame["pass_spatial_length"] = frame.L_damp_amplitude_m >= frame.required_spatial_length_m
    frame["level2_mode_accepted"] = frame.propagating_Q_ge_threshold & frame.pass_spatial_length
    return frame


def calculate_final_regimes(core: pd.DataFrame) -> pd.DataFrame:
    keys = ["pressure_Pa", "absorbed_power_W", "ion_temperature_eV", "k_lambda_De"]
    return core.groupby(keys, as_index=False).agg(
        mode_rows=("mode", "size"),
        compositions=("composition", "nunique"),
        all_modes_accepted=("level2_mode_accepted", "all"),
        minimum_length_margin=("L_damp_amplitude_m", "min"),
        minimum_quality=("quality_factor", "min"),
        maximum_dielectric_residual=("dielectric_residual", "max"),
    )


def calculate_center_curves(root_dir: Path, cfg: dict) -> pd.DataFrame:
    backgrounds = load_backgrounds(root_dir, cfg)
    center = cfg["center_point"]
    background = select_background(backgrounds, center["pressure_Pa"], center["absorbed_power_W"])
    start, stop, count = center["curve_k_lambda_De"]
    rows = []
    for kappa in np.linspace(start, stop, int(count)):
        for composition in cfg["composition_scenarios"]:
            for mode in ("langmuir", "ion_acoustic_fast"):
                rows.append(
                    solve_mode_point(
                        mode,
                        background,
                        center["ion_temperature_eV"],
                        float(kappa),
                        composition,
                        cfg["solver"],
                    )
                )
    return pd.DataFrame(rows)


def calculate_slow_diagnostic(root_dir: Path, cfg: dict) -> pd.DataFrame:
    backgrounds = load_backgrounds(root_dir, cfg)
    center = cfg["center_point"]
    background = select_background(backgrounds, center["pressure_Pa"], center["absorbed_power_W"])
    rows = []
    for composition in cfg["composition_scenarios"]:
        result = find_least_damped_slow_candidate(
            background,
            center["ion_temperature_eV"],
            center["k_lambda_De"],
            composition,
            cfg["solver"],
        )
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def summarize(core: pd.DataFrame, curves: pd.DataFrame, slow: pd.DataFrame, final_regimes: pd.DataFrame, cfg: dict) -> dict:
    def interval(frame: pd.DataFrame, column: str) -> list[float]:
        return [float(frame[column].min()), float(frame[column].max())]

    def mode_ranges(frame: pd.DataFrame) -> dict:
        compositions = {}
        for composition, group in frame.groupby("composition"):
            compositions[composition] = {}
            for mode, mode_group in group.groupby("mode"):
                compositions[composition][mode] = {
                    "frequency_Hz": interval(mode_group, "frequency_Hz"),
                    "wavelength_m": interval(mode_group, "wavelength_m"),
                    "phase_velocity_ms": interval(mode_group, "phase_velocity_ms"),
                    "group_velocity_ms": interval(mode_group, "group_velocity_ms"),
                    "gamma_kinetic_s-1": interval(mode_group, "gamma_kinetic_s-1"),
                    "gamma_collision_s-1": interval(mode_group, "gamma_collision_s-1"),
                    "gamma_total_s-1": interval(mode_group, "gamma_total_s-1"),
                    "L_damp_amplitude_m": interval(mode_group, "L_damp_amplitude_m"),
                    "quality_factor": interval(mode_group, "quality_factor"),
                    "max_dielectric_residual": float(mode_group.dielectric_residual.max()),
                    "accepted_fraction": float(mode_group.level2_mode_accepted.mean()),
                }
        return compositions
    center = cfg["center_point"]
    mask = (
        np.isclose(core.pressure_Pa, center["pressure_Pa"])
        & np.isclose(core.absorbed_power_W, center["absorbed_power_W"])
        & np.isclose(core.ion_temperature_eV, center["ion_temperature_eV"])
        & np.isclose(core.k_lambda_De, center["k_lambda_De"])
    )
    center_rows = core[mask].to_dict("records")
    accepted = final_regimes[final_regimes.all_modes_accepted]
    accepted_mode_rows = core.merge(
        accepted[["pressure_Pa", "absorbed_power_W", "ion_temperature_eV", "k_lambda_De"]],
        on=["pressure_Pa", "absorbed_power_W", "ion_temperature_eV", "k_lambda_De"],
    )

    def accepted_interval(column: str) -> list[float] | None:
        if accepted.empty:
            return None
        return [float(accepted[column].min()), float(accepted[column].max())]

    return {
        "model_scope": "unmagnetized electrostatic Maxwellian kinetic roots plus separately reported neutral-drag closure",
        "counts": {
            "core_mode_rows": int(len(core)),
            "core_parameter_combinations": int(len(final_regimes)),
            "final_accepted_combinations": int(len(accepted)),
            "center_curve_rows": int(len(curves)),
        },
        "final_accepted_projection": {
            "pressure_Pa": accepted_interval("pressure_Pa"),
            "absorbed_power_W": accepted_interval("absorbed_power_W"),
            "ion_temperature_eV": accepted_interval("ion_temperature_eV"),
            "k_lambda_De": accepted_interval("k_lambda_De"),
        },
        "core_ranges_by_composition": mode_ranges(core),
        "final_ranges_by_composition": mode_ranges(accepted_mode_rows),
        "center_point_rows": center_rows,
        "slow_mode_diagnostic": slow.to_dict("records"),
    }


def make_figures(curves: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"pure_Dplus": "#1f77b4", "molecular_nominal": "#d62728", "D3plus_dominant": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for composition, group in curves.groupby("composition"):
        langmuir = group[group["mode"] == "langmuir"]
        acoustic = group[group["mode"] == "ion_acoustic_fast"]
        axes[0].plot(langmuir.k_lambda_De, langmuir.frequency_Hz / 1e9, label=composition, color=colors[composition])
        axes[1].plot(acoustic.k_lambda_De, acoustic.frequency_Hz / 1e6, label=composition, color=colors[composition])
    axes[0].set(xlabel=r"$k\lambda_{De}$", ylabel="Частота, ГГц", title="Ленгмюровская ветвь")
    axes[1].set(xlabel=r"$k\lambda_{De}$", ylabel="Частота, МГц", title="Быстрая ионно-звуковая ветвь")
    axes[1].legend(title="Ионный состав")
    fig.suptitle("Комплексное дисперсионное соотношение в центральной точке")
    fig.tight_layout()
    fig.savefig(output_dir / "07_level2_dispersion_frequencies.png", dpi=180)
    plt.close(fig)

    acoustic = curves[curves["mode"] == "ion_acoustic_fast"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for composition, group in acoustic.groupby("composition"):
        axes[0].plot(group.k_lambda_De, group.phase_velocity_ms / 1e3, "--", color=colors[composition], alpha=0.75)
        axes[0].plot(group.k_lambda_De, group.group_velocity_ms / 1e3, label=composition, color=colors[composition])
        axes[1].plot(group.k_lambda_De, group.L_damp_amplitude_m * 1e3, label=composition, color=colors[composition])
    axes[0].set(xlabel=r"$k\lambda_{De}$", ylabel="Скорость, км/с", title="Фазовая (штрих) и групповая (линия)")
    axes[1].set(xlabel=r"$k\lambda_{De}$", ylabel="Амплитудная длина, мм", title="Полное затухание")
    axes[1].legend(title="Ионный состав")
    fig.suptitle("Ионно-звуковая ветвь: перенос и затухание")
    fig.tight_layout()
    fig.savefig(output_dir / "08_level2_acoustic_transport.png", dpi=180)
    plt.close(fig)

    molecular = curves[curves.composition == "molecular_nominal"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for mode, group in molecular.groupby("mode"):
        label = "Ленгмюровская" if mode == "langmuir" else "Ионно-звуковая"
        ax.semilogy(group.k_lambda_De, group.quality_factor, label=label)
    ax.set(xlabel=r"$k\lambda_{De}$", ylabel="Добротность Q", title="Добротность ветвей: молекулярный состав")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "09_level2_quality.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_level2.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    core = calculate_core(root_dir, cfg)
    curves = calculate_center_curves(root_dir, cfg)
    slow = calculate_slow_diagnostic(root_dir, cfg)
    final_regimes = calculate_final_regimes(core)
    summary = summarize(core, curves, slow, final_regimes, cfg)
    core.to_csv(args.output / "level2_dispersion_core.csv", index=False)
    curves.to_csv(args.output / "level2_center_curves.csv", index=False)
    slow.to_csv(args.output / "level2_slow_mode_diagnostic.csv", index=False)
    final_regimes.to_csv(args.output / "level2_final_regimes.csv", index=False)
    final_regimes[final_regimes.all_modes_accepted].to_csv(args.output / "level2_final_regimes_accepted.csv", index=False)
    (args.output / "level2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(curves, root_dir / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
