#!/usr/bin/env python3
"""Level-1 global and wave-survival model for a low-pressure D2 plasma.

The package is deliberately a screening model. Cross-section CSV files are
replaceable inputs, and every phenomenological closure is exposed in config.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import Boltzmann as KB
from scipy.constants import atomic_mass as AMU
from scipy.constants import e, epsilon_0, m_e, pi
from scipy.optimize import brentq, root
from scipy.special import wofz


@dataclass(frozen=True)
class Geometry:
    radius_m: float
    length_m: float
    interaction_length_m: float
    wall_transmission_factor: float

    @property
    def volume_m3(self) -> float:
        return pi * self.radius_m**2 * self.length_m

    @property
    def area_m2(self) -> float:
        return 2 * pi * self.radius_m * self.length_m + 2 * pi * self.radius_m**2

    @property
    def inverse_diffusion_length2_m2(self) -> float:
        return (2.405 / self.radius_m) ** 2 + (pi / self.length_m) ** 2


@dataclass(frozen=True)
class Gas:
    neutral_temperature_K: float
    ion_mass_kg: float
    neutral_mass_kg: float


class CrossSection:
    """Tabulated cross section with linear interpolation and zero-safe ends."""

    def __init__(self, path: Path):
        frame = pd.read_csv(path, comment="#")
        required = {"energy_eV", "sigma_m2"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: required columns are {sorted(required)}")
        frame = frame.sort_values("energy_eV").drop_duplicates("energy_eV")
        self.energy_eV = frame.energy_eV.to_numpy(float)
        self.sigma_m2 = frame.sigma_m2.to_numpy(float)
        self.path = path

    def __call__(self, energy_eV: np.ndarray | float) -> np.ndarray:
        x = np.asarray(energy_eV, dtype=float)
        return np.interp(x, self.energy_eV, self.sigma_m2, left=self.sigma_m2[0], right=self.sigma_m2[-1])

    def maxwell_rate(self, temperature_eV: float, mass_kg: float = m_e) -> float:
        # Maxwellian energy probability density is normalized in eV^-1.
        lo = max(1.0e-4, self.energy_eV[0])
        hi = max(self.energy_eV[-1], 40.0 * temperature_eV)
        energy = np.geomspace(lo, hi, 5000)
        pdf = 2.0 / math.sqrt(pi) * np.sqrt(energy) / temperature_eV**1.5 * np.exp(-energy / temperature_eV)
        speed = np.sqrt(2.0 * e * energy / mass_kg)
        return float(np.trapezoid(self(energy) * speed * pdf, energy))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_cross_sections(root: Path, cfg: dict | None = None) -> Dict[str, CrossSection]:
    if cfg is None:
        cfg = load_config(root / "config.yaml")
    return {name: CrossSection(root / relative_path) for name, relative_path in cfg["cross_sections"].items()}


def neutral_density(pressure_Pa: float, neutral_temperature_K: float) -> float:
    return pressure_Pa / (KB * neutral_temperature_K)


def debye_length(ne_m3: float, Te_eV: float) -> float:
    return math.sqrt(epsilon_0 * Te_eV / (ne_m3 * e))


def plasma_parameter(ne_m3: float, lambda_De_m: float) -> float:
    return 4.0 * pi * ne_m3 * lambda_De_m**3 / 3.0


def electron_ion_collision_frequency(ne_m3: float, Te_eV: float, Z: float = 1.0) -> float:
    # NRL formulary fit, converted from density in cm^-3; Coulomb logarithm is bounded for screening use.
    ne_cm3 = ne_m3 * 1.0e-6
    ln_lambda = max(2.0, 23.0 - math.log(math.sqrt(ne_cm3) * Z / Te_eV**1.5))
    return 2.91e-6 * ne_cm3 * Z * ln_lambda / Te_eV**1.5


def ion_neutral_rate(
    nn_m3: float,
    Te_eV: float,
    Ti_eV: float,
    gas: Gas,
    sigma: CrossSection,
    sigma_scale: float = 1.0,
) -> Tuple[float, float, float]:
    cs = math.sqrt(e * max(Te_eV + 3.0 * Ti_eV, 1.0e-6) / gas.ion_mass_kg)
    v_thermal_i = math.sqrt(8.0 * e * Ti_eV / (pi * gas.ion_mass_kg))
    v_thermal_n = math.sqrt(8.0 * KB * gas.neutral_temperature_K / (pi * gas.neutral_mass_kg))
    v_rel = math.sqrt(cs**2 + v_thermal_i**2 + v_thermal_n**2)
    reduced_mass = gas.ion_mass_kg * gas.neutral_mass_kg / (gas.ion_mass_kg + gas.neutral_mass_kg)
    relative_energy_eV = 0.5 * reduced_mass * v_rel**2 / e
    rate_coefficient = float(sigma(relative_energy_eV)) * v_rel * sigma_scale
    return nn_m3 * rate_coefficient, relative_energy_eV, rate_coefficient


def loss_frequency(
    pressure_Pa: float,
    Te_eV: float,
    Ti_eV: float,
    geometry: Geometry,
    gas: Gas,
    xs: Dict[str, CrossSection],
    ion_sigma_scale: float = 1.0,
) -> Tuple[float, dict]:
    nn = neutral_density(pressure_Pa, gas.neutral_temperature_K)
    nu_in, relative_energy_eV, kin = ion_neutral_rate(nn, Te_eV, Ti_eV, gas, xs["ion_momentum"], ion_sigma_scale)
    cs = math.sqrt(e * (Te_eV + 3.0 * Ti_eV) / gas.ion_mass_kg)
    nu_bohm = geometry.wall_transmission_factor * cs * geometry.area_m2 / geometry.volume_m3
    Da = cs**2 / max(nu_in, cs / min(geometry.radius_m, geometry.length_m))
    nu_diff = Da * geometry.inverse_diffusion_length2_m2
    # Transport and sheath loss act in series; the slower stage limits wall loss.
    nu_loss = min(nu_bohm, nu_diff)
    return nu_loss, {
        "nu_in_s-1": nu_in,
        "nu_bohm_s-1": nu_bohm,
        "nu_diff_s-1": nu_diff,
        "ion_relative_energy_eV": relative_energy_eV,
        "K_in_m3s": kin,
        "sound_speed_ms": cs,
    }


def particle_balance_residual(
    Te_eV: float,
    pressure_Pa: float,
    Ti_eV: float,
    geometry: Geometry,
    gas: Gas,
    xs: Dict[str, CrossSection],
    ion_sigma_scale: float = 1.0,
) -> float:
    nn = neutral_density(pressure_Pa, gas.neutral_temperature_K)
    production = nn * xs["ionization"].maxwell_rate(Te_eV)
    losses, _ = loss_frequency(pressure_Pa, Te_eV, Ti_eV, geometry, gas, xs, ion_sigma_scale)
    return math.log(max(production, 1.0e-300) / max(losses, 1.0e-300))


def solve_background(
    pressure_Pa: float,
    absorbed_power_W: float,
    Ti_seed_eV: float,
    geometry: Geometry,
    gas: Gas,
    xs: Dict[str, CrossSection],
    cfg: dict,
    ion_sigma_scale: float = 1.0,
) -> dict:
    bounds = cfg["global_model"]["electron_temperature_bounds_eV"]
    grid = np.linspace(bounds[0], bounds[1], 120)
    residuals = [particle_balance_residual(t, pressure_Pa, Ti_seed_eV, geometry, gas, xs, ion_sigma_scale) for t in grid]
    brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if residuals[i] * residuals[i + 1] <= 0]
    if not brackets:
        return {"converged": False, "pressure_Pa": pressure_Pa, "absorbed_power_W": absorbed_power_W}
    Te = brentq(
        particle_balance_residual,
        brackets[0][0],
        brackets[0][1],
        args=(pressure_Pa, Ti_seed_eV, geometry, gas, xs, ion_sigma_scale),
        xtol=1.0e-7,
    )
    nn = neutral_density(pressure_Pa, gas.neutral_temperature_K)
    rates = {name: xs[name].maxwell_rate(Te) for name in ["e_momentum", "ionization", "dissociation", "excitation"]}
    nu_loss, loss_meta = loss_frequency(pressure_Pa, Te, Ti_seed_eV, geometry, gas, xs, ion_sigma_scale)
    gm = cfg["global_model"]
    energy_per_pair = (
        gm["ionization_energy_eV"]
        + rates["dissociation"] / rates["ionization"] * gm["dissociation_energy_eV"]
        + rates["excitation"] / rates["ionization"] * gm["effective_excitation_energy_eV"]
        + gm["wall_energy_loss_Te_factor"] * Te
    )
    ne = absorbed_power_W / (geometry.volume_m3 * nn * rates["ionization"] * energy_per_pair * e)
    lambda_de = debye_length(ne, Te)
    ionization_fraction = ne / nn
    particle_residual = nn * rates["ionization"] / nu_loss - 1.0
    power_reconstructed = ne * geometry.volume_m3 * nn * rates["ionization"] * energy_per_pair * e
    accepted_ne = gm["electron_density_bounds_m3"][0] <= ne <= gm["electron_density_bounds_m3"][1]
    accepted_te = gm["electron_temperature_acceptance_eV"][0] <= Te <= gm["electron_temperature_acceptance_eV"][1]
    eta_min, eta_max = gm["absorbed_fraction_bounds"]
    generator_low, generator_high = absorbed_power_W / eta_max, absorbed_power_W / eta_min
    source_low, source_high = gm["available_generator_power_W"]
    generator_compatible = max(generator_low, source_low) <= min(generator_high, source_high)
    return {
        "converged": True,
        "pressure_Pa": pressure_Pa,
        "absorbed_power_W": absorbed_power_W,
        "electron_temperature_eV": Te,
        "electron_density_m3": ne,
        "neutral_density_m3": nn,
        "ionization_fraction": ionization_fraction,
        "lambda_De_m": lambda_de,
        "plasma_parameter": plasma_parameter(ne, lambda_de),
        "energy_per_pair_eV": energy_per_pair,
        "K_ion_m3s": rates["ionization"],
        "K_e_momentum_m3s": rates["e_momentum"],
        "K_dissociation_m3s": rates["dissociation"],
        "K_excitation_m3s": rates["excitation"],
        "nu_en_s-1": nn * rates["e_momentum"],
        "nu_loss_s-1": nu_loss,
        "particle_balance_relative_residual": particle_residual,
        "power_balance_relative_residual": power_reconstructed / absorbed_power_W - 1.0,
        "generator_power_interval_W_low": generator_low,
        "generator_power_interval_W_high": generator_high,
        "generator_compatible": generator_compatible,
        "background_accepted": bool(accepted_ne and accepted_te and generator_compatible),
        **loss_meta,
    }


def plasma_dispersion(zeta: complex) -> complex:
    return 1j * math.sqrt(pi) * wofz(zeta)


def ion_acoustic_kinetic_root(k: float, ne: float, Te: float, Ti: float, ion_mass: float) -> complex:
    lambda_e = debye_length(ne, Te)
    lambda_i = math.sqrt(epsilon_0 * Ti / (ne * e))
    vte = math.sqrt(e * Te / m_e)
    vti = math.sqrt(e * Ti / ion_mass)
    cs = math.sqrt(e * (Te + 3.0 * Ti) / ion_mass)
    initial = k * cs / math.sqrt(1.0 + (k * lambda_e) ** 2)

    def dielectric(omega: complex) -> complex:
        ze = omega / (math.sqrt(2.0) * k * vte)
        zi = omega / (math.sqrt(2.0) * k * vti)
        chi_e = (1.0 + ze * plasma_dispersion(ze)) / (k * lambda_e) ** 2
        chi_i = (1.0 + zi * plasma_dispersion(zi)) / (k * lambda_i) ** 2
        return 1.0 + chi_e + chi_i

    def equations(pair: Iterable[float]) -> Tuple[float, float]:
        value = dielectric(complex(pair[0], pair[1]))
        return value.real, value.imag

    solution = root(equations, [initial, -0.02 * initial], method="hybr", tol=1.0e-10)
    omega = complex(solution.x[0], solution.x[1])
    if not solution.success or omega.real <= 0 or omega.imag > 0 or abs(omega.imag) > omega.real:
        # Conservative analytic fallback for difficult corners of the scan.
        ratio = Ti / Te
        gamma_ratio = -math.sqrt(pi / 8.0) * (
            math.sqrt(m_e / ion_mass) + ratio**1.5 * math.exp(-1.0 / max(2.0 * ratio, 1.0e-12))
        )
        omega = complex(initial, gamma_ratio * initial)
    return omega


def wave_state(
    background: dict,
    Ti_eV: float,
    kappa: float,
    geometry: Geometry,
    gas: Gas,
    xs: Dict[str, CrossSection],
    cfg: dict,
    electron_sigma_scale: float = 1.0,
    ion_sigma_scale: float = 1.0,
) -> dict:
    p = background["pressure_Pa"]
    ne = background["electron_density_m3"]
    Te = background["electron_temperature_eV"]
    nn = background["neutral_density_m3"]
    lambda_de = debye_length(ne, Te)
    k = kappa / lambda_de
    vte = math.sqrt(e * Te / m_e)
    omega_pe = math.sqrt(ne * e**2 / (epsilon_0 * m_e))
    omega_pi = math.sqrt(ne * e**2 / (epsilon_0 * gas.ion_mass_kg))
    omega_l = math.sqrt(omega_pe**2 + 3.0 * k**2 * vte**2)
    vg_l = 3.0 * k * vte**2 / omega_l
    gamma_l_landau = -math.sqrt(pi / 8.0) * omega_pe / kappa**3 * math.exp(-1.0 / (2.0 * kappa**2) - 1.5)
    K_e_momentum = xs["e_momentum"].maxwell_rate(Te)
    nu_en = nn * K_e_momentum * electron_sigma_scale
    nu_ei = electron_ion_collision_frequency(ne, Te)
    gamma_l_coll = -0.5 * (nu_en + nu_ei)
    gamma_l = gamma_l_landau + gamma_l_coll
    L_l = vg_l / abs(gamma_l)
    Q_l = omega_l / (2.0 * abs(gamma_l))

    omega_s_kinetic = ion_acoustic_kinetic_root(k, ne, Te, Ti_eV, gas.ion_mass_kg)
    cs = math.sqrt(e * (Te + 3.0 * Ti_eV) / gas.ion_mass_kg)
    vg_s = cs / (1.0 + kappa**2) ** 1.5
    nu_in, ion_relative_energy, K_in = ion_neutral_rate(nn, Te, Ti_eV, gas, xs["ion_momentum"], ion_sigma_scale)
    electron_drag_weight = kappa**2 / (1.0 + kappa**2)
    gamma_s_coll = -0.5 * (nu_in + electron_drag_weight * nu_en)
    gamma_s = omega_s_kinetic.imag + gamma_s_coll
    omega_s = omega_s_kinetic.real
    L_s = vg_s / abs(gamma_s)
    Q_s = omega_s / (2.0 * abs(gamma_s))

    wavelength = 2.0 * pi / k
    caviton_scale = cfg["wave_acceptance"]["caviton_wavelengths"] * wavelength
    acc = cfg["wave_acceptance"]
    checks = {
        "background": bool(background["background_accepted"]),
        "plasma_parameter": background["plasma_parameter"] >= acc["min_plasma_parameter"],
        "ionization_fraction": background["ionization_fraction"] <= acc["max_ionization_fraction"],
        "langmuir_Q": Q_l >= acc["min_langmuir_Q"],
        "langmuir_length": L_l >= acc["min_langmuir_amplitude_lengths_per_interaction"] * geometry.interaction_length_m,
        "ion_acoustic_Q": Q_s >= acc["min_ion_acoustic_Q"],
        "ion_acoustic_length": L_s >= acc["min_ion_acoustic_amplitude_lengths_per_caviton"] * caviton_scale,
    }
    return {
        **background,
        "ion_temperature_eV": Ti_eV,
        "k_lambda_De": kappa,
        "k_m-1": k,
        "wavelength_m": wavelength,
        "caviton_scale_m": caviton_scale,
        "omega_pe_rad_s": omega_pe,
        "omega_pi_rad_s": omega_pi,
        "langmuir_frequency_Hz": omega_l / (2.0 * pi),
        "langmuir_group_velocity_ms": vg_l,
        "gamma_langmuir_landau_s-1": gamma_l_landau,
        "gamma_langmuir_collisional_s-1": gamma_l_coll,
        "gamma_langmuir_total_s-1": gamma_l,
        "L_damp_langmuir_amplitude_m": L_l,
        "Q_langmuir": Q_l,
        "ion_acoustic_frequency_Hz": omega_s / (2.0 * pi),
        "ion_acoustic_group_velocity_ms": vg_s,
        "gamma_ion_acoustic_kinetic_s-1": omega_s_kinetic.imag,
        "gamma_ion_acoustic_collisional_s-1": gamma_s_coll,
        "gamma_ion_acoustic_total_s-1": gamma_s,
        "L_damp_ion_acoustic_amplitude_m": L_s,
        "Q_ion_acoustic": Q_s,
        "nu_en_scaled_s-1": nu_en,
        "nu_ei_s-1": nu_ei,
        "nu_in_scaled_s-1": nu_in,
        "ion_relative_energy_wave_eV": ion_relative_energy,
        "K_in_wave_m3s": K_in,
        **{f"pass_{name}": passed for name, passed in checks.items()},
        "wave_accepted": bool(all(checks.values())),
    }


def run_scan(root_dir: Path, config_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    xs = load_cross_sections(root_dir, cfg)
    gcfg = cfg["geometry"]
    geometry = Geometry(**gcfg)
    gas_cfg = cfg["gas"]
    gas = Gas(
        neutral_temperature_K=gas_cfg["neutral_temperature_K"],
        ion_mass_kg=gas_cfg["ion_mass_u"] * AMU,
        neutral_mass_kg=gas_cfg["neutral_mass_u"] * AMU,
    )
    ti_seed = min(cfg["scan"]["ion_temperature_eV"], key=lambda value: abs(value - 0.1))
    backgrounds = [
        solve_background(p, power, ti_seed, geometry, gas, xs, cfg)
        for p in cfg["scan"]["pressure_Pa"]
        for power in cfg["scan"]["absorbed_power_W"]
    ]
    background_df = pd.DataFrame(backgrounds)
    converged = background_df[background_df.converged].copy()
    wave_rows = []
    for state in converged.to_dict("records"):
        for Ti in cfg["scan"]["ion_temperature_eV"]:
            for kappa in cfg["scan"]["k_lambda_De"]:
                wave_rows.append(wave_state(state, Ti, kappa, geometry, gas, xs, cfg))
    wave_df = pd.DataFrame(wave_rows)
    return background_df, wave_df


def sensitivity_scan(root_dir: Path, config_path: Path, scan_name: str = "scan") -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    xs = load_cross_sections(root_dir, cfg)
    base_geometry = Geometry(**cfg["geometry"])
    gas_cfg = cfg["gas"]
    scan = cfg[scan_name]
    ti_seed = min(scan["ion_temperature_eV"], key=lambda value: abs(value - 0.1))
    rows = []
    for scenario in cfg["uncertainty"]["scenarios"]:
        geometry = replace(base_geometry, wall_transmission_factor=base_geometry.wall_transmission_factor * scenario["wall_scale"])
        gas = Gas(
            gas_cfg["neutral_temperature_K"],
            scenario["ion_mass_u"] * AMU,
            gas_cfg["neutral_mass_u"] * AMU,
        )
        for pressure in scan["pressure_Pa"]:
            for power in scan["absorbed_power_W"]:
                state = solve_background(
                    pressure, power, ti_seed, geometry, gas, xs, cfg, scenario["ion_sigma_scale"]
                )
                if not state["converged"]:
                    continue
                for Ti in scan["ion_temperature_eV"]:
                    for kappa in scan["k_lambda_De"]:
                        row = wave_state(
                            state, Ti, kappa, geometry, gas, xs, cfg,
                            scenario["electron_sigma_scale"], scenario["ion_sigma_scale"]
                        )
                        row["scenario"] = scenario["name"]
                        row["scenario_ion_mass_u"] = scenario["ion_mass_u"]
                        row["scenario_wall_scale"] = scenario["wall_scale"]
                        row["scenario_electron_sigma_scale"] = scenario["electron_sigma_scale"]
                        row["scenario_ion_sigma_scale"] = scenario["ion_sigma_scale"]
                        rows.append(row)
    sensitivity = pd.DataFrame(rows)
    keys = ["pressure_Pa", "absorbed_power_W", "ion_temperature_eV", "k_lambda_De"]
    required_scenarios = len(cfg["uncertainty"]["scenarios"])
    grouped = sensitivity.groupby(keys, as_index=False).agg(
        scenarios_present=("scenario", "nunique"),
        scenarios_accepted=("wave_accepted", "sum"),
        electron_density_min_m3=("electron_density_m3", "min"),
        electron_density_max_m3=("electron_density_m3", "max"),
        electron_temperature_min_eV=("electron_temperature_eV", "min"),
        electron_temperature_max_eV=("electron_temperature_eV", "max"),
        L_langmuir_min_m=("L_damp_langmuir_amplitude_m", "min"),
        L_langmuir_max_m=("L_damp_langmuir_amplitude_m", "max"),
        L_ion_min_m=("L_damp_ion_acoustic_amplitude_m", "min"),
        L_ion_max_m=("L_damp_ion_acoustic_amplitude_m", "max"),
    )
    grouped["robustly_accepted"] = (
        (grouped.scenarios_present == required_scenarios)
        & (grouped.scenarios_accepted == required_scenarios)
    )
    return sensitivity, grouped


def make_plots(background: pd.DataFrame, wave: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    accepted_bg = background[background.get("background_accepted", False) == True]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    scatter = ax.scatter(background.pressure_Pa, background.electron_density_m3 / 1e16,
                         c=background.electron_temperature_eV, s=36 + 2.0 * background.absorbed_power_W,
                         cmap="viridis", alpha=0.75, edgecolor="none")
    if len(accepted_bg):
        ax.scatter(accepted_bg.pressure_Pa, accepted_bg.electron_density_m3 / 1e16,
                   facecolors="none", edgecolors="#d62728", linewidths=1.2, s=90, label="допустимый фон")
    ax.set(xlabel="Давление D₂, Па", ylabel=r"$n_e$, $10^{16}$ м$^{-3}$",
           title="Самосогласованные состояния глобальной модели")
    ax.set_yscale("log")
    fig.colorbar(scatter, ax=ax, label=r"$T_e$, эВ")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "01_global_balance.png", dpi=180)
    plt.close(fig)

    if wave.empty:
        return
    target = []
    for pressure, group in wave.groupby("pressure_Pa"):
        distance = abs(np.log10(group.electron_density_m3 / 1e16)) + abs(group.ion_temperature_eV - 0.1) * 10
        power = group.loc[distance.idxmin(), "absorbed_power_W"]
        selected = group[(group.absorbed_power_W == power) & (group.ion_temperature_eV == 0.1)]
        target.append(selected)
    slice_df = pd.concat(target, ignore_index=True)
    for column, filename, title in [
        ("L_damp_langmuir_amplitude_m", "02_Ldamp_langmuir.png", "Длина затухания ленгмюровской волны"),
        ("L_damp_ion_acoustic_amplitude_m", "03_Ldamp_ion_acoustic.png", "Длина затухания ионно-звуковой волны"),
    ]:
        pivot = slice_df.pivot_table(index="pressure_Pa", columns="k_lambda_De", values=column, aggfunc="median")
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        image = ax.pcolormesh(pivot.columns.to_numpy(), pivot.index.to_numpy(), np.log10(pivot.values),
                              shading="nearest", cmap="magma")
        ax.set(xlabel=r"$k\lambda_{De}$", ylabel="Давление D₂, Па", title=title + " (log₁₀ м)")
        fig.colorbar(image, ax=ax, label=r"$\log_{10}(L_A/\mathrm{м})$")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    survival = wave.groupby(["pressure_Pa", "k_lambda_De"], as_index=False).wave_accepted.mean()
    pivot = survival.pivot(index="pressure_Pa", columns="k_lambda_De", values="wave_accepted")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    image = ax.pcolormesh(pivot.columns.to_numpy(), pivot.index.to_numpy(), pivot.values,
                          shading="nearest", cmap="YlGn", vmin=0, vmax=1)
    ax.set(xlabel=r"$k\lambda_{De}$", ylabel="Давление D₂, Па",
           title="Доля режимов, прошедших все критерии уровня 1")
    fig.colorbar(image, ax=ax, label="Доля прошедших комбинаций")
    fig.tight_layout()
    fig.savefig(output_dir / "04_acceptance_fraction.png", dpi=180)
    plt.close(fig)


def make_robust_plots(uncertainty: pd.DataFrame, output_dir: Path) -> None:
    if uncertainty.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    definitions = [
        ("k_lambda_De", r"$k\lambda_{De}$", "05_robust_pressure_k.png",
         "Устойчивая область: усреднение по мощности и $T_i$"),
        ("absorbed_power_W", r"$P_{abs}$, Вт", "06_robust_pressure_power.png",
         r"Устойчивая область: усреднение по $k\lambda_{De}$ и $T_i$"),
    ]
    for x_column, x_label, filename, title in definitions:
        pivot = uncertainty.pivot_table(index="pressure_Pa", columns=x_column,
                                        values="robustly_accepted", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        image = ax.pcolormesh(pivot.columns.to_numpy(), pivot.index.to_numpy(), pivot.values,
                              shading="nearest", cmap="YlGn", vmin=0, vmax=1)
        ax.set(xlabel=x_label, ylabel="Давление D₂, Па", title=title)
        fig.colorbar(image, ax=ax, label="Доля комбинаций, прошедших 5 сценариев")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def summarize(background: pd.DataFrame, wave: pd.DataFrame, uncertainty: pd.DataFrame) -> dict:
    bg_ok = background[background.get("background_accepted", False) == True]
    wave_ok = wave[wave.wave_accepted == True]
    robust = uncertainty[uncertainty.robustly_accepted == True] if not uncertainty.empty else uncertainty

    def interval(frame: pd.DataFrame, column: str):
        return None if frame.empty else [float(frame[column].min()), float(frame[column].max())]

    return {
        "counts": {
            "background_total": int(len(background)),
            "background_accepted": int(len(bg_ok)),
            "wave_total": int(len(wave)),
            "wave_accepted": int(len(wave_ok)),
            "refined_robust_combinations": int(len(robust)),
        },
        "accepted_window_nominal": {
            "pressure_Pa": interval(wave_ok, "pressure_Pa"),
            "absorbed_power_W": interval(wave_ok, "absorbed_power_W"),
            "electron_density_m3": interval(wave_ok, "electron_density_m3"),
            "electron_temperature_eV": interval(wave_ok, "electron_temperature_eV"),
            "ion_temperature_eV": interval(wave_ok, "ion_temperature_eV"),
            "k_lambda_De": interval(wave_ok, "k_lambda_De"),
            "langmuir_frequency_Hz": interval(wave_ok, "langmuir_frequency_Hz"),
            "ion_acoustic_frequency_Hz": interval(wave_ok, "ion_acoustic_frequency_Hz"),
            "L_damp_langmuir_amplitude_m": interval(wave_ok, "L_damp_langmuir_amplitude_m"),
            "L_damp_ion_acoustic_amplitude_m": interval(wave_ok, "L_damp_ion_acoustic_amplitude_m"),
        },
        "robust_window_sensitivity_envelope": {
            "pressure_Pa": interval(robust, "pressure_Pa"),
            "absorbed_power_W": interval(robust, "absorbed_power_W"),
            "electron_density_all_scenarios_m3": None if robust.empty else [float(robust.electron_density_min_m3.min()), float(robust.electron_density_max_m3.max())],
            "electron_temperature_all_scenarios_eV": None if robust.empty else [float(robust.electron_temperature_min_eV.min()), float(robust.electron_temperature_max_eV.max())],
            "ion_temperature_eV": interval(robust, "ion_temperature_eV"),
            "k_lambda_De": interval(robust, "k_lambda_De"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    background, wave = run_scan(root_dir, args.config)
    accepted = wave[wave.wave_accepted == True]
    sensitivity, uncertainty = sensitivity_scan(root_dir, args.config, "scan")
    refined_sensitivity, refined_uncertainty = sensitivity_scan(root_dir, args.config, "refinement")
    background.to_csv(args.output / "global_balance.csv", index=False)
    wave.to_csv(args.output / "wave_map.csv", index=False)
    accepted.to_csv(args.output / "accepted_regimes_nominal.csv", index=False)
    sensitivity.to_csv(args.output / "sensitivity_scenarios.csv", index=False)
    uncertainty.to_csv(args.output / "uncertainty_envelope.csv", index=False)
    uncertainty[uncertainty.robustly_accepted == True].to_csv(args.output / "accepted_regimes_robust.csv", index=False)
    refined_sensitivity.to_csv(args.output / "refined_sensitivity_scenarios.csv", index=False)
    refined_uncertainty.to_csv(args.output / "refined_uncertainty_envelope.csv", index=False)
    refined_uncertainty[refined_uncertainty.robustly_accepted == True].to_csv(args.output / "accepted_regimes_robust_refined.csv", index=False)
    summary = summarize(background, wave, refined_uncertainty)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plots(background, wave, root_dir / "figures")
    make_robust_plots(refined_uncertainty, root_dir / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
