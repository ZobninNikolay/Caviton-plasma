#!/usr/bin/env python3
"""Level 4.8: power-closed resonant cascade and adaptive rare-tail map.

This is the locally executable branch selected after the Level-4.7 resource
audit.  It deliberately separates three statements:

* a lower-hybrid stage can be placed in a technically accessible f-B-ne window;
* prescribed, phase-matched travelling fields can transport a weighted D+ tail;
* only a later self-consistent EM-PIC calculation can prove that the fields are
  generated and survive electron/ion loading.

The wave-energy equation is closed by neutral damping and ion loading.  Rare
tails are resolved with sequential conditional resampling (particle splitting),
not by pretending that a small brute-force PIC ensemble resolves 1e-7 events.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import atomic_mass, e, epsilon_0, m_e, pi
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, differential_evolution


M_D = 2.014 * atomic_mass


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def omega_pe(ne_m3: float) -> float:
    return math.sqrt(ne_m3 * e**2 / (epsilon_0 * m_e))


def debye_length(ne_m3: float, Te_eV: float) -> float:
    return math.sqrt(epsilon_0 * Te_eV / (ne_m3 * e))


def lower_hybrid_omega(B_T: float, ne_m3: float) -> float:
    wpe = omega_pe(ne_m3)
    wce = e * B_T / m_e
    wci = e * B_T / M_D
    return math.sqrt(wce * wci / (1.0 + wce**2 / wpe**2))


def resonant_B(omega_rad_s: float, ne_m3: float) -> float:
    residual = lambda field: lower_hybrid_omega(field, ne_m3) - omega_rad_s
    if residual(1.0e-6) * residual(1.0) > 0.0:
        return math.nan
    return brentq(residual, 1.0e-6, 1.0)


def mode_damping_rate(cfg: dict, collision_scale: float = 1.0) -> float:
    col = cfg["collisions"]
    return 0.5 * collision_scale * (
        float(col["electron_neutral_s-1"]) + float(col["ion_neutral_effective_s-1"])
    )


def required_trapping_potential(previous_eV: float, target_eV: float) -> float:
    """Minimum travelling-potential amplitude from the 1-D separatrix bound."""
    return (math.sqrt(target_eV) - math.sqrt(previous_eV)) ** 2


def calibrated_capture_probability(trap_ratio: float) -> float:
    """Monotone calibration to Level-4.7 right-exit orbit/MCC controls.

    This is a design surrogate, not a universal lower-hybrid efficiency law.
    The adaptive phase map below independently checks the optimized result.
    """
    ratio = np.array([0.0, 1.0, 1.18, 1.43, 2.36, 4.32, 8.0])
    probability = np.array([0.0, 0.0, 0.191, 0.330, 0.493, 0.796, 0.900])
    return float(np.interp(trap_ratio, ratio, probability))


@dataclass
class CascadeEvaluation:
    stages: pd.DataFrame
    radius_m: float
    total_power_W: float
    final_rate_s1: float
    analytic_tail_fraction: float
    feasible: bool
    penalty: float


def stage_density(frequency_Hz: float, cfg: dict) -> float:
    plasma = cfg["plasma"]
    ratio = float(plasma["preferred_omega_pe_over_omega_ce"])
    omega = 2.0 * pi * frequency_Hz
    density_for_ratio = epsilon_0 * M_D * (ratio**2 + 1.0) * omega**2 / e**2
    return max(float(plasma["minimum_density_m3"]), density_for_ratio)


def evaluate_design(vector: np.ndarray, budget_W: float, cfg: dict,
                    mode_multiplier: float | None = None,
                    capture_multiplier: float = 1.0,
                    collision_multiplier: float = 1.0) -> CascadeEvaluation:
    cas = cfg["cascade"]
    plasma = cfg["plasma"]
    targets = np.asarray(cas["target_energies_eV"], dtype=float)
    previous = np.r_[float(plasma["injection_energy_eV"]), targets[:-1]]
    count = len(targets)
    trap_ratios = np.asarray(vector[:count], dtype=float)
    lengths = np.asarray(vector[count:], dtype=float)
    multiplier = float(cas["mode_energy_multiplier"] if mode_multiplier is None else mode_multiplier)
    gamma = mode_damping_rate(cfg, collision_multiplier)
    nu_in = collision_multiplier * float(cfg["collisions"]["ion_neutral_effective_s-1"])
    density0 = float(plasma["minimum_density_m3"])
    source_rate_per_r2 = density0 * pi * math.sqrt(2.0 * e * previous[0] / M_D)
    incoming_fraction = 1.0
    power_per_r2 = 0.0
    rows: list[dict] = []
    penalty = max(0.0, lengths.sum() - float(cas["total_length_ceiling_m"])) ** 2 * 1.0e6

    for index, (initial, target, ratio, length) in enumerate(
        zip(previous, targets, trap_ratios, lengths), start=1
    ):
        velocity = math.sqrt(2.0 * e * target / M_D)
        frequency = velocity / length
        omega = 2.0 * pi * frequency
        density = stage_density(frequency, cfg)
        field_B = resonant_B(omega, density)
        frequency_ceiling = float(cas["frequency_ceiling_Hz"])
        density_ceiling = float(plasma["maximum_density_m3"])
        B_ceiling = float(cas["magnetic_field_ceiling_T"])
        penalty += max(0.0, frequency / frequency_ceiling - 1.0) ** 2 * 1.0e6
        penalty += max(0.0, density / density_ceiling - 1.0) ** 2 * 1.0e6
        penalty += max(0.0, field_B / B_ceiling - 1.0) ** 2 * 1.0e6

        wavenumber = 2.0 * pi / length
        potential_required = required_trapping_potential(initial, target)
        potential = ratio * potential_required
        field = wavenumber * potential
        wavebreaking = 2.0 * wavenumber * target
        cap = float(cas["wavebreaking_fraction_cap"]) * wavebreaking
        penalty += max(0.0, field / cap - 1.0) ** 2 * 1.0e6
        capture = capture_multiplier * calibrated_capture_probability(ratio)
        collision_survival = math.exp(-nu_in * length / velocity)
        capture = min(1.0, capture * collision_survival)
        incoming_rate_per_r2 = source_rate_per_r2 * incoming_fraction
        outgoing_fraction = incoming_fraction * capture

        # W = G eps0 E^2 V / 4, P_damp = 2 gamma W.
        damping_per_r2 = 0.5 * gamma * epsilon_0 * field**2 * pi * length * multiplier
        ion_loading_per_r2 = (
            incoming_rate_per_r2 * capture * (target - initial) * e
        )
        power_per_r2 += damping_per_r2 + ion_loading_per_r2
        Q = omega / (2.0 * gamma)
        rows.append({
            "stage": index,
            "initial_energy_eV": initial,
            "target_energy_eV": target,
            "section_length_m": length,
            "frequency_Hz": frequency,
            "electron_density_m3": density,
            "resonant_B_T": field_B,
            "omega_pe_over_omega_ce": omega_pe(density) / (e * field_B / m_e),
            "Q_collisional": Q,
            "trap_ratio": ratio,
            "required_potential_V": potential_required,
            "operating_potential_V": potential,
            "electric_field_Vm": field,
            "wavebreaking_field_Vm": wavebreaking,
            "capture_probability_surrogate": capture,
            "incoming_fraction": incoming_fraction,
            "outgoing_fraction": outgoing_fraction,
            "damping_power_per_r2_Wm-2": damping_per_r2,
            "ion_loading_per_r2_Wm-2": ion_loading_per_r2,
        })
        incoming_fraction = outgoing_fraction

    radius_unbounded = math.sqrt(budget_W / max(power_per_r2, 1.0e-30))
    radius = min(radius_unbounded, float(cas["maximum_radius_m"]))
    minimum_radius = float(cas["minimum_radius_m"])
    if radius_unbounded < minimum_radius:
        penalty += (minimum_radius / max(radius_unbounded, 1.0e-30) - 1.0) ** 2 * 1.0e6
        radius = minimum_radius
    total_power = power_per_r2 * radius**2
    final_rate = source_rate_per_r2 * radius**2 * incoming_fraction
    stages = pd.DataFrame(rows)
    stages["radius_m"] = radius
    stages["damping_power_W"] = stages["damping_power_per_r2_Wm-2"] * radius**2
    stages["ion_loading_W"] = stages["ion_loading_per_r2_Wm-2"] * radius**2
    stages["absorbed_drive_power_W"] = stages.damping_power_W + stages.ion_loading_W
    feasible = bool(
        penalty < 1.0e-5
        and total_power <= budget_W * (1.0 + 2.0e-4)
        and radius >= minimum_radius
    )
    return CascadeEvaluation(
        stages=stages,
        radius_m=radius,
        total_power_W=total_power,
        final_rate_s1=final_rate,
        analytic_tail_fraction=incoming_fraction,
        feasible=feasible,
        penalty=penalty,
    )


def optimization_bounds(cfg: dict) -> list[tuple[float, float]]:
    count = len(cfg["cascade"]["target_energies_eV"])
    qbounds = tuple(float(v) for v in cfg["cascade"]["trap_ratio_bounds"])
    return [qbounds] * count + [tuple(float(v) for v in pair)
                                for pair in cfg["cascade"]["section_length_bounds_m"]]


def optimize_budget(budget_W: float, cfg: dict) -> tuple[np.ndarray, CascadeEvaluation]:
    opt = cfg["optimization"]

    def objective(vector: np.ndarray) -> float:
        result = evaluate_design(vector, budget_W, cfg)
        return -math.log10(max(result.final_rate_s1, 1.0)) + result.penalty

    result = differential_evolution(
        objective,
        optimization_bounds(cfg),
        seed=int(opt["random_seed"]) + int(round(budget_W)),
        maxiter=int(opt["maximum_iterations"]),
        popsize=int(opt["population_size"]),
        tol=float(opt["tolerance"]),
        polish=True,
        workers=1,
        updating="immediate",
    )
    return result.x, evaluate_design(result.x, budget_W, cfg)


def integrate_mode_ode(stage: pd.Series, cfg: dict) -> pd.DataFrame:
    cas = cfg["cascade"]
    gamma = mode_damping_rate(cfg)
    multiplier = float(cas["mode_energy_multiplier"])
    radius = float(stage.radius_m)
    volume = pi * radius**2 * float(stage.section_length_m)
    drive = float(stage.absorbed_drive_power_W)
    beam_start = float(cas["beam_start_s"])
    duration = float(cas["pulse_duration_s"])
    target_delta = float(stage.target_energy_eV - stage.initial_energy_eV)
    incoming_rate = (
        float(stage.ion_loading_W)
        / max(float(stage.capture_probability_surrogate) * target_delta * e, 1.0e-30)
    )
    required_phi = float(stage.required_potential_V)
    k = 2.0 * pi / float(stage.section_length_m)
    W_to_E = 4.0 / (multiplier * epsilon_0 * volume)
    nu = float(cfg["collisions"]["ion_neutral_effective_s-1"])
    velocity = math.sqrt(2.0 * e * float(stage.target_energy_eV) / M_D)
    survival = math.exp(-nu * float(stage.section_length_m) / velocity)

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        energy = max(float(state[0]), 0.0)
        field = math.sqrt(W_to_E * energy)
        trap_ratio = field / max(k * required_phi, 1.0e-30)
        capture = calibrated_capture_probability(trap_ratio) * survival
        ion_power = incoming_rate * capture * target_delta * e if time_s >= beam_start else 0.0
        return np.array([drive - 2.0 * gamma * energy - ion_power])

    time = np.linspace(0.0, duration, 401)
    solution = solve_ivp(rhs, (0.0, duration), [0.0], t_eval=time,
                         rtol=2.0e-8, atol=1.0e-14, max_step=duration / 1000.0)
    energy = np.maximum(solution.y[0], 0.0)
    field = np.sqrt(W_to_E * energy)
    trap_ratio = field / max(k * required_phi, 1.0e-30)
    capture = np.array([
        calibrated_capture_probability(value) * survival if time_s >= beam_start else 0.0
        for value, time_s in zip(trap_ratio, solution.t)
    ])
    ion_power = incoming_rate * capture * target_delta * e
    return pd.DataFrame({
        "stage": int(stage.stage),
        "time_s": solution.t,
        "wave_energy_J": energy,
        "electric_field_Vm": field,
        "trap_ratio": trap_ratio,
        "instant_capture_probability": capture,
        "neutral_damping_power_W": 2.0 * gamma * energy,
        "ion_loading_power_W": ion_power,
        "drive_power_W": drive,
    })


def phase_map_stage(initial_energy_eV: np.ndarray, stage: pd.Series, cfg: dict,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-pass physical-mass orbit map with open ends and post-flight MCC."""
    apm = cfg["adaptive_phase_map"]
    count = len(initial_energy_eV)
    target = float(stage.target_energy_eV)
    length = float(stage.section_length_m)
    frequency = float(stage.frequency_Hz)
    omega = 2.0 * pi * frequency
    k = 2.0 * pi / length
    field = float(stage.electric_field_Vm)
    velocity = np.sqrt(2.0 * e * np.maximum(initial_energy_eV, 1.0e-6) / M_D)
    position = np.zeros(count)
    phase = rng.uniform(0.0, 2.0 * pi, count)
    active = np.ones(count, dtype=bool)
    output_energy = np.zeros(count)
    flight_time = np.zeros(count)
    dt = 1.0 / (float(apm["steps_per_wave_period"]) * frequency)
    speed_reference = max(float(np.percentile(velocity, 5.0)), 1.0)
    tmax = float(apm["maximum_slow_transits"]) * length / speed_reference
    time_s = 0.0

    while time_s < tmax and np.any(active):
        indices = np.flatnonzero(active)
        z = position[indices]
        v = velocity[indices]
        envelope = np.sin(pi * np.clip(z / length, 0.0, 1.0)) ** 2
        electric = field * envelope * np.sin(k * z - omega * time_s + phase[indices])
        v += (e / M_D) * electric * dt
        z += v * dt
        velocity[indices] = v
        position[indices] = z
        done = indices[(z >= length) | (z < 0.0)]
        if len(done):
            output_energy[done] = 0.5 * M_D * velocity[done] ** 2 / e
            flight_time[done] = time_s + dt
            active[done] = False
        time_s += dt

    if np.any(active):
        output_energy[active] = 0.5 * M_D * velocity[active] ** 2 / e
        flight_time[active] = tmax
    nu = float(cfg["collisions"]["ion_neutral_effective_s-1"])
    collision_survival = rng.random(count) < np.exp(-nu * flight_time)
    reached_right = position >= length
    success = reached_right & collision_survival & (output_energy >= target)
    return output_energy, success, flight_time


def adaptive_phase_cascade(stages: pd.DataFrame, cfg: dict,
                           label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sequential conditional resampling resolves the rare surviving branch."""
    apm = cfg["adaptive_phase_map"]
    rng = np.random.default_rng(int(apm["random_seed"]) + int(round(stages.radius_m.iloc[0] * 1.0e6)))
    count = int(apm["particles_per_stage"])
    injection = float(cfg["plasma"]["injection_energy_eV"])
    Ti = float(cfg["plasma"]["ion_temperature_eV"])
    sigma_energy = math.sqrt(max(2.0 * injection * Ti, 1.0e-12))
    energy = np.maximum(rng.normal(injection, sigma_energy, count), 1.0e-3)
    active_weight = 1.0
    histories: list[dict] = []

    for _, stage in stages.iterrows():
        output, success, flight = phase_map_stage(energy, stage, cfg, rng)
        conditional = float(np.mean(success))
        active_weight *= conditional
        histories.append({
            "case": label,
            "stage": int(stage.stage),
            "target_energy_eV": float(stage.target_energy_eV),
            "conditional_success_probability": conditional,
            "cumulative_tail_fraction": active_weight,
            "successful_particles_before_resampling": int(np.sum(success)),
            "conditional_binomial_sigma": math.sqrt(max(conditional * (1.0 - conditional) / count, 0.0)),
            "mean_flight_time_s": float(np.mean(flight)),
            "successful_median_energy_eV": float(np.median(output[success])) if np.any(success) else math.nan,
            "successful_p99_energy_eV": float(np.quantile(output[success], 0.99)) if np.any(success) else math.nan,
        })
        if not np.any(success):
            break
        successful_energy = output[success]
        # Particle splitting / conditional resampling: keep N representatives
        # on the rare branch while reducing their statistical weight.
        selected = rng.integers(0, len(successful_energy), count)
        energy = successful_energy[selected]

    bins = np.geomspace(float(apm["minimum_energy_eV"]), float(apm["maximum_energy_eV"]),
                       int(apm["energy_bins"]) + 1)
    histogram = np.zeros(len(bins) - 1)
    # The delivered spectrum contains the final transmitted branch.  Ions lost
    # to charge exchange or to the wrong open boundary are returned to the cold
    # background rather than being counted at their pre-loss kinetic energy.
    histogram += np.histogram(np.full(count, injection), bins=bins)[0] * (
        (1.0 - active_weight) / count
    )
    if active_weight > 0.0:
        histogram += np.histogram(energy, bins=bins)[0] * (active_weight / len(energy))
    total_hist = histogram.sum()
    if total_hist > 0.0:
        histogram /= total_hist
    spectrum = pd.DataFrame({
        "case": label,
        "energy_eV": np.sqrt(bins[:-1] * bins[1:]),
        "probability_per_log_bin": histogram,
    })
    history = pd.DataFrame(histories)
    tail = {
        f"fraction_above_{int(row['target_energy_eV'])}eV": float(row["cumulative_tail_fraction"])
        for row in histories
    }
    tail["adaptive_final_branch_weight"] = active_weight
    tail["particles_per_conditional_stage"] = count
    tail["equivalent_bruteforce_particles_for_30_final_counts"] = 30.0 / max(active_weight, 1.0e-300)
    return history, spectrum, tail


def sensitivity_table(vector: np.ndarray, nominal_budget: float, cfg: dict) -> pd.DataFrame:
    rows = []
    for multiplier in cfg["sensitivity"]["mode_energy_multipliers"]:
        for capture in cfg["sensitivity"]["capture_efficiency_multipliers"]:
            for collision in cfg["sensitivity"]["collision_rate_multipliers"]:
                result = evaluate_design(
                    vector, nominal_budget, cfg,
                    mode_multiplier=float(multiplier),
                    capture_multiplier=float(capture),
                    collision_multiplier=float(collision),
                )
                rows.append({
                    "mode_energy_multiplier": float(multiplier),
                    "capture_efficiency_multiplier": float(capture),
                    "collision_rate_multiplier": float(collision),
                    "radius_m": result.radius_m,
                    "total_power_W": result.total_power_W,
                    "analytic_100keV_tail_fraction": result.analytic_tail_fraction,
                    "final_100keV_rate_s-1": result.final_rate_s1,
                    "feasible_at_nominal_budget": result.feasible,
                })
    return pd.DataFrame(rows)


def semi_implicit_pic_audit(stages: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    audit = cfg["semi_implicit_pic_audit"]
    Te = float(cfg["plasma"]["electron_temperature_eV"])
    rows = []
    for _, stage in stages.iterrows():
        length = float(stage.section_length_m)
        density = float(stage.electron_density_m3)
        frequency = float(stage.frequency_Hz)
        B = float(stage.resonant_B_T)
        ld = debye_length(density, Te)
        cells = max(
            int(audit["minimum_cells_per_section"]),
            int(math.ceil(length / (float(audit["cells_per_Debye_length"]) * ld))),
        )
        particles = cells * int(audit["particles_per_cell_per_species"]) * int(audit["kinetic_species"])
        wce = e * B / m_e
        dt = min(
            1.0 / (float(audit["wave_steps_per_period"]) * frequency),
            float(audit["gyro_accuracy_factor"]) / wce,
        )
        duration = float(audit["periods_to_observe"]) / frequency
        steps = int(math.ceil(duration / dt))
        rows.append({
            "stage": int(stage.stage),
            "model": "1D3V semi-implicit energy-conserving EM-PIC validation",
            "grid_cells": cells,
            "particles_all_species": particles,
            "time_step_s": dt,
            "observation_time_s": duration,
            "time_steps": steps,
            "particle_pushes": particles * steps,
            "particle_memory_GB": particles * float(audit["bytes_per_particle"]) / 1.0e9,
            "local_laptop_run_recommended": False,
            "small_cluster_or_HPC_feasible": True,
        })
    return pd.DataFrame(rows)


def make_figure(geometry: pd.DataFrame, histories: pd.DataFrame,
                spectra: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    budget_summary = geometry.groupby("budget_W").first().reset_index()
    axes[0].plot(budget_summary.budget_W, 1.0e3 * budget_summary.radius_m, "o-")
    axes[0].set(xscale="log", xlabel="поглощённая мощность, Вт", ylabel="радиус трубки, мм",
                title="Мощностно допустимая геометрия")
    nominal = geometry[geometry.is_nominal_budget]
    axes[1].bar(nominal.stage.astype(str), nominal.absorbed_drive_power_W,
                label="всего")
    axes[1].bar(nominal.stage.astype(str), nominal.ion_loading_W,
                label="в ионы")
    axes[1].set(xlabel="ступень", ylabel="мощность, Вт", title="Распределение 59,3 Вт")
    axes[1].legend()
    for case, group in spectra.groupby("case"):
        axes[2].plot(group.energy_eV, group.probability_per_log_bin, label=case)
    for threshold in [1.0e3, 1.0e4, 1.0e5]:
        axes[2].axvline(threshold, color="k", lw=0.6, ls="--")
    axes[2].set(xscale="log", yscale="log", xlabel="энергия D⁺, эВ",
                ylabel="вероятность на log-интервал", title="Адаптивный хвост")
    axes[2].set_ylim(1.0e-8, 1.0)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_level48.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.config.resolve().parent
    cfg = load_yaml(args.config)
    with (root / cfg["input"]["level47_summary_json"]).open("r", encoding="utf-8") as stream:
        previous_summary = json.load(stream)
    if previous_summary["observation"]["100keV_tail_seen"]:
        raise ValueError("Level 4.8 assumes that Level 4.7 did not already resolve 100 keV")
    args.output.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    nominal_budget = float(cfg["cascade"]["nominal_power_budget_W"])
    designs: dict[float, tuple[np.ndarray, CascadeEvaluation]] = {}
    geometry_parts = []
    cascade_histories = []
    cascade_spectra = []
    cascade_rows = []
    ode_parts = []
    for budget in cfg["cascade"]["absorbed_power_budgets_W"]:
        budget = float(budget)
        vector, result = optimize_budget(budget, cfg)
        designs[budget] = (vector, result)
        stages = result.stages.copy()
        stages["budget_W"] = budget
        stages["is_nominal_budget"] = math.isclose(budget, nominal_budget, rel_tol=0.0, abs_tol=1.0e-6)
        geometry_parts.append(stages)
        label = f"{budget:g} W"
        history, spectrum, tail = adaptive_phase_cascade(result.stages, cfg, label)
        cascade_histories.append(history)
        cascade_spectra.append(spectrum)
        cascade_rows.append({
            "budget_W": budget,
            "radius_m": result.radius_m,
            "total_length_m": float(result.stages.section_length_m.sum()),
            "total_absorbed_power_W": result.total_power_W,
            "analytic_100keV_tail_fraction": result.analytic_tail_fraction,
            "adaptive_100keV_tail_fraction": tail["adaptive_final_branch_weight"],
            "adaptive_100keV_rate_s-1": (
                result.final_rate_s1 * tail["adaptive_final_branch_weight"]
                / max(result.analytic_tail_fraction, 1.0e-300)
            ),
            "equivalent_bruteforce_particles_for_30_final_counts": tail[
                "equivalent_bruteforce_particles_for_30_final_counts"
            ],
            "feasible": result.feasible,
        } | {key: value for key, value in tail.items() if key.startswith("fraction_above_")})
        if math.isclose(budget, nominal_budget, rel_tol=0.0, abs_tol=1.0e-6):
            for _, stage in result.stages.iterrows():
                ode_parts.append(integrate_mode_ode(stage, cfg))
        print(
            f"{budget:g} W: r={1e3*result.radius_m:.2f} mm, "
            f"analytic={result.analytic_tail_fraction:.3e}, "
            f"adaptive={tail['adaptive_final_branch_weight']:.3e}, feasible={result.feasible}"
        )

    geometry = pd.concat(geometry_parts, ignore_index=True)
    phase_history = pd.concat(cascade_histories, ignore_index=True)
    spectra = pd.concat(cascade_spectra, ignore_index=True)
    cascade_summary = pd.DataFrame(cascade_rows)
    ode_history = pd.concat(ode_parts, ignore_index=True)
    nominal_vector, nominal_result = designs[nominal_budget]
    sensitivity = sensitivity_table(nominal_vector, nominal_budget, cfg)
    resource_audit = semi_implicit_pic_audit(nominal_result.stages, cfg)
    robust_minimum_power = float(sensitivity.total_power_W.max())

    ode_end = ode_history.sort_values("time_s").groupby("stage").tail(1)
    summary = {
        "scope": "technically feasible power-closed cascade screening with adaptive rare-tail map",
        "calculation_status": {
            "local_reduced_Vlasov_orbit_MCC_completed": True,
            "self_consistent_mode_energy_ODE_completed": True,
            "adaptive_particle_splitting_completed": True,
            "full_self_consistent_EM_PIC_completed": False,
        },
        "nominal_budget_W": nominal_budget,
        "nominal_design": {
            "radius_m": nominal_result.radius_m,
            "total_length_m": float(nominal_result.stages.section_length_m.sum()),
            "total_power_W": nominal_result.total_power_W,
            "analytic_100keV_tail_fraction": nominal_result.analytic_tail_fraction,
            "adaptive_100keV_tail_fraction": float(
                cascade_summary.loc[np.isclose(cascade_summary.budget_W, nominal_budget),
                                    "adaptive_100keV_tail_fraction"].iloc[0]
            ),
            "all_mode_ODEs_end_above_threshold": bool(np.all(ode_end.trap_ratio >= 1.0)),
            "maximum_frequency_Hz": float(nominal_result.stages.frequency_Hz.max()),
            "maximum_B_T": float(nominal_result.stages.resonant_B_T.max()),
            "maximum_field_Vm": float(nominal_result.stages.electric_field_Vm.max()),
        },
        "sensitivity": {
            "minimum_analytic_tail_fraction": float(sensitivity.analytic_100keV_tail_fraction.min()),
            "maximum_analytic_tail_fraction": float(sensitivity.analytic_100keV_tail_fraction.max()),
            "all_scenarios_power_feasible": bool(sensitivity.feasible_at_nominal_budget.all()),
            "minimum_absorbed_power_for_all_scanned_scenarios_at_2mm_W": robust_minimum_power,
            "recommended_validation_budget_W": 200.0,
        },
        "interpretation": (
            "The calculation demonstrates a power- and orbit-accessible prescribed-wave cascade. "
            "It does not demonstrate self-consistent lower-hybrid wave generation or phase locking."
        ),
    }

    geometry.to_csv(args.output / "level48_stage_design.csv", index=False)
    cascade_summary.to_csv(args.output / "level48_cascade_summary.csv", index=False)
    phase_history.to_csv(args.output / "level48_adaptive_phase_history.csv", index=False)
    spectra.to_csv(args.output / "level48_tail_distribution.csv", index=False)
    ode_history.to_csv(args.output / "level48_mode_ode.csv", index=False)
    sensitivity.to_csv(args.output / "level48_sensitivity.csv", index=False)
    resource_audit.to_csv(args.output / "level48_semiimplicit_pic_audit.csv", index=False)
    with (args.output / "level48_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    make_figure(geometry, phase_history, spectra, root / "figures" / "23_level48_cascade.png")


if __name__ == "__main__":
    main()
