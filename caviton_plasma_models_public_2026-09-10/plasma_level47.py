#!/usr/bin/env python3
"""Level 4.7: lower-hybrid kinetic observation and explicit-PIC feasibility audit.

Two intentionally separate calculations are provided:

1. a full-scale explicit electromagnetic PIC resource/design audit, with a
   Smilei input deck emitted separately for an HPC run;
2. a locally executable open-boundary, physical-ion-orbit/MCC predictor in a
   prescribed lower-hybrid wave.  This second calculation tests phase trapping,
   neutral losses, species selectivity and power loading.  It does not prove
   that the prescribed wave is generated self-consistently.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import atomic_mass, c, e, epsilon_0, m_e, pi
from scipy.optimize import brentq


M_D = 2.014 * atomic_mass


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def selected_regime(root: Path, cfg: dict) -> pd.Series:
    table = pd.read_csv(root / cfg["input"]["extrema_summary_csv"])
    rows = table[table.search == cfg["input"]["selected_regime"]]
    if len(rows) != 1:
        raise ValueError("The selected regime is missing or ambiguous")
    return rows.iloc[0]


def omega_pe(ne_m3: float) -> float:
    return math.sqrt(ne_m3 * e**2 / (epsilon_0 * m_e))


def debye_length(ne_m3: float, Te_eV: float) -> float:
    return math.sqrt(epsilon_0 * Te_eV / (ne_m3 * e))


def lower_hybrid_omega(B_T: float, ne_m3: float, ion_mass_kg: float = M_D) -> float:
    wpe = omega_pe(ne_m3)
    wce = e * B_T / m_e
    wci = e * B_T / ion_mass_kg
    return math.sqrt(wce * wci / (1.0 + wce**2 / wpe**2))


def resonant_B(omega_rad_s: float, ne_m3: float, ion_mass_kg: float = M_D) -> float:
    residual = lambda field: lower_hybrid_omega(field, ne_m3, ion_mass_kg) - omega_rad_s
    lo, hi = 1.0e-5, 2.0
    if residual(lo) * residual(hi) > 0.0:
        return math.nan
    return brentq(residual, lo, hi)


def active_volume(cfg: dict) -> float:
    ex = cfg["experiment"]
    return pi * float(ex["radius_m"]) ** 2 * float(ex["chamber_length_m"])


def damping_rate(cfg: dict) -> float:
    d = cfg["damping"]
    return 0.5 * float(d["lower_hybrid_scale"]) * (
        float(d["electron_neutral_s-1"]) + float(d["ion_neutral_effective_s-1"])
    )


def driven_field(power_W: float, cfg: dict) -> float:
    gamma = damping_rate(cfg)
    pulse = float(cfg["experiment"]["pulse_duration_s"])
    steady_energy = power_W / (2.0 * gamma) if power_W > 0.0 else 0.0
    energy = steady_energy * (1.0 - math.exp(-2.0 * gamma * pulse))
    return math.sqrt(4.0 * energy / (epsilon_0 * active_volume(cfg)))


def harmonic_for_target(frequency_Hz: float, length_m: float, target_eV: float) -> int:
    velocity = math.sqrt(2.0 * e * target_eV / M_D)
    return max(1, int(round(frequency_Hz * length_m / velocity)))


def explicit_pic_audit(cfg: dict) -> pd.DataFrame:
    p = cfg["plasma"]
    ex = cfg["experiment"]
    audit = cfg["explicit_pic_audit"]
    length = float(ex["chamber_length_m"])
    duration = float(ex["pulse_duration_s"])
    Te = float(p["electron_temperature_cold_eV"])
    rows = []
    for ne in p["density_scan_m3"]:
        ne = float(ne)
        ld = debye_length(ne, Te)
        dx = ld / float(audit["cells_per_Debye_length"])
        cells = int(math.ceil(length / dx))
        dt_cfl = float(audit["maxwell_CFL"]) * dx / c
        steps = int(math.ceil(duration / dt_cfl))
        minimum_particles = int(math.ceil(
            float(audit["required_tail_counts"]) / float(audit["required_tail_fraction"])
        ))
        ppc_tail = int(math.ceil(minimum_particles / cells))
        ppc = max(int(audit["particles_per_cell_floor"]), ppc_tail)
        particles_per_species = cells * ppc
        pushes_five_species = 5.0 * particles_per_species * steps
        B = resonant_B(2.0 * pi * 1.0e6, ne)
        rows.append({
            "electron_density_m3": ne,
            "lambda_De_m": ld,
            "omega_pe_s-1": omega_pe(ne),
            "resonant_B_1MHz_T": B,
            "omega_pe_over_omega_ce": omega_pe(ne) / (e * B / m_e),
            "grid_spacing_m": dx,
            "grid_cells": cells,
            "Maxwell_time_step_s": dt_cfl,
            "time_steps_20us": steps,
            "particles_per_cell_for_tail": ppc,
            "particles_per_species": particles_per_species,
            "five_species_particle_pushes": pushes_five_species,
            "zero_count_95pct_upper_tail_fraction": 3.0 / minimum_particles,
        })
    return pd.DataFrame(rows)


def frequency_power_scan(cfg: dict) -> pd.DataFrame:
    ex = cfg["experiment"]
    p = cfg["plasma"]
    length = float(ex["chamber_length_m"])
    target = float(ex["target_energy_eV"])
    rows = []
    for ne in p["density_scan_m3"]:
        for frequency in ex["drive_frequencies_Hz"]:
            frequency = float(frequency)
            harmonic = harmonic_for_target(frequency, length, target)
            k = 2.0 * pi * harmonic / length
            vphase = 2.0 * pi * frequency / k
            resonant_energy = 0.5 * M_D * vphase**2 / e
            B = resonant_B(2.0 * pi * frequency, float(ne))
            for fraction in ex["coherent_power_fractions"]:
                power = float(fraction) * float(ex["absorbed_power_W"])
                field_uncapped = driven_field(power, cfg)
                wavebreaking = M_D * (2.0 * pi * frequency) * vphase / e
                field = min(
                    field_uncapped,
                    float(cfg["orbit_mcc"]["wavebreaking_fraction_cap"]) * wavebreaking,
                )
                potential = field / k
                for preenergy in [0.0, 2.5, 200.0]:
                    kinetic_ceiling = (math.sqrt(preenergy) + math.sqrt(max(potential, 0.0))) ** 2
                    rows.append({
                        "electron_density_m3": float(ne),
                        "frequency_Hz": frequency,
                        "harmonic": harmonic,
                        "wavenumber_m-1": k,
                        "phase_velocity_ms": vphase,
                        "resonant_Dplus_energy_eV": resonant_energy,
                        "resonant_B_T": B,
                        "omega_pe_over_omega_ce": omega_pe(float(ne)) / (e * B / m_e),
                        "coherent_fraction": float(fraction),
                        "mode_power_W": power,
                        "uncapped_field_Vm": field_uncapped,
                        "applied_field_Vm": field,
                        "ion_wavebreaking_field_Vm": wavebreaking,
                        "potential_amplitude_V": potential,
                        "preenergy_eV": preenergy,
                        "single_pass_kinematic_ceiling_eV": kinetic_ceiling,
                        "kinematically_reaches_1keV": kinetic_ceiling >= target,
                    })
    return pd.DataFrame(rows)


def initialize_species(count: int, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    species = cfg["plasma"]["species"]
    choices = np.empty(count, dtype=np.int8)
    start = 0
    for index, item in enumerate(species):
        stop = count if index == len(species) - 1 else start + int(round(count * float(item["fraction"])))
        choices[start:stop] = index
        start = stop
    rng.shuffle(choices)
    masses = np.array([float(item["mass_u"]) * atomic_mass for item in species])
    deuterons = np.array([int(item["deuterons_per_ion"]) for item in species])
    return masses[choices], deuterons[choices]


def energies_per_deuteron(vx: np.ndarray, vz: np.ndarray, mass: np.ndarray,
                          deuterons: np.ndarray) -> np.ndarray:
    return 0.5 * mass * (vx**2 + vz**2) / (e * deuterons)


def run_orbit_case(case: dict, cfg: dict, case_index: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ex = cfg["experiment"]
    orbit = cfg["orbit_mcc"]
    density = float(case["density_m3"])
    frequency = float(case["frequency_Hz"])
    fraction = float(case["coherent_fraction"])
    preenergy = float(case["preenergy_eV"])
    count = int(case.get(
        "particles",
        orbit["refined_particles"] if case.get("refined", False) else orbit["base_particles"],
    ))
    duration = float(ex["local_observation_duration_s"])
    dt = min(float(orbit["time_step_s"]), 1.0 / (200.0 * frequency))
    steps = int(math.ceil(duration / dt))
    dt = duration / steps
    record_every = int(orbit["record_every_steps"])
    length = float(ex["chamber_length_m"])
    target = float(ex["target_energy_eV"])
    harmonic = harmonic_for_target(frequency, length, target)
    k = 2.0 * pi * harmonic / length
    omega = 2.0 * pi * frequency
    vphase = omega / k
    B0 = resonant_B(omega, density)
    magnetic_multiplier = float(case.get(
        "magnetic_multiplier",
        cfg["magnetic_profile"]["off_resonance_multiplier"] if case.get("off_resonance", False) else 1.0,
    ))
    B0 *= magnetic_multiplier
    gradient = float(cfg["magnetic_profile"]["fractional_gradient"])
    power = fraction * float(ex["absorbed_power_W"])
    field_uncapped = driven_field(power, cfg)
    wavebreaking = M_D * omega * vphase / e
    nominal_field = min(field_uncapped, float(orbit["wavebreaking_fraction_cap"]) * wavebreaking)
    field_scale = float(case.get("field_scale", 1.0))
    field0 = field_scale * nominal_field
    Ti = float(cfg["plasma"]["ion_temperature_eV"])
    rng = np.random.default_rng(int(orbit["random_seed"]) + case_index)
    mass, deuterons = initialize_species(count, cfg, rng)
    is_Dplus = deuterons == 1
    z = (np.arange(count, dtype=float) + 0.5) * length / count
    z = z[rng.permutation(count)]
    sigma = np.sqrt(e * Ti / mass)
    drift = np.sqrt(2.0 * e * preenergy * deuterons / mass)
    vz = drift + rng.normal(0.0, 1.0, count) * sigma
    vx = rng.normal(0.0, 1.0, count) * sigma
    initial_mean = float(np.mean(energies_per_deuteron(vx, vz, mass, deuterons)))
    max_energy = energies_per_deuteron(vx, vz, mass, deuterons).copy()
    nu = float(cfg["damping"]["ion_neutral_effective_s-1"])
    collisions_enabled = bool(case.get("collisions_enabled", True))
    next_collision = rng.exponential(1.0 / nu, count) if collisions_enabled else np.full(count, math.inf)
    cx_fraction = float(cfg["damping"]["charge_exchange_fraction"])
    cumulative_work_eV_per_ion = 0.0
    cumulative_open_loss_eV_per_ion = 0.0
    cumulative_collision_loss_eV_per_ion = 0.0
    exits_right = 0
    exits_above = {threshold: 0 for threshold in [1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5]}
    records = []

    def reset_particles(mask: np.ndarray, time_s: float, boundary: bool) -> None:
        nonlocal cumulative_open_loss_eV_per_ion, cumulative_collision_loss_eV_per_ion
        number = int(mask.sum())
        if number == 0:
            return
        before = energies_per_deuteron(vx[mask], vz[mask], mass[mask], deuterons[mask])
        local_sigma = np.sqrt(e * Ti / mass[mask])
        vz[mask] = drift[mask] + rng.normal(0.0, 1.0, number) * local_sigma
        vx[mask] = rng.normal(0.0, 1.0, number) * local_sigma
        after = energies_per_deuteron(vx[mask], vz[mask], mass[mask], deuterons[mask])
        change = float(np.sum(before - after) / count)
        if boundary:
            cumulative_open_loss_eV_per_ion += change
        else:
            cumulative_collision_loss_eV_per_ion += change
        if collisions_enabled:
            next_collision[mask] = time_s + rng.exponential(1.0 / nu, number)

    for step in range(steps + 1):
        time_s = step * dt
        energy = energies_per_deuteron(vx, vz, mass, deuterons)
        np.maximum(max_energy, energy, out=max_energy)
        if step % record_every == 0 or step == steps:
            dplus_energy = energy[is_Dplus]
            records.append({
                "case": case["name"],
                "time_s": time_s,
                "field_envelope_Vm": field0 * min(1.0, (time_s * frequency / 2.0) ** 2),
                "mean_energy_per_deuteron_eV": float(np.mean(energy)),
                "Dplus_p99_eV": float(np.quantile(dplus_energy, 0.99)),
                "Dplus_p999_eV": float(np.quantile(dplus_energy, 0.999)),
                "Dplus_max_eV": float(np.max(dplus_energy)),
                "Dplus_fraction_above_1keV": float(np.mean(dplus_energy >= 1.0e3)),
            })
        if step == steps:
            break

        old_vz = vz.copy()
        envelope = np.sin(pi * z / length) ** 2
        ramp = min(1.0, (time_s * frequency / 2.0) ** 2)
        Ez = field0 * ramp * envelope * np.sin(k * z - omega * time_s)
        B = B0 * (1.0 + gradient * (2.0 * z / length - 1.0))
        q_over_m = e / mass
        vmz = vz + q_over_m * Ez * dt / 2.0
        vmx = vx
        rotation = q_over_m * B * dt / 2.0
        twice = 2.0 * rotation / (1.0 + rotation**2)
        vpx = vmx - vmz * rotation
        vpz = vmz + vmx * rotation
        vx = vmx - vpz * twice
        vz = vmz + vpx * twice + q_over_m * Ez * dt / 2.0
        cumulative_work_eV_per_ion += float(np.sum(Ez * 0.5 * (old_vz + vz) * dt / deuterons) / count)
        z += vz * dt

        right = z >= length
        left = z < 0.0
        if np.any(right):
            exit_energy = energies_per_deuteron(vx[right], vz[right], mass[right], deuterons[right])
            exits_right += int(right.sum())
            for threshold in exits_above:
                exits_above[threshold] += int(np.sum(exit_energy >= threshold))
        boundary = right | left
        if np.any(boundary):
            z[boundary] %= length
            reset_particles(boundary, time_s, True)

        due = next_collision <= time_s
        if np.any(due):
            charge_exchange = due & (rng.random(count) < cx_fraction)
            if np.any(charge_exchange):
                # A resonant charge-exchange event replaces the fast ion by a
                # newly born slow ion in the tracked charged population.
                saved_drift = drift[charge_exchange].copy()
                drift[charge_exchange] = 0.0
                reset_particles(charge_exchange, time_s, False)
                drift[charge_exchange] = saved_drift
            elastic = due & ~charge_exchange
            if np.any(elastic):
                speed = np.sqrt(vx[elastic] ** 2 + vz[elastic] ** 2)
                angle = rng.uniform(0.0, 2.0 * pi, int(elastic.sum()))
                vx[elastic] = speed * np.sin(angle)
                vz[elastic] = speed * np.cos(angle)
                next_collision[elastic] = time_s + rng.exponential(1.0 / nu, int(elastic.sum()))

    final_energy = energies_per_deuteron(vx, vz, mass, deuterons)
    dplus_final = final_energy[is_Dplus]
    dplus_maximum = max_energy[is_Dplus]
    mean_final = float(np.mean(final_energy))
    balance_residual = (
        initial_mean + cumulative_work_eV_per_ion
        - cumulative_open_loss_eV_per_ion - cumulative_collision_loss_eV_per_ion
        - mean_final
    )
    ion_count = density * active_volume(cfg)
    ion_loading_power = cumulative_work_eV_per_ion * e * ion_count / duration
    minimum_counts = int(orbit["minimum_tail_counts"])
    resolved_fraction = minimum_counts / int(is_Dplus.sum())
    neutral_damping_power = power * field_scale**2
    total_required_power = neutral_damping_power + ion_loading_power
    required_volume_fraction = (
        min(1.0, power / total_required_power)
        if power > 0.0 and total_required_power > 0.0 else 1.0
    )
    tail_count = int(np.sum(dplus_final >= 1.0e3))
    result = {
        "case": case["name"],
        "model": "open prescribed-wave physical-ion-orbit PIC/MCC predictor",
        "self_consistent_wave": False,
        "electron_density_m3": density,
        "particles": count,
        "Dplus_particles": int(is_Dplus.sum()),
        "duration_s": duration,
        "time_step_s": dt,
        "frequency_Hz": frequency,
        "harmonic": harmonic,
        "phase_velocity_ms": vphase,
        "resonant_Dplus_energy_eV": 0.5 * M_D * vphase**2 / e,
        "center_B_T": B0,
        "omega_pe_over_omega_ce": (
            omega_pe(density) / (e * B0 / m_e) if B0 > 0.0 else math.inf
        ),
        "field_Vm": field0,
        "field_scale_from_unloaded_solution": field_scale,
        "potential_amplitude_V": field0 / k,
        "mode_power_W": power,
        "preenergy_eV": preenergy,
        "collisions_enabled": collisions_enabled,
        "final_mean_energy_per_deuteron_eV": mean_final,
        "Dplus_final_p99_eV": float(np.quantile(dplus_final, 0.99)),
        "Dplus_final_p999_eV": float(np.quantile(dplus_final, 0.999)),
        "Dplus_final_max_eV": float(np.max(dplus_final)),
        "Dplus_ever_max_eV": float(np.max(dplus_maximum)),
        "Dplus_final_fraction_above_1keV": float(np.mean(dplus_final >= 1.0e3)),
        "Dplus_final_count_above_1keV": tail_count,
        "Dplus_ever_fraction_above_1keV": float(np.mean(dplus_maximum >= 1.0e3)),
        "Dplus_final_fraction_above_3keV": float(np.mean(dplus_final >= 3.0e3)),
        "Dplus_final_fraction_above_10keV": float(np.mean(dplus_final >= 1.0e4)),
        "Dplus_final_fraction_above_30keV": float(np.mean(dplus_final >= 3.0e4)),
        "Dplus_final_fraction_above_100keV": float(np.mean(dplus_final >= 1.0e5)),
        "resolved_tail_fraction_for_30_counts": resolved_fraction,
        "zero_count_95pct_upper_fraction": 3.0 / int(is_Dplus.sum()),
        "right_boundary_crossings": exits_right,
        "right_exit_fraction_above_1keV": exits_above[1.0e3] / max(exits_right, 1),
        "right_exit_fraction_above_3keV": exits_above[3.0e3] / max(exits_right, 1),
        "right_exit_fraction_above_10keV": exits_above[1.0e4] / max(exits_right, 1),
        "right_exit_fraction_above_30keV": exits_above[3.0e4] / max(exits_right, 1),
        "right_exit_fraction_above_100keV": exits_above[1.0e5] / max(exits_right, 1),
        "wave_work_eV_per_initial_ion": cumulative_work_eV_per_ion,
        "collision_loss_eV_per_initial_ion": cumulative_collision_loss_eV_per_ion,
        "open_boundary_loss_eV_per_initial_ion": cumulative_open_loss_eV_per_ion,
        "energy_balance_residual_eV_per_ion": balance_residual,
        "energy_balance_relative_error": abs(balance_residual) / max(abs(cumulative_work_eV_per_ion), 1.0),
        "ion_loading_power_W": ion_loading_power,
        "neutral_damping_power_W": neutral_damping_power,
        "total_required_mode_power_W": total_required_power,
        "loading_to_mode_power_ratio": ion_loading_power / max(power, 1.0e-30),
        "total_required_to_available_power_ratio": total_required_power / max(power, 1.0e-30),
        "power_consistent": total_required_power <= power if power > 0.0 else abs(total_required_power) < 1.0e-9,
        "maximum_active_volume_fraction_at_available_power": required_volume_fraction,
        "maximum_flux_tube_radius_m_at_available_power": float(ex["radius_m"]) * math.sqrt(required_volume_fraction),
        "tail_statistically_resolved": tail_count >= minimum_counts,
        "localized_channel_screen_pass": tail_count >= minimum_counts and required_volume_fraction < 1.0,
    }

    bins = np.geomspace(1.0e-2, 1.0e5, 141)
    histogram, edges = np.histogram(dplus_final, bins=bins)
    spectrum = pd.DataFrame({
        "case": case["name"],
        "energy_eV": np.sqrt(edges[:-1] * edges[1:]),
        "probability_per_log_bin": histogram / max(histogram.sum(), 1),
    })
    return result, pd.DataFrame(records), spectrum


def make_summary(audit: pd.DataFrame, scan: pd.DataFrame, cases: pd.DataFrame) -> dict:
    refined = cases[cases.case == "low_density_pre200_3pct_refined"].iloc[0]
    best = cases.sort_values("Dplus_final_fraction_above_1keV", ascending=False).iloc[0]
    localized = cases[cases.localized_channel_screen_pass]
    best_localized = (
        localized.sort_values("maximum_flux_tube_radius_m_at_available_power", ascending=False).iloc[0].to_dict()
        if not localized.empty else None
    )
    resonance_names = {
        "B0": "low_density_pre200_B0_refined",
        "Bres": "low_density_pre200_3pct_refined",
        "2Bres": "low_density_pre200_2B_refined",
    }
    resonance_rows = {
        label: cases[cases.case == name].iloc[0]
        for label, name in resonance_names.items() if np.any(cases.case == name)
    }
    resonance_test = None
    if len(resonance_rows) == 3:
        p_res = float(resonance_rows["Bres"].Dplus_final_fraction_above_1keV)
        n_res = int(resonance_rows["Bres"].Dplus_particles)
        controls = []
        for label in ["B0", "2Bres"]:
            row = resonance_rows[label]
            p_control = float(row.Dplus_final_fraction_above_1keV)
            n_control = int(row.Dplus_particles)
            sigma = math.sqrt(
                p_res * (1.0 - p_res) / n_res
                + p_control * (1.0 - p_control) / n_control
            )
            controls.append({
                "control": label,
                "resonant_fraction": p_res,
                "control_fraction": p_control,
                "difference_sigma": (p_res - p_control) / max(sigma, 1.0e-30),
            })
        resonance_test = {
            "comparisons": controls,
            "three_sigma_resonance_specific": all(item["difference_sigma"] >= 3.0 for item in controls),
        }
    resonance_specific = bool(
        resonance_test is not None and resonance_test["three_sigma_resonance_specific"]
    )
    return {
        "scope": "lower-hybrid channel observation; prescribed-wave MCC predictor plus explicit EM-PIC audit",
        "full_explicit_EM_PIC_run_completed": False,
        "reason_full_run_not_local": "Maxwell-Debye CFL and rare-tail statistics require HPC/Smilei",
        "resource_audit_current_density": audit.sort_values("electron_density_m3").iloc[-1].to_dict(),
        "best_prescribed_wave_case": best.to_dict(),
        "best_localized_power_screen_case": best_localized,
        "refined_reference_case": refined.to_dict(),
        "magnetic_resonance_control": resonance_test,
        "observation": {
            "prescribed_wave_orbit_accessibility_seen": bool(best.Dplus_ever_fraction_above_1keV > 0.0),
            "stationary_1keV_tail_seen": bool(best.Dplus_final_fraction_above_1keV > 0.0),
            "10keV_tail_seen": bool(cases.Dplus_final_fraction_above_10keV.max() > 0.0),
            "100keV_tail_seen": bool(cases.Dplus_final_fraction_above_100keV.max() > 0.0),
            "power_consistent_case_exists": bool((cases.power_consistent & (cases.Dplus_final_fraction_above_1keV > 0.0)).any()),
            "localized_power_screen_case_exists": bool(not localized.empty),
            "magnetic_resonance_specific_at_3sigma": resonance_specific,
            "lower_hybrid_channel_confirmed": bool(resonance_specific and (cases.power_consistent & (cases.Dplus_final_fraction_above_1keV > 0.0)).any()),
        },
        "interpretation": "A tail in the predictor demonstrates orbit accessibility only; self-consistent EM-PIC is required to confirm wave survival under ion loading.",
    }


def make_figure(cases: pd.DataFrame, histories: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ordered = cases.sort_values("Dplus_final_fraction_above_1keV")
    axes[0].barh(ordered.case, 100.0 * ordered.Dplus_final_fraction_above_1keV)
    axes[0].set(xlabel="финальная доля D⁺ выше 1 кэВ, %", title="Орбитальный PIC/MCC-предиктор")
    for name in ["current_density_pre200_3pct", "low_density_pre200_3pct", "low_density_pre200_10pct", "low_density_pre200_3pct_refined"]:
        group = histories[histories.case == name]
        axes[1].plot(group.time_s * 1.0e6, group.Dplus_p999_eV, label=name)
    axes[1].axhline(1.0e3, color="k", ls="--", lw=1)
    axes[1].set(xlabel="время, мкс", ylabel="99.9-й процентиль D⁺, эВ", title="Рост редкого хвоста")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_level47.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.config.resolve().parent
    cfg = load_yaml(args.config)
    selected_regime(root, cfg)  # validates continuity with level 4.6
    args.output.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    audit = explicit_pic_audit(cfg)
    scan = frequency_power_scan(cfg)
    case_rows, histories, spectra = [], [], []
    for index, case in enumerate(cfg["orbit_mcc"]["cases"]):
        result, history, spectrum = run_orbit_case(case, cfg, index)
        case_rows.append(result)
        histories.append(history)
        spectra.append(spectrum)
        print(f"{case['name']}: p>1keV={result['Dplus_final_fraction_above_1keV']:.3e}, "
              f"max={result['Dplus_final_max_eV']:.1f} eV, loading/P={result['loading_to_mode_power_ratio']:.2f}")
    cases = pd.DataFrame(case_rows)
    history_table = pd.concat(histories, ignore_index=True)
    spectrum_table = pd.concat(spectra, ignore_index=True)
    summary = make_summary(audit, scan, cases)

    audit.to_csv(args.output / "level47_explicit_pic_resource_audit.csv", index=False)
    scan.to_csv(args.output / "level47_frequency_power_scan.csv", index=False)
    cases.to_csv(args.output / "level47_orbit_mcc_cases.csv", index=False)
    history_table.to_csv(args.output / "level47_orbit_mcc_history.csv", index=False)
    spectrum_table.to_csv(args.output / "level47_orbit_mcc_spectra.csv", index=False)
    with (args.output / "level47_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    make_figure(cases, history_table, root / "figures" / "22_level47_lower_hybrid_tail.png")


if __name__ == "__main__":
    main()
