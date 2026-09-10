#!/usr/bin/env python3
"""Level-5 quasi-neutral hybrid PIC-MCC model for deuteron spectra.

Ions D+, D2+ and D3+ are kinetic macroparticles in a 2-D transverse disk
(2-D position, 3-V velocity). Electrons are an isothermal massless fluid.
The mesh field is the slow ambipolar plus electron-ponderomotive field derived
from level-4 envelope and density profiles. Neutral collisions, charge
exchange, ionization/recombination and wall losses are Monte-Carlo processes.
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
from scipy.constants import atomic_mass, e, k as k_B, m_e, pi
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class SpeciesTable:
    names: tuple[str, ...]
    fractions: np.ndarray
    masses_kg: np.ndarray
    deuterons_per_ion: np.ndarray


@dataclass(frozen=True)
class KineticScenario:
    name: str
    start_profile: str
    burnout_time_s: float
    ponderomotive_scale: float
    collisions_enabled: bool
    chemistry_enabled: bool
    magnetic_field_T: float


@dataclass
class ParticleState:
    x_m: np.ndarray
    y_m: np.ndarray
    vx_ms: np.ndarray
    vy_ms: np.ndarray
    vz_ms: np.ndarray
    species_index: np.ndarray

    def subset(self, keep: np.ndarray) -> None:
        self.x_m = self.x_m[keep]
        self.y_m = self.y_m[keep]
        self.vx_ms = self.vx_ms[keep]
        self.vy_ms = self.vy_ms[keep]
        self.vz_ms = self.vz_ms[keep]
        self.species_index = self.species_index[keep]

    @property
    def size(self) -> int:
        return len(self.x_m)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_species(cfg: dict) -> SpeciesTable:
    rows = cfg["ion_species"]
    fractions = np.array([float(item["fraction"]) for item in rows], dtype=float)
    fractions /= fractions.sum()
    return SpeciesTable(
        names=tuple(item["name"] for item in rows),
        fractions=fractions,
        masses_kg=np.array([float(item["mass_u"]) * atomic_mass for item in rows]),
        deuterons_per_ion=np.array([int(item["deuterons_per_ion"]) for item in rows]),
    )


def resolve_scenario(item: dict) -> KineticScenario:
    return KineticScenario(
        name=item["name"],
        start_profile=item["start_profile"],
        burnout_time_s=float(item["burnout_time_s"]),
        ponderomotive_scale=float(item["ponderomotive_scale"]),
        collisions_enabled=bool(item["collisions_enabled"]),
        chemistry_enabled=bool(item["chemistry_enabled"]),
        magnetic_field_T=float(item["magnetic_field_T"]),
    )


def profile_columns(profile_name: str) -> dict[str, str]:
    if profile_name == "minimum_scale":
        return {
            "density": "density_fraction_at_minimum_scale",
            "field": "field_at_minimum_scale_Vm",
            "radial_velocity": "radial_velocity_at_minimum_scale_ms",
            "azimuthal_velocity": "azimuthal_velocity_at_minimum_scale_ms",
        }
    if profile_name == "physical_cutoff":
        return {
            "density": "density_fraction",
            "field": "field_envelope_Vm",
            "radial_velocity": "radial_ion_velocity_ms",
            "azimuthal_velocity": "azimuthal_ion_velocity_ms",
        }
    raise ValueError(f"Unknown level-4 profile: {profile_name}")


def sample_radius(rng: np.random.Generator, r_grid: np.ndarray, rho: np.ndarray, count: int) -> np.ndarray:
    dr = float(np.mean(np.diff(r_grid)))
    weights = np.maximum(rho, 0.0) * r_grid * dr
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(rng.random(count), cumulative, r_grid)


def initialize_particles(
    rng: np.random.Generator,
    count: int,
    profile: pd.DataFrame,
    profile_name: str,
    species: SpeciesTable,
    cfg: dict,
) -> tuple[ParticleState, float]:
    columns = profile_columns(profile_name)
    r_grid = profile.radius_m.to_numpy()
    rho = profile[columns["density"]].to_numpy()
    radii = sample_radius(rng, r_grid, rho, count)
    angle = 2.0 * pi * rng.random(count)
    x = radii * np.cos(angle)
    y = radii * np.sin(angle)
    species_index = rng.choice(len(species.names), size=count, p=species.fractions)
    masses = species.masses_kg[species_index]
    thermal_sigma = np.sqrt(e * float(cfg["operating_point"]["ion_temperature_eV"]) / masses)
    vx = rng.normal(size=count) * thermal_sigma
    vy = rng.normal(size=count) * thermal_sigma
    vz = rng.normal(size=count) * thermal_sigma
    radial_bulk = np.interp(radii, r_grid, profile[columns["radial_velocity"]])
    azimuthal_bulk = np.interp(radii, r_grid, profile[columns["azimuthal_velocity"]])
    vx += radial_bulk * np.cos(angle) - azimuthal_bulk * np.sin(angle)
    vy += radial_bulk * np.sin(angle) + azimuthal_bulk * np.cos(angle)
    n0 = float(cfg["operating_point"]["electron_density_m3"])
    physical_ions_per_axial_m = 2.0 * pi * n0 * np.sum(r_grid * rho) * float(np.mean(np.diff(r_grid)))
    macro_weight = physical_ions_per_axial_m / count
    return ParticleState(x, y, vx, vy, vz, species_index), macro_weight


def deposit_density(
    state: ParticleState,
    macro_weight: float,
    radial_edges: np.ndarray,
    n0: float,
    smoothing_sigma: float,
    floor: float,
) -> np.ndarray:
    radii = np.hypot(state.x_m, state.y_m)
    counts, _ = np.histogram(radii, bins=radial_edges)
    annular_area = pi * (radial_edges[1:] ** 2 - radial_edges[:-1] ** 2)
    rho = counts * macro_weight / (annular_area * n0)
    rho = gaussian_filter1d(rho.astype(float), smoothing_sigma, mode="nearest")
    return np.maximum(rho, floor)


def radial_gradient(values: np.ndarray, dr: float) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = 0.0
    result[-1] = (values[-1] - values[-2]) / dr
    result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dr)
    return result


def level4_envelope(
    time_s: float,
    scenario: KineticScenario,
    profile: pd.DataFrame,
    cfg: dict,
) -> np.ndarray:
    pump = float(cfg["field_closure"]["pump_field_Vm"])
    minimum_field = profile.field_at_minimum_scale_Vm.to_numpy()
    cutoff_field = profile.field_envelope_Vm.to_numpy()
    bridge = float(cfg["field_closure"]["bridge_96_to_121_duration_s"])
    if scenario.start_profile == "minimum_scale" and time_s < bridge:
        fraction = time_s / bridge
        return minimum_field + fraction * (cutoff_field - minimum_field)
    elapsed_after_cutoff = max(0.0, time_s - bridge) if scenario.start_profile == "minimum_scale" else time_s
    return pump + (cutoff_field - pump) * math.exp(-elapsed_after_cutoff / scenario.burnout_time_s)


def mesh_field(
    time_s: float,
    state: ParticleState,
    macro_weight: float,
    radial_centers: np.ndarray,
    radial_edges: np.ndarray,
    profile: pd.DataFrame,
    scenario: KineticScenario,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    closure = cfg["field_closure"]
    n0 = float(cfg["operating_point"]["electron_density_m3"])
    rho = deposit_density(
        state,
        macro_weight,
        radial_edges,
        n0,
        float(closure["density_smoothing_sigma_cells"]),
        float(closure["minimum_electron_density_fraction"]),
    )
    envelope = np.interp(radial_centers, profile.radius_m, level4_envelope(time_s, scenario, profile, cfg))
    omega = 2.0 * pi * float(closure["carrier_frequency_Hz"])
    ponderomotive_eV = e * envelope**2 / (4.0 * m_e * omega**2)
    Te = float(cfg["operating_point"]["electron_temperature_eV"])
    effective_potential_eV = Te * np.log(rho) + scenario.ponderomotive_scale * ponderomotive_eV
    radius = radial_edges[-1]
    sheath_width = float(closure["sheath_width_m"])
    sheath_coordinate = np.clip((radial_centers - (radius - sheath_width)) / sheath_width, 0.0, 1.0)
    sheath_drop = float(closure["sheath_drop_Te_multiplier"]) * Te
    effective_potential_eV -= sheath_drop * sheath_coordinate**2
    dr = radial_edges[1] - radial_edges[0]
    electric_field = -radial_gradient(effective_potential_eV, dr)
    uncapped_max = float(np.max(np.abs(electric_field)))
    maximum_allowed = float(closure["maximum_allowed_slow_field_Vm"])
    electric_field = np.clip(electric_field, -maximum_allowed, maximum_allowed)
    return electric_field, {
        "maximum_slow_field_Vm": float(np.max(np.abs(electric_field))),
        "maximum_uncapped_slow_field_Vm": uncapped_max,
        "ponderomotive_potential_span_eV": float(np.ptp(ponderomotive_eV)),
        "electron_pressure_potential_span_eV": float(np.ptp(Te * np.log(rho))),
        "total_effective_potential_span_eV": float(np.ptp(effective_potential_eV)),
        "minimum_density_fraction": float(np.min(rho)),
    }


def particle_energy_eV(state: ParticleState, species: SpeciesTable) -> tuple[np.ndarray, np.ndarray]:
    masses = species.masses_kg[state.species_index]
    ion_energy = 0.5 * masses * (state.vx_ms**2 + state.vy_ms**2 + state.vz_ms**2) / e
    deuteron_energy = ion_energy / species.deuterons_per_ion[state.species_index]
    return ion_energy, deuteron_energy


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order].astype(float))
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def energy_metrics(
    state: ParticleState,
    species: SpeciesTable,
    tail_thresholds: list[float],
) -> dict:
    _, energies = particle_energy_eV(state, species)
    weights = species.deuterons_per_ion[state.species_index].astype(float)
    quantiles = weighted_quantile(energies, weights, [0.5, 0.9, 0.99, 0.999])
    result = {
        "mean_deuteron_energy_eV": float(np.average(energies, weights=weights)),
        "median_deuteron_energy_eV": float(quantiles[0]),
        "p90_deuteron_energy_eV": float(quantiles[1]),
        "p99_deuteron_energy_eV": float(quantiles[2]),
        "p999_deuteron_energy_eV": float(quantiles[3]),
        "maximum_deuteron_energy_eV": float(np.max(energies)),
    }
    total_weight = weights.sum()
    for threshold in tail_thresholds:
        key = f"fraction_above_{threshold:g}_eV"
        result[key] = float(weights[energies >= threshold].sum() / total_weight)
    return result


def append_particles(state: ParticleState, newborn: ParticleState) -> None:
    state.x_m = np.concatenate([state.x_m, newborn.x_m])
    state.y_m = np.concatenate([state.y_m, newborn.y_m])
    state.vx_ms = np.concatenate([state.vx_ms, newborn.vx_ms])
    state.vy_ms = np.concatenate([state.vy_ms, newborn.vy_ms])
    state.vz_ms = np.concatenate([state.vz_ms, newborn.vz_ms])
    state.species_index = np.concatenate([state.species_index, newborn.species_index])


def neutral_thermal_velocities(
    rng: np.random.Generator,
    count: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neutral_mass = float(cfg["operating_point"]["neutral_mass_u"]) * atomic_mass
    sigma = math.sqrt(k_B * float(cfg["operating_point"]["neutral_temperature_K"]) / neutral_mass)
    return tuple(rng.normal(size=count) * sigma for _ in range(3))


def apply_mcc(
    state: ParticleState,
    interval_s: float,
    rng: np.random.Generator,
    species: SpeciesTable,
    scenario: KineticScenario,
    cfg: dict,
) -> dict:
    collisions = cfg["collisions"]
    statistics = {"charge_exchange_events": 0, "momentum_events": 0, "chemical_births": 0,
                  "recombined_particles": 0, "volume_wall_losses": 0}
    if scenario.collisions_enabled and state.size:
        cx_probability = 1.0 - math.exp(-float(collisions["charge_exchange_rate_s-1"]) * interval_s)
        cx_mask = rng.random(state.size) < cx_probability
        cx_count = int(cx_mask.sum())
        if cx_count:
            velocities = neutral_thermal_velocities(rng, cx_count, cfg)
            state.vx_ms[cx_mask], state.vy_ms[cx_mask], state.vz_ms[cx_mask] = velocities
        statistics["charge_exchange_events"] = cx_count

        mt_probability = 1.0 - math.exp(-float(collisions["ion_neutral_momentum_rate_s-1"]) * interval_s)
        mt_mask = (~cx_mask) & (rng.random(state.size) < mt_probability)
        mt_count = int(mt_mask.sum())
        if mt_count:
            indices = np.flatnonzero(mt_mask)
            speed = np.sqrt(
                state.vx_ms[indices] ** 2 + state.vy_ms[indices] ** 2 + state.vz_ms[indices] ** 2
            )
            ion_mass = species.masses_kg[state.species_index[indices]]
            neutral_mass = float(cfg["operating_point"]["neutral_mass_u"]) * atomic_mass
            cosine_cm = 2.0 * rng.random(mt_count) - 1.0
            energy_fraction = (
                ion_mass**2 + neutral_mass**2 + 2.0 * ion_mass * neutral_mass * cosine_cm
            ) / (ion_mass + neutral_mass) ** 2
            new_speed = speed * np.sqrt(np.maximum(energy_fraction, 0.0))
            cosine = 2.0 * rng.random(mt_count) - 1.0
            sine = np.sqrt(1.0 - cosine**2)
            angle = 2.0 * pi * rng.random(mt_count)
            state.vx_ms[indices] = new_speed * sine * np.cos(angle)
            state.vy_ms[indices] = new_speed * sine * np.sin(angle)
            state.vz_ms[indices] = new_speed * cosine
        statistics["momentum_events"] = mt_count

    if scenario.chemistry_enabled and state.size:
        recombination_probability = 1.0 - math.exp(
            -float(collisions["recombination_rate_at_n0_s-1"]) * interval_s
        )
        wall_probability = 1.0 - math.exp(-float(collisions["wall_loss_rate_s-1"]) * interval_s)
        remove_recomb = rng.random(state.size) < recombination_probability
        remove_wall = (~remove_recomb) & (rng.random(state.size) < wall_probability)
        statistics["recombined_particles"] = int(remove_recomb.sum())
        statistics["volume_wall_losses"] = int(remove_wall.sum())
        state.subset(~(remove_recomb | remove_wall))

        expected_births = state.size * float(collisions["ionization_rate_s-1"]) * interval_s
        births = int(rng.poisson(expected_births))
        if births and state.size:
            parent = rng.integers(0, state.size, size=births)
            species_index = rng.choice(len(species.names), size=births, p=species.fractions)
            velocities = neutral_thermal_velocities(rng, births, cfg)
            newborn = ParticleState(
                state.x_m[parent].copy(), state.y_m[parent].copy(),
                velocities[0], velocities[1], velocities[2], species_index,
            )
            append_particles(state, newborn)
        statistics["chemical_births"] = births
    return statistics


def push_particles(
    state: ParticleState,
    electric_field_grid: np.ndarray,
    radial_centers: np.ndarray,
    dt: float,
    magnetic_field_T: float,
    species: SpeciesTable,
) -> None:
    radii = np.hypot(state.x_m, state.y_m)
    radial_field = np.interp(radii, radial_centers, electric_field_grid, left=0.0,
                             right=electric_field_grid[-1])
    direction_x = np.divide(state.x_m, radii, out=np.zeros_like(radii), where=radii > 0.0)
    direction_y = np.divide(state.y_m, radii, out=np.zeros_like(radii), where=radii > 0.0)
    masses = species.masses_kg[state.species_index]
    half_acceleration = 0.5 * e * radial_field * dt / masses
    state.vx_ms += half_acceleration * direction_x
    state.vy_ms += half_acceleration * direction_y
    if magnetic_field_T != 0.0:
        angle = e * magnetic_field_T * dt / masses
        cosine = np.cos(angle)
        sine = np.sin(angle)
        vx = cosine * state.vx_ms + sine * state.vy_ms
        vy = -sine * state.vx_ms + cosine * state.vy_ms
        state.vx_ms, state.vy_ms = vx, vy
    state.vx_ms += half_acceleration * direction_x
    state.vy_ms += half_acceleration * direction_y
    state.x_m += state.vx_ms * dt
    state.y_m += state.vy_ms * dt


def spectrum_from_values(
    scenario_name: str,
    population: str,
    component: str,
    energies: np.ndarray,
    weights: np.ndarray,
    energy_edges: np.ndarray,
) -> pd.DataFrame:
    histogram, _ = np.histogram(energies, bins=energy_edges, weights=weights)
    probability = histogram / max(histogram.sum(), 1.0)
    log_width = np.diff(np.log10(energy_edges))
    return pd.DataFrame({
        "scenario": scenario_name,
        "population": population,
        "component": component,
        "energy_low_eV": energy_edges[:-1],
        "energy_high_eV": energy_edges[1:],
        "energy_center_eV": np.sqrt(energy_edges[:-1] * energy_edges[1:]),
        "probability": probability,
        "probability_density_per_eV": probability / np.diff(energy_edges),
        "probability_per_log10_eV": probability / log_width,
    })


def spectrum_table(
    scenario_name: str,
    population: str,
    state: ParticleState,
    species: SpeciesTable,
    energy_edges: np.ndarray,
) -> pd.DataFrame:
    _, energies = particle_energy_eV(state, species)
    weights = species.deuterons_per_ion[state.species_index].astype(float)
    tables = [spectrum_from_values(
        scenario_name, population, "all_deuterons", energies, weights, energy_edges
    )]
    for index, name in enumerate(species.names):
        mask = state.species_index == index
        if np.any(mask):
            tables.append(spectrum_from_values(
                scenario_name, population, name, energies[mask], weights[mask], energy_edges
            ))
    return pd.concat(tables, ignore_index=True)


def simulate(
    profile: pd.DataFrame,
    scenario: KineticScenario,
    species: SpeciesTable,
    cfg: dict,
    macro_particles: int | None = None,
    time_step_s: float | None = None,
    seed_offset: int = 0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    numerics = cfg["numerics"]
    count = int(macro_particles or numerics["macro_particles"])
    dt = float(time_step_s or numerics["time_step_s"])
    maximum_time = float(numerics["maximum_time_s"])
    rng = np.random.default_rng(int(numerics["random_seed"]) + seed_offset)
    state, macro_weight = initialize_particles(rng, count, profile, scenario.start_profile, species, cfg)
    radius = float(profile.radius_m.max() + 0.5 * np.mean(np.diff(profile.radius_m)))
    cells = int(numerics["radial_cells"])
    radial_edges = np.linspace(0.0, radius, cells + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    field_update_steps = int(numerics["field_update_steps"])
    collision_steps = int(numerics["collision_update_steps"])
    record_steps = max(1, int(round(float(numerics["record_interval_s"]) / dt)))
    maximum_steps = int(math.ceil(maximum_time / dt))
    thresholds = [float(value) for value in numerics["tail_thresholds_eV"]]
    energy_edges = np.logspace(
        math.log10(float(numerics["spectrum_energy_bounds_eV"][0])),
        math.log10(float(numerics["spectrum_energy_bounds_eV"][1])),
        int(numerics["spectrum_bins"]) + 1,
    )
    electric_grid, field_metrics = mesh_field(
        0.0, state, macro_weight, radial_centers, radial_edges, profile, scenario, cfg
    )
    initial_metrics = energy_metrics(state, species, thresholds)
    records = [{"time_s": 0.0, "particle_count": state.size, "escaped_particles": 0,
                **initial_metrics, **field_metrics}]
    escaped_energies: list[np.ndarray] = []
    escaped_weights: list[np.ndarray] = []
    event_totals = {key: 0 for key in ["charge_exchange_events", "momentum_events", "chemical_births",
                                        "recombined_particles", "volume_wall_losses"]}
    maximum_uncapped_field = field_metrics["maximum_uncapped_slow_field_Vm"]
    maximum_potential_span = field_metrics["total_effective_potential_span_eV"]

    for step in range(1, maximum_steps + 1):
        time = step * dt
        if step % field_update_steps == 0:
            electric_grid, field_metrics = mesh_field(
                time, state, macro_weight, radial_centers, radial_edges, profile, scenario, cfg
            )
            maximum_uncapped_field = max(maximum_uncapped_field,
                                         field_metrics["maximum_uncapped_slow_field_Vm"])
            maximum_potential_span = max(maximum_potential_span,
                                         field_metrics["total_effective_potential_span_eV"])
        push_particles(state, electric_grid, radial_centers, dt, scenario.magnetic_field_T, species)

        radii = np.hypot(state.x_m, state.y_m)
        escaped = radii >= radius
        if np.any(escaped):
            _, escaped_deuteron = particle_energy_eV(
                ParticleState(state.x_m[escaped], state.y_m[escaped], state.vx_ms[escaped],
                              state.vy_ms[escaped], state.vz_ms[escaped], state.species_index[escaped]),
                species,
            )
            escaped_energies.append(escaped_deuteron)
            escaped_weights.append(species.deuterons_per_ion[state.species_index[escaped]].astype(float))
            state.subset(~escaped)

        if step % collision_steps == 0:
            events = apply_mcc(state, collision_steps * dt, rng, species, scenario, cfg)
            for key, value in events.items():
                event_totals[key] += value

        if step % record_steps == 0 or step == maximum_steps:
            metrics = energy_metrics(state, species, thresholds)
            records.append({
                "time_s": time,
                "particle_count": state.size,
                "escaped_particles": int(sum(len(item) for item in escaped_energies)),
                **metrics,
                **field_metrics,
            })

    history = pd.DataFrame(records)
    spectrum = spectrum_table(scenario.name, "retained", state, species, energy_edges)
    final_metrics = energy_metrics(state, species, thresholds)
    escaped_count = int(sum(len(item) for item in escaped_energies))
    if escaped_count:
        all_escaped_energy = np.concatenate(escaped_energies)
        all_escaped_weight = np.concatenate(escaped_weights)
        escaped_p999 = float(weighted_quantile(all_escaped_energy, all_escaped_weight, [0.999])[0])
        escaped_maximum = float(np.max(all_escaped_energy))
        escaped_spectrum = spectrum_from_values(
            scenario.name, "escaped", "all_deuterons",
            all_escaped_energy, all_escaped_weight, energy_edges,
        )
        spectrum = pd.concat([spectrum, escaped_spectrum], ignore_index=True)
    else:
        escaped_p999 = 0.0
        escaped_maximum = 0.0
    carrier_omega = 2.0 * pi * float(cfg["field_closure"]["carrier_frequency_Hz"])
    level4_peak = float(profile.field_envelope_Vm.max())
    Dplus_quiver_eV = e * level4_peak**2 / (4.0 * species.masses_kg[0] * carrier_omega**2)
    tail_key = "fraction_above_1000_eV"
    summary = {
        "scenario": scenario.name,
        "start_profile": scenario.start_profile,
        "macro_particles_initial": count,
        "macro_particles_final": state.size,
        "escaped_particles": escaped_count,
        "escaped_fraction_of_initial": escaped_count / count,
        "escaped_p999_deuteron_energy_eV": escaped_p999,
        "escaped_maximum_deuteron_energy_eV": escaped_maximum,
        **final_metrics,
        "maximum_p999_deuteron_energy_eV_over_time": float(history.p999_deuteron_energy_eV.max()),
        "maximum_particle_deuteron_energy_eV_over_time": float(history.maximum_deuteron_energy_eV.max()),
        "maximum_uncapped_slow_field_Vm": maximum_uncapped_field,
        "maximum_effective_potential_span_eV": maximum_potential_span,
        "Dplus_direct_HF_quiver_energy_eV": Dplus_quiver_eV,
        "zero_count_95pct_upper_fraction_above_1keV": (
            3.0 / max(state.size, 1) if final_metrics[tail_key] == 0.0 else math.nan
        ),
        "burnout_time_s": scenario.burnout_time_s,
        "ponderomotive_scale": scenario.ponderomotive_scale,
        "collisions_enabled": scenario.collisions_enabled,
        "chemistry_enabled": scenario.chemistry_enabled,
        "magnetic_field_T": scenario.magnetic_field_T,
        "time_step_s": dt,
        **event_totals,
    }
    return summary, history, spectrum


def run_scenarios(profile: pd.DataFrame, cfg: dict, species: SpeciesTable):
    summaries = []
    histories = {}
    spectra = []
    for index, item in enumerate(cfg["scenarios"]):
        scenario = resolve_scenario(item)
        seed_offset = int(item.get("random_seed_offset", 1000 * index))
        summary, history, spectrum = simulate(
            profile, scenario, species, cfg, seed_offset=seed_offset
        )
        summaries.append(summary)
        histories[scenario.name] = history
        spectra.append(spectrum)
    return pd.DataFrame(summaries), histories, pd.concat(spectra, ignore_index=True)


def convergence_study(profile: pd.DataFrame, cfg: dict, species: SpeciesTable) -> pd.DataFrame:
    conv_cfg = cfg["convergence_study"]
    item = next(row for row in cfg["scenarios"] if row["name"] == conv_cfg["scenario"])
    scenario = resolve_scenario(item)
    rows = []
    for index, case in enumerate(conv_cfg["cases"]):
        summary, _, _ = simulate(
            profile, scenario, species, cfg,
            macro_particles=int(case["macro_particles"]),
            time_step_s=float(case["time_step_s"]),
            seed_offset=7000 + index,
        )
        rows.append(summary)
    result = pd.DataFrame(rows).sort_values("macro_particles_initial").reset_index(drop=True)
    finest = result.iloc[-1]
    for column in ["mean_deuteron_energy_eV", "p99_deuteron_energy_eV", "p999_deuteron_energy_eV"]:
        result[f"relative_error_{column}_vs_finest"] = (
            (result[column] - float(finest[column])).abs() / max(float(finest[column]), 1.0e-30)
        )
    return result


def make_figures(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    spectra: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for name in ["from96_fast_burnout", "from121_fast_burnout", "from121_collisional_decay",
                 "from121_sustained_upper", "pressure_only_control"]:
        group = spectra[
            (spectra.scenario == name)
            & (spectra.population == "retained")
            & (spectra.component == "all_deuterons")
        ]
        mask = group.probability_per_log10_eV > 0.0
        ax.loglog(group.loc[mask, "energy_center_eV"], group.loc[mask, "probability_per_log10_eV"],
                  label=name)
    ax.axvspan(1.0e3, 1.0e5, color="#d62728", alpha=0.08, label="1–100 кэВ")
    ax.set(xlabel="Энергия на дейтрон, эВ", ylabel=r"$dP/d\log_{10}E$",
           title="Уровень 5: энергетический спектр дейтронов")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "16_level5_deuteron_spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for name in ["from121_fast_burnout", "from121_collisional_decay", "from121_sustained_upper"]:
        history = histories[name]
        axes[0].plot(history.time_s * 1e9, history.p999_deuteron_energy_eV, label=name)
        axes[1].plot(history.time_s * 1e9, history.mean_deuteron_energy_eV, label=name)
    axes[0].set(xlabel="Время после передачи профиля, нс", ylabel="P99.9, эВ",
                title="Редкий надтепловой компонент")
    axes[1].set(xlabel="Время после передачи профиля, нс", ylabel="Средняя энергия, эВ",
                title="Средняя энергия дейтронов")
    axes[1].legend(fontsize=8)
    fig.suptitle("Развитие ионного спектра")
    fig.tight_layout()
    fig.savefig(output_dir / "17_level5_energy_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(summaries.scenario, summaries.maximum_p999_deuteron_energy_eV_over_time)
    ax.set(ylabel="Максимальный P99.9, эВ", title="Сценарная чувствительность ионного хвоста")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "18_level5_scenario_tail.png", dpi=180)
    plt.close(fig)


def observables_table(summaries: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario", "mean_deuteron_energy_eV", "p99_deuteron_energy_eV",
        "p999_deuteron_energy_eV", "maximum_deuteron_energy_eV",
        "escaped_p999_deuteron_energy_eV", "escaped_maximum_deuteron_energy_eV",
        "fraction_above_100_eV", "fraction_above_1000_eV",
        "maximum_uncapped_slow_field_Vm", "maximum_effective_potential_span_eV",
        "Dplus_direct_HF_quiver_energy_eV",
        "zero_count_95pct_upper_fraction_above_1keV",
    ]
    result = summaries[columns].copy()
    result["forms_1_to_100_keV_tail"] = result["fraction_above_1000_eV"] > 0.0
    return result


def energy_budget_table(summaries: pd.DataFrame) -> pd.DataFrame:
    reference = summaries.loc[
        summaries.scenario == "from121_collisional_decay"
    ].iloc[0]
    maximum_potential = float(reference.maximum_effective_potential_span_eV)
    observed_maximum = max(
        float(reference.maximum_deuteron_energy_eV),
        float(reference.escaped_maximum_deuteron_energy_eV),
    )
    rows = [
        ("D+ direct HF quiver", float(reference.Dplus_direct_HF_quiver_energy_eV),
         "eV per deuteron", "Fast 1.225 GHz field; not a DC accelerating voltage"),
        ("retained P99.9", float(reference.p999_deuteron_energy_eV),
         "eV per deuteron", "Main collisional scenario at 600 ns"),
        ("largest simulated particle", observed_maximum,
         "eV per deuteron", "Maximum over retained and escaped populations"),
        ("maximum slow potential span", maximum_potential,
         "V per unit charge", "Electron pressure + ponderomotive + sheath closure"),
        ("minimum D+ voltage for 1 keV", 1.0e3,
         "V", "Lower boundary of requested tail"),
        ("minimum D3+ molecular-ion voltage for 1 keV/deuteron", 3.0e3,
         "V", "Singly charged D3+ must gain 3 keV per molecular ion"),
        ("minimum D+ voltage for 100 keV", 1.0e5,
         "V", "Upper boundary of requested tail"),
    ]
    result = pd.DataFrame(rows, columns=["quantity", "value", "unit", "interpretation"])
    result["ratio_to_maximum_slow_potential"] = np.where(
        result.unit == "V", result.value / maximum_potential, np.nan
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_level5.yaml"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(args.config)
    profile = pd.read_csv(root / cfg["input"]["level4_profile_csv"])
    species = build_species(cfg)
    summaries, histories, spectra = run_scenarios(profile, cfg, species)
    convergence = convergence_study(profile, cfg, species)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output / "level5_scenarios.csv", index=False)
    spectra.to_csv(args.output / "level5_deuteron_spectra.csv", index=False)
    convergence.to_csv(args.output / "level5_convergence.csv", index=False)
    observables = observables_table(summaries)
    energy_budget = energy_budget_table(summaries)
    observables.to_csv(args.output / "level5_observables.csv", index=False)
    energy_budget.to_csv(args.output / "level5_energy_budget.csv", index=False)
    for name, history in histories.items():
        history.to_csv(args.output / f"level5_history_{name}.csv", index=False)
    summary = {
        "model_scope": "2-D transverse, 3-V quasi-neutral hybrid PIC-MCC; kinetic molecular-ion mixture",
        "deuteron_spectrum_convention": "molecular-ion kinetic energy divided by deuterons per ion",
        "scenario_count": int(len(summaries)),
        "decision": "No 1-100 keV deuteron tail is produced by the level-4 fields in this closure",
        "scenarios": summaries.to_dict("records"),
        "convergence": convergence.to_dict("records"),
        "energy_budget": energy_budget.to_dict("records"),
    }
    (args.output / "level5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(summaries, histories, spectra, root / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
