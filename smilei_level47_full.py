"""Smilei 5.1 pilot deck for the level-4.7 lower-hybrid EM-PIC run.

The single spatial coordinate x is perpendicular to Bz(x), so k_perp is
represented.  Particle momenta and all electromagnetic fields are 3-D.

IMPORTANT: Smilei's standard Collisions block below covers Coulomb collisions.
Its stock binary-collision module is not a cross-section-resolved D+/D2 neutral
charge-exchange MCC operator.  The neutral MCC rates used by plasma_level47.py
must therefore be added as a custom operator before this deck is called the
complete PIC/MCC proof.  This file is a runnable EM-PIC/Coulomb pilot, not a
claim that the local container executed the HPC calculation.
"""

from math import ceil, cos, exp, pi, sin, sqrt

# SI reference state: the low-density level-4.7 candidate.
c_si = 299792458.0
e_si = 1.602176634e-19
me_si = 9.1093837015e-31
eps0_si = 8.8541878128e-12
ne_si = 8.0e14
Te_cold_eV = 8.072408
Te_hot_eV = 300.0
hot_fraction = 0.01236
omega_r = sqrt(ne_si * e_si**2 / (eps0_si * me_si))
skin = c_si / omega_r
lambda_D_si = sqrt(eps0_si * Te_cold_eV / (ne_si * e_si))
length_si = 0.311831
length = length_si / skin
dx = 0.5 * lambda_D_si / skin
dt = 0.95 * dx
simulation_time = 20.0e-6 * omega_r

frequency_si = 1.0e6
omega_drive = 2.0 * pi * frequency_si / omega_r
B_center_si = 2.228924e-3
B_unit_si = me_si * omega_r / e_si
B_center = B_center_si / B_unit_si
B_gradient = 0.25

# The pilot is intended to locate the instability.  A statistically resolved
# 1e-5 tail with 30 counts needs about 3589 particles/cell at this grid size;
# set ppc=3589 for the production ensemble or use controlled particle splitting.
ppc = 256
patches = 32

Main(
    geometry="1Dcartesian",
    interpolation_order=2,
    cell_length=[dx],
    grid_length=[length],
    number_of_patches=[patches],
    timestep=dt,
    simulation_time=simulation_time,
    maxwell_solver="Yee",
    EM_boundary_conditions=[["silver-muller", "silver-muller"]],
    reference_angular_frequency_SI=omega_r,
    print_every=10000,
    random_seed=20260812,
)


def density_envelope(x):
    # Smooth open source; the 2% floors keep injectors numerically well posed.
    return 0.02 + 0.98 * sin(pi * x / length) ** 2


def magnetic_profile(x):
    return B_center * (1.0 + B_gradient * (2.0 * x / length - 1.0))


Species(
    name="electron_cold", position_initialization="random",
    momentum_initialization="maxwell-juettner", particles_per_cell=ppc,
    mass=1.0, charge=-1.0,
    number_density=lambda x: (1.0 - hot_fraction) * density_envelope(x),
    temperature=[Te_cold_eV / 510998.95],
    boundary_conditions=[["remove", "remove"]],
)
Species(
    name="electron_hot", position_initialization="random",
    momentum_initialization="maxwell-juettner", particles_per_cell=max(64, ppc // 4),
    mass=1.0, charge=-1.0,
    number_density=lambda x: hot_fraction * density_envelope(x),
    temperature=[Te_hot_eV / 510998.95],
    boundary_conditions=[["remove", "remove"]],
)


def add_ion(name, fraction, mass_ratio, deuterons):
    Species(
        name=name, position_initialization="random",
        momentum_initialization="maxwell-juettner", particles_per_cell=ppc,
        mass=mass_ratio, charge=1.0,
        number_density=lambda x: fraction * density_envelope(x),
        temperature=[deuterons * 0.10 / 510998.95],
        mean_velocity=[sqrt(2.0 * deuterons * 200.0 / (mass_ratio * 510998.95)), 0.0, 0.0],
        boundary_conditions=[["remove", "remove"]],
        atomic_number=1,
    )


add_ion("Dplus", 0.90, 3671.30, 1)
add_ion("D2plus", 0.05, 7342.59, 2)
add_ion("D3plus", 0.05, 11013.89, 3)

ExternalField(field="Bz", profile=magnetic_profile)
ExternalField(field="Bz_m", profile=magnetic_profile)

# A longitudinal current antenna excites Ex self-consistently.  The amplitude
# is a pilot value and must be calibrated from the measured injected Poynting
# flux; it is intentionally not replaced by a prescribed electric field.
target_field_si = 7.288e3
J_unit_si = e_si * ne_si * c_si
antenna_J = eps0_si * 2.0 * pi * frequency_si * target_field_si / J_unit_si


def antenna_xt(x, t):
    ramp = min(1.0, (t * omega_drive / (4.0 * pi)) ** 2)
    return antenna_J * ramp * sin(pi * x / length) ** 2 * cos(omega_drive * t)


Antenna(field="Jx", space_time_profile=antenna_xt)

Collisions(
    species1=["electron_cold", "electron_hot"],
    species2=["Dplus", "D2plus", "D3plus"],
    coulomb_log=14.0,
    every=20,
    debug_every=100000,
)

DiagScalar(every=10000, vars=["Utot", "Ukin", "Uelm"])
DiagFields(every=10000, fields=["Ex", "Ey", "Ez", "By", "Bz", "Rho"])
DiagParticleBinning(
    name="Dplus_energy_tail", deposited_quantity="weight",
    every=10000, species=["Dplus"],
    axes=[["ekin", 1.0e-8, 0.20, 320, "logscale", "edge_inclusive"]],
)
DiagParticleBinning(
    name="electron_populations", deposited_quantity="weight",
    every=10000, species=["electron_cold", "electron_hot"],
    axes=[["x", 0.0, length, 256], ["ekin", 1.0e-8, 0.02, 200, "logscale"]],
)
DiagTrackParticles(
    species="Dplus", every=10000,
    filter=lambda particles: particles.px > sqrt(2.0 * 1000.0 / (3671.30 * 510998.95)),
    attributes=["x", "px", "py", "pz", "Ex", "Ey", "Bz", "w"],
)

