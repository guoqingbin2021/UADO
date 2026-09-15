from __future__ import annotations

import math


def shannon_rate_bps(
    bandwidth_hz: float,
    *,
    signal_power_w: float,
    noise_interference_w: float,
) -> float:
    bandwidth = float(bandwidth_hz)
    signal = float(signal_power_w)
    noise = float(noise_interference_w)
    if bandwidth < 0 or signal < 0 or noise <= 0:
        raise ValueError("invalid Shannon-capacity arguments")
    return bandwidth * math.log2(1.0 + signal / noise)


def rotary_wing_power_w(
    speed_mps: float,
    *,
    blade_profile_power_w: float = 79.86,
    induced_power_w: float = 88.63,
    rotor_tip_speed_mps: float = 120.0,
    mean_rotor_induced_velocity_mps: float = 4.03,
    fuselage_drag_ratio: float = 0.6,
    air_density_kg_m3: float = 1.225,
    rotor_solidity: float = 0.05,
    rotor_disc_area_m2: float = 0.503,
) -> float:
    speed = float(speed_mps)
    if speed < 0:
        raise ValueError("UAV speed cannot be negative")
    profile = blade_profile_power_w * (1.0 + 3.0 * speed**2 / rotor_tip_speed_mps**2)
    nested = math.sqrt(1.0 + speed**4 / (4.0 * mean_rotor_induced_velocity_mps**4))
    induced_factor = math.sqrt(max(0.0, nested - speed**2 / (2.0 * mean_rotor_induced_velocity_mps**2)))
    induced = induced_power_w * induced_factor
    parasite = (
        0.5
        * fuselage_drag_ratio
        * air_density_kg_m3
        * rotor_solidity
        * rotor_disc_area_m2
        * speed**3
    )
    return profile + induced + parasite


def compute_energy_j(*, effective_capacitance: float, cycles: float, frequency_hz: float) -> float:
    if effective_capacitance < 0 or cycles < 0 or frequency_hz < 0:
        raise ValueError("compute-energy inputs must be non-negative")
    return float(effective_capacitance) * float(cycles) * float(frequency_hz) ** 2


def communication_energy_j(*, power_w: float, duration_s: float) -> float:
    if power_w < 0 or duration_s < 0:
        raise ValueError("communication-energy inputs must be non-negative")
    return float(power_w) * float(duration_s)

