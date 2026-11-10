"""Overdamped Langevin dynamics in one dimension.

Quick start::

    import overdamped_ld as old

    pot = old.DoubleWell.in_kT(4)                  # 4 kT barrier
    traj = old.simulate(pot, x0=-1, n_steps=200_000, n_walkers=64, stride=50, seed=0)
    states = old.assign_states(traj.x)
    k_sim = old.transition_rate(states, traj.save_interval)
    k_kramers = old.kramers_rate(pot)
"""

from .analysis import (
    assign_states,
    boltzmann_density,
    count_transitions,
    dwell_times,
    kramers_rate,
    mfpt_exact,
    populations,
    transition_rate,
)
from .integrators import Trajectory, first_passage_times, simulate
from .potentials import DoubleWell, Potential

__version__ = "0.2.0"

__all__ = [
    "DoubleWell",
    "Potential",
    "Trajectory",
    "assign_states",
    "boltzmann_density",
    "count_transitions",
    "dwell_times",
    "first_passage_times",
    "kramers_rate",
    "mfpt_exact",
    "populations",
    "simulate",
    "transition_rate",
]
