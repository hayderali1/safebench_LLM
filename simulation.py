"""
Closed-loop simulation harness.

Implements:
 * a Highway-Env scenario with the continuous action space,
 * a black-box supervisor latency model (zero-order-hold delay buffer),
 * a perception-staleness model (reported state lags by tau_perc),
 * a single closed-loop run that returns the metrics defined in
   Sec. 4.5 of the paper (collision rate, h_min, intervention, etc.).

This is the minimal, CPU-only loop that powers the benchmark grid.
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np

import gymnasium as gym
import highway_env  # noqa: F401  -- registers envs with gymnasium

from .filters import SafetyFilter, Obstacle, FilterDiagnostics, LAWS


@dataclass
class RunResult:
    collided: bool
    constraint_violated: bool
    h_min: float
    avg_intervention: float
    qp_infeas_rate: float
    avg_solve_ms: float
    steps: int
    survived: bool
    avg_speed: float
    notes: str = ""


def _ego_state_from_env(env) -> np.ndarray:
    """Extract  [x, y, v, psi]  from the Highway-Env ego vehicle."""
    veh = env.unwrapped.vehicle
    return np.array([veh.position[0], veh.position[1], veh.speed, veh.heading])


def _obstacles_from_env(env, ego_state: np.ndarray, sense_radius: float = 80.0
                        ) -> List[Obstacle]:
    """All other road vehicles within  sense_radius  of the ego."""
    out: List[Obstacle] = []
    for v in env.unwrapped.road.vehicles:
        if v is env.unwrapped.vehicle:
            continue
        dx = v.position[0] - ego_state[0]
        dy = v.position[1] - ego_state[1]
        if dx*dx + dy*dy > sense_radius * sense_radius:
            continue
        # heading-aligned velocity
        vx = v.speed * np.cos(v.heading)
        vy = v.speed * np.sin(v.heading)
        out.append(Obstacle(x=v.position[0], y=v.position[1],
                            vx=vx, vy=vy, radius=2.5))
    return out


def _u_to_env_action(u: np.ndarray, env) -> np.ndarray:
    """Highway-Env continuous-action expects  [acceleration, steering]
    in [-1, 1].  Map our SI-unit  (a, delta)  to that range."""
    space = env.action_space
    a_lo, a_hi = -7.0, 5.0   # m/s^2
    d_lo, d_hi = -np.pi/6, np.pi/6
    # linear scale to [-1, 1]
    a_n = 2 * (u[0] - a_lo) / (a_hi - a_lo) - 1
    d_n = 2 * (u[1] - d_lo) / (d_hi - d_lo) - 1
    a_n = float(np.clip(a_n, space.low[0], space.high[0]))
    d_n = float(np.clip(d_n, space.low[1], space.high[1]))
    return np.array([a_n, d_n], dtype=np.float32)


# ----------------------------------------------------------------------
@dataclass
class SupervisorLatencyModel:
    """Zero-order-hold delay buffer that returns the supervisor command
    issued  T_sup  seconds in the past.  Calibrated to published
    LLM/VLM inference timings (ChatMPC, AsyncDriver, AdaDrive)."""
    T_sup: float = 0.5     # s
    dt:    float = 0.1     # s, simulation step
    jitter_std: float = 0.02

    def __post_init__(self):
        n = max(1, int(round(self.T_sup / self.dt)))
        self._buf: deque = deque([np.zeros(2)] * n, maxlen=n)

    def push_pull(self, u_nom: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Push the freshest LLM command, pop the one we should apply now."""
        # tiny jitter around the mean latency
        if self.jitter_std > 0:
            self.T_sup_actual = max(0.0, self.T_sup
                                    + rng.normal(0, self.jitter_std))
        delayed = self._buf[0]
        self._buf.append(np.asarray(u_nom, dtype=float))
        return delayed.copy()


@dataclass
class PerceptionStaleness:
    """Returns the state estimate from  tau_perc  seconds ago."""
    tau_perc: float = 0.0
    dt: float = 0.1

    def __post_init__(self):
        n = max(1, int(round(self.tau_perc / self.dt)))
        self._buf: deque = deque(maxlen=n)

    def update_and_get(self, state: np.ndarray) -> np.ndarray:
        if self._buf.maxlen == 0 or self.tau_perc < 1e-6:
            return state.copy()
        self._buf.append(state.copy())
        if len(self._buf) < self._buf.maxlen:
            return state.copy()
        return self._buf[0].copy()


# ----------------------------------------------------------------------
def make_env(scenario: str, seed: int):
    """Construct one of the five SafeBench-LLM scenarios.

    All five envs are built on Highway-Env (Leurent 2018-2026, MIT, [39]).
    Scenario behaviour beyond the env config (S4 cut-in, S5 OOD spikes)
    is injected per-step inside `run_one`; the env itself is just the
    backdrop and traffic generator."""
    if scenario == "S1":
        # Dense highway, lane change required.
        env = gym.make("highway-v0",
                       config={"action": {"type": "ContinuousAction"},
                               "duration": 30, "vehicles_count": 30,
                               "lanes_count": 4, "policy_frequency": 10,
                               "vehicles_density": 1.5})
    elif scenario == "S2":
        # Ego on on-ramp joining highway.
        env = gym.make("merge-v0",
                       config={"action": {"type": "ContinuousAction"},
                               "duration": 30, "policy_frequency": 10})
    elif scenario == "S3":
        # Unsignalised 4-way intersection with stochastic opponents.
        env = gym.make("intersection-v0",
                       config={"action": {"type": "ContinuousAction"},
                               "duration": 20, "policy_frequency": 10})
    elif scenario == "S4":
        # Adversarial cut-in: a tightly-packed highway where a lead
        # vehicle will be scripted (in `run_one`) to brake at -5 m/s^2
        # at step 10.  Density is high so the ego cannot trivially
        # change lane.
        env = gym.make("highway-v0",
                       config={"action": {"type": "ContinuousAction"},
                               "duration": 25, "vehicles_count": 25,
                               "lanes_count": 3, "policy_frequency": 10,
                               "vehicles_density": 2.0,
                               "initial_spacing": 1.5})
    elif scenario == "S5":
        # OOD stress: same as S1 but the perception layer drops or
        # spikes obstacles (handled by `ood_dropout` / `ood_spikes`
        # flags in `run_one`).
        env = gym.make("highway-v0",
                       config={"action": {"type": "ContinuousAction"},
                               "duration": 30, "vehicles_count": 30,
                               "lanes_count": 4, "policy_frequency": 10,
                               "vehicles_density": 1.5})
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    env.reset(seed=seed)
    return env


# ----------------------------------------------------------------------
def _find_lead_vehicle(env, ego_state):
    """Return the closest other vehicle ahead of the ego in the same
    or adjacent lane, used by S4 to script the adversarial brake."""
    ego = env.unwrapped.vehicle
    best, best_dx = None, float("inf")
    for v in env.unwrapped.road.vehicles:
        if v is ego:
            continue
        # ahead = positive x in ego frame
        dx = (v.position[0] - ego_state[0]) * np.cos(ego_state[3]) \
             + (v.position[1] - ego_state[1]) * np.sin(ego_state[3])
        # lateral offset: only consider near-lane
        lat = abs(v.position[1] - ego_state[1])
        if 0 < dx < best_dx and lat < 6.0:
            best, best_dx = v, dx
    return best


# ----------------------------------------------------------------------
def run_one(scenario: str,
            filter_obj: SafetyFilter,
            T_sup: float,
            tau_perc: float,
            seed: int,
            dt: float = 0.1,
            max_steps: int = 200,
            ) -> RunResult:
    """A single closed-loop run.  Returns one row of metrics.

    Scenario-specific adversarial behaviour is *not* baked into the env
    config — it is injected per-step here so the harness stays in one
    place and so each scenario's perturbation is visible/auditable.

    S4: at step 10 the closest lead vehicle is forced to a hard brake
        of -5 m/s^2 for 30 sim steps  (Sec. 4.2 of the paper).
    S5: with probability 0.15 a perception step *drops* every obstacle
        (mimicking SlowPerception-style attacks [32]); independently
        with probability 0.10 a phantom obstacle is spiked into the
        sensed list ahead of the ego.
    """
    rng = np.random.default_rng(seed)
    env = make_env(scenario, seed)
    sup = SupervisorLatencyModel(T_sup=T_sup, dt=dt)
    perc = PerceptionStaleness(tau_perc=tau_perc, dt=dt)

    interventions, h_mins, infeas, solves, speeds = [], [], [], [], []
    collided = False
    constraint_violated = False
    notes = ""

    # S4 setup: lock onto the lead vehicle once at step 0.
    s4_lead = None
    s4_brake_start = 10        # step at which the brake begins
    s4_brake_steps = 30        # how long the brake lasts
    s4_brake_a     = -5.0      # m/s^2  (Sec. 4.2 of paper)

    for step in range(max_steps):
        ego_true = _ego_state_from_env(env)
        ego_obs  = perc.update_and_get(ego_true)
        speeds.append(env.unwrapped.vehicle.speed)
        obstacles = _obstacles_from_env(env, ego_obs)

        # --- scenario-specific OOD perception model (S5) ---
        if scenario == "S5":
            if rng.random() < 0.15:
                # full perception dropout: filter sees an empty road
                obstacles = []
                notes = "OOD perception dropout"
            elif rng.random() < 0.10:
                # phantom obstacle spike 15 m ahead in ego heading
                px = ego_obs[0] + 15.0 * np.cos(ego_obs[3])
                py = ego_obs[1] + 15.0 * np.sin(ego_obs[3])
                obstacles.append(Obstacle(x=px, y=py, vx=0.0, vy=0.0,
                                          radius=2.5))
                notes = "OOD perception spike"

        # --- scenario-specific adversary (S4) ---
        if scenario == "S4":
            if step == 0:
                s4_lead = _find_lead_vehicle(env, ego_true)
            if (s4_lead is not None
                and s4_brake_start <= step < s4_brake_start + s4_brake_steps):
                # Override the lead vehicle's behaviour: directly slow it.
                s4_lead.speed = max(0.0, s4_lead.speed + s4_brake_a * dt)

        # Nominal LLM/VLM command: simple proportional cruise (target 25 m/s)
        v_target = 25.0
        a_nom = 1.0 * (v_target - ego_obs[2])
        delta_nom = 0.0
        u_nom_fresh = np.array([a_nom, delta_nom])

        # Supervisor latency: filter sees the *delayed* command
        u_nom_delayed = sup.push_pull(u_nom_fresh, rng)

        # If LAWS, notify of arrival timestamp
        if isinstance(filter_obj, LAWS):
            filter_obj.report_supervisor_arrival(step * dt)

        u_safe, diag = filter_obj.filter(ego_obs, u_nom_delayed,
                                         obstacles, dt)
        interventions.append(diag.intervention)
        h_mins.append(diag.h_min)
        solves.append(diag.solve_time_ms)
        infeas.append(0 if diag.feasible else 1)
        if diag.h_min < 0:
            constraint_violated = True

        action = _u_to_env_action(u_safe, env)
        _, _, terminated, truncated, info = env.step(action)
        if env.unwrapped.vehicle.crashed:
            collided = True
            break
        if terminated or truncated:
            break

    env.close()
    survived = not collided
    return RunResult(
        collided=collided,
        constraint_violated=constraint_violated,
        h_min=float(np.min(h_mins)) if h_mins else float("inf"),
        avg_intervention=float(np.mean(interventions)) if interventions else 0.0,
        qp_infeas_rate=float(np.mean(infeas)) if infeas else 0.0,
        avg_solve_ms=float(np.mean(solves)) if solves else 0.0,
        steps=step + 1,
        survived=survived,
        avg_speed=float(np.mean(speeds)) if speeds else 0.0,
        notes=notes,
    )
