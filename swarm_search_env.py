import numpy as np
import functools
import json
import sys
import time
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete, Dict


class SwarmSearchEnv(ParallelEnv):
    """
    Decentralized Swarm Target Search Environment
    Features: Ray-casting LiDAR, Moving Targets, Graph-Communication Topologies.
    """
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        "name": "swarm_target_search_v1"
    }

    def __init__(
        self,
        num_agents=4,
        grid_size=20,
        num_targets=3,
        sensor_range=5,
        comm_range=8,
        max_timesteps=300
    ):
        super().__init__()
        self._num_agents = num_agents
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.max_timesteps = max_timesteps
        self.num_rays = 16  # LiDAR rays

        self.possible_agents = [f"drone_{i}" for i in range(self._num_agents)]

        # State tracking
        self.agent_positions = {}
        self.target_positions = []
        self.coverage_map = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.obstacle_map = self._generate_obstacles()
        self.found_targets = set()
        self.timestep = 0

        # Episode metrics
        self._episode_rewards = {}
        self._discovery_times = {}

    def _generate_obstacles(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        obs[8:12, 8] = 1   # vertical wall
        obs[8, 8:12] = 1   # horizontal wall
        obs[4:7, 14] = 1   # second obstacle cluster
        obs[14:17, 5:7] = 1
        return obs

    @functools.lru_cache(maxsize=None)
    def _observation_space(self, agent):
        return Dict({
            "ego":              Box(low=0.0,  high=1.0,  shape=(2,),                        dtype=np.float32),
            "lidar":            Box(low=0.0,  high=1.0,  shape=(self.num_rays,),             dtype=np.float32),
            "comm_nodes":       Box(low=-1.0, high=1.0,  shape=(self._num_agents - 1, 2),   dtype=np.float32),
            "comm_adj":         Box(low=0.0,  high=1.0,  shape=(self._num_agents - 1,),     dtype=np.float32),
            "target_indicators":Box(low=0.0,  high=1.0,  shape=(self.num_targets,),         dtype=np.float32),
        })

    def observation_space(self, agent):
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def _action_space(self, agent):
        return Discrete(5)  # 0:Up 1:Down 2:Left 3:Right 4:Hover

    def action_space(self, agent):
        return self._action_space(agent)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.found_targets = set()
        self._episode_rewards = {a: 0.0 for a in self.agents}
        self._discovery_times = {}

        # Clustered spawn in top-left corner
        self.agent_positions = {
            a: np.array([
                np.random.randint(0, 4),
                np.random.randint(0, 4)
            ])
            for a in self.agents
        }

        # Targets in bottom-right region
        self.target_positions = [
            np.array([
                np.random.randint(self.grid_size - 7, self.grid_size),
                np.random.randint(self.grid_size - 7, self.grid_size)
            ])
            for _ in range(self.num_targets)
        ]

        self.coverage_map.fill(0)
        self._update_coverage()

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, actions):
        self.timestep += 1
        rewards = {a: 0.0 for a in self.agents}

        # ── 1. Agent movement + collision check ──────────────────────────────
        for agent_id, action in actions.items():
            pos = self.agent_positions[agent_id]
            new_pos = np.copy(pos)
            if   action == 0: new_pos[1] += 1
            elif action == 1: new_pos[1] -= 1
            elif action == 2: new_pos[0] -= 1
            elif action == 3: new_pos[0] += 1

            if (0 <= new_pos[0] < self.grid_size and
                    0 <= new_pos[1] < self.grid_size and
                    self.obstacle_map[new_pos[0], new_pos[1]] == 0):
                self.agent_positions[agent_id] = new_pos
            else:
                rewards[agent_id] -= 0.5   # collision penalty

            rewards[agent_id] -= 0.05       # step penalty

        # ── 2. Dynamic evading targets ────────────────────────────────────────
        for idx in range(self.num_targets):
            t_pos = self.target_positions[idx]
            closest_dist = min(
                np.linalg.norm(self.agent_positions[a] - t_pos)
                for a in self.agents
            )

            if closest_dist < self.sensor_range + 2:
                move_dir = np.array([
                    np.random.choice([-1, 0, 1]),
                    np.random.choice([-1, 0, 1])
                ])
            elif np.random.rand() < 0.1:
                move_dir = np.array([
                    np.random.choice([-1, 0, 1]),
                    np.random.choice([-1, 0, 1])
                ])
            else:
                move_dir = np.array([0, 0])

            new_t = np.clip(t_pos + move_dir, 0, self.grid_size - 1)
            if self.obstacle_map[new_t[0], new_t[1]] == 0:
                self.target_positions[idx] = new_t

        # ── 3. Coverage + dispersion rewards ─────────────────────────────────
        new_cells = self._update_coverage()
        for a in self.agents:
            rewards[a] += new_cells * 0.5
            for other_a in self.agents:
                if a != other_a:
                    dist = np.linalg.norm(
                        self.agent_positions[a] - self.agent_positions[other_a]
                    )
                    if dist < 2.0:
                        rewards[a] -= 0.5

        # ── 4. Discovery + tracking rewards ──────────────────────────────────
        currently_tracked = set()
        for a in self.agents:
            for t_idx, t_pos in enumerate(self.target_positions):
                if self._has_line_of_sight(self.agent_positions[a], t_pos, self.sensor_range):
                    currently_tracked.add(t_idx)

        for t_idx in currently_tracked:
            if t_idx not in self.found_targets:
                self.found_targets.add(t_idx)
                self._discovery_times[t_idx] = self.timestep
                for a in self.agents:
                    rewards[a] += 100.0   # first-discovery bonus
            else:
                for a in self.agents:
                    rewards[a] += 1.0     # continuous tracking bonus

        # ── 5. Terminal conditions ────────────────────────────────────────────
        env_done      = len(self.found_targets) == self.num_targets
        env_truncated = self.timestep >= self.max_timesteps

        if env_done:
            for a in self.agents:
                rewards[a] += 500.0   # mission success

        for a in self.agents:
            self._episode_rewards[a] += rewards[a]

        terminations = {a: env_done      for a in self.agents}
        truncations  = {a: env_truncated for a in self.agents}

        if env_done or env_truncated:
            self.agents = []

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        infos        = {a: {} for a in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    # ──────────────────────────────────────────────────────────────────────────
    def _has_line_of_sight(self, p1, p2, max_dist):
        """Bresenham line-of-sight check."""
        if np.linalg.norm(p1 - p2) > max_dist:
            return False
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        dx, dy  = abs(x1 - x0), abs(y1 - y0)
        x, y    = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.obstacle_map[x, y] == 1:
                    return False
                err -= dy
                if err < 0:
                    y  += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.obstacle_map[x, y] == 1:
                    return False
                err -= dx
                if err < 0:
                    x  += sx
                    err += dy
                y += sy
        return self.obstacle_map[x, y] == 0

    def _update_coverage(self):
        new_explored = 0
        for pos in self.agent_positions.values():
            x, y = pos
            for cx in range(max(0, x - self.sensor_range), min(self.grid_size, x + self.sensor_range + 1)):
                for cy in range(max(0, y - self.sensor_range), min(self.grid_size, y + self.sensor_range + 1)):
                    if self.coverage_map[cx, cy] == 0:
                        if self._has_line_of_sight(pos, np.array([cx, cy]), self.sensor_range):
                            self.coverage_map[cx, cy] = 1
                            new_explored += 1
        return new_explored

    def _get_obs(self, agent_id):
        pos = self.agent_positions[agent_id]
        ego = pos.astype(np.float32) / self.grid_size

        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        lidar  = np.ones(self.num_rays, dtype=np.float32)
        for i, angle in enumerate(angles):
            for r in range(1, self.sensor_range + 1):
                rx = int(pos[0] + r * np.cos(angle))
                ry = int(pos[1] + r * np.sin(angle))
                if (rx < 0 or rx >= self.grid_size or
                        ry < 0 or ry >= self.grid_size or
                        self.obstacle_map[rx, ry] == 1):
                    lidar[i] = r / self.sensor_range
                    break

        comm_nodes, comm_adj = [], []
        for other_a in self.possible_agents:
            if other_a == agent_id:
                continue
            if other_a in self.agent_positions:
                other_pos = self.agent_positions[other_a]
                dist = np.linalg.norm(pos - other_pos)
                if dist <= self.comm_range:
                    comm_nodes.append((other_pos - pos) / self.grid_size)
                    comm_adj.append(1.0)
                else:
                    comm_nodes.append(np.zeros(2))
                    comm_adj.append(0.0)
            else:
                comm_nodes.append(np.zeros(2))
                comm_adj.append(0.0)

        target_indicators = np.zeros(self.num_targets, dtype=np.float32)
        for t_idx, t_pos in enumerate(self.target_positions):
            if self._has_line_of_sight(pos, t_pos, self.sensor_range):
                target_indicators[t_idx] = 1.0

        return {
            "ego":               ego,
            "lidar":             lidar,
            "comm_nodes":        np.array(comm_nodes, dtype=np.float32),
            "comm_adj":          np.array(comm_adj,   dtype=np.float32),
            "target_indicators": target_indicators,
        }

    def render(self):
        pass

    def get_state_snapshot(self):
        """Return serialisable state for the frontend visualiser."""
        return {
            "timestep":       self.timestep,
            "grid_size":      self.grid_size,
            "agent_positions":{a: self.agent_positions[a].tolist() for a in self.agent_positions},
            "target_positions":[p.tolist() for p in self.target_positions],
            "found_targets":  list(self.found_targets),
            "coverage_pct":   float(self.coverage_map.sum()) / (self.grid_size ** 2) * 100,
            "obstacle_map":   self.obstacle_map.tolist(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Random-policy MARL runner  —  logs every step as newline-delimited JSON
# to stdout so the frontend (or a terminal) can consume it live.
# ════════════════════════════════════════════════════════════════════════════

def run_episode(env, episode_num=1, delay=0.0):
    obs, _ = env.reset(seed=episode_num)
    done    = False
    total_rewards = {a: 0.0 for a in env.possible_agents}

    print(json.dumps({
        "type":    "episode_start",
        "episode": episode_num,
        "agents":  env.possible_agents,
        "config": {
            "grid_size":    env.grid_size,
            "num_agents":   env._num_agents,
            "num_targets":  env.num_targets,
            "sensor_range": env.sensor_range,
            "comm_range":   env.comm_range,
            "max_steps":    env.max_timesteps,
        }
    }), flush=True)

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        for a, r in rewards.items():
            total_rewards[a] += r

        snap = env.get_state_snapshot()

        # Per-step JSON event (consumed by frontend)
        print(json.dumps({
            "type":       "step",
            "episode":    episode_num,
            "timestep":   snap["timestep"],
            "agents":     snap["agent_positions"],
            "targets":    snap["target_positions"],
            "found":      snap["found_targets"],
            "coverage":   round(snap["coverage_pct"], 2),
            "rewards":    {a: round(r, 3) for a, r in rewards.items()},
            "step_reward":round(sum(rewards.values()), 3),
        }), flush=True)

        if delay > 0:
            time.sleep(delay)

    # Episode summary
    outcome = "success" if len(env.found_targets) == env.num_targets else "timeout"
    print(json.dumps({
        "type":            "episode_end",
        "episode":         episode_num,
        "outcome":         outcome,
        "timesteps":       env.timestep,
        "targets_found":   len(env.found_targets),
        "total_targets":   env.num_targets,
        "coverage_pct":    round(float(env.coverage_map.sum()) / (env.grid_size ** 2) * 100, 2),
        "discovery_times": env._discovery_times,
        "total_reward":    round(sum(total_rewards.values()), 2),
        "per_agent_reward":{a: round(r, 2) for a, r in total_rewards.items()},
    }), flush=True)

    return outcome == "success"


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwarmSearch MARL runner")
    parser.add_argument("--episodes",    type=int,   default=5,    help="Number of episodes to run")
    parser.add_argument("--agents",      type=int,   default=4,    help="Number of drones")
    parser.add_argument("--targets",     type=int,   default=3,    help="Number of targets")
    parser.add_argument("--grid",        type=int,   default=20,   help="Grid size NxN")
    parser.add_argument("--max-steps",   type=int,   default=300,  help="Max steps per episode")
    parser.add_argument("--delay",       type=float, default=0.0,  help="Seconds between steps (for live viz)")
    parser.add_argument("--api-test",    action="store_true",       help="Run PettingZoo API test only")
    args = parser.parse_args()

    env = SwarmSearchEnv(
        num_agents=args.agents,
        grid_size=args.grid,
        num_targets=args.targets,
        max_timesteps=args.max_steps,
    )

    if args.api_test:
        from pettingzoo.test import parallel_api_test
        print(json.dumps({"type": "info", "msg": "Running PettingZoo API test..."}), flush=True)
        try:
            parallel_api_test(env, num_cycles=50)
            print(json.dumps({"type": "info", "msg": "API test PASSED"}), flush=True)
        except Exception as e:
            print(json.dumps({"type": "error", "msg": str(e)}), flush=True)
        sys.exit(0)

    successes = 0
    print(json.dumps({
        "type": "run_start",
        "episodes": args.episodes,
        "policy": "random"
    }), flush=True)

    for ep in range(1, args.episodes + 1):
        ok = run_episode(env, episode_num=ep, delay=args.delay)
        if ok:
            successes += 1

    print(json.dumps({
        "type":        "run_end",
        "episodes":    args.episodes,
        "successes":   successes,
        "success_rate":round(successes / args.episodes * 100, 1),
    }), flush=True)
