import numpy as np
import functools
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

    def __init__(self, num_agents=3, grid_size=20, num_targets=2, sensor_range=5, comm_range=8, max_timesteps=200):
        super().__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.max_timesteps = max_timesteps
        self.num_rays = 16 # For LiDAR
        
        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        
        # State tracking
        self.agent_positions = {}
        self.target_positions = []
        self.coverage_map = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.obstacle_map = self._generate_obstacles()
        self.found_targets = set()
        self.timestep = 0

    def _generate_obstacles(self):
        # Generate some static obstacles to test LiDAR line-of-sight
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        obs[8:12, 8] = 1 # vertical wall
        obs[8, 8:12] = 1 # horizontal wall
        return obs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Complex observation Dictionary designed for GAT and LiDAR processing
        return Dict({
            "ego": Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32), 
            "lidar": Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32),
            "comm_nodes": Box(low=-1.0, high=1.0, shape=(self.num_agents - 1, 2), dtype=np.float32),
            "comm_adj": Box(low=0.0, high=1.0, shape=(self.num_agents - 1,), dtype=np.float32),
            "target_indicators": Box(low=0.0, high=1.0, shape=(self.num_targets,), dtype=np.float32)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Hover
        return Discrete(5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        # Random cluster spawn
        self.agent_positions = {
            a: np.array([np.random.randint(0, 3), np.random.randint(0, 3)]) 
            for a in self.agents
        }
        
        # Targets start far away from agents
        self.target_positions = [
            np.array([np.random.randint(self.grid_size - 5, self.grid_size), 
                      np.random.randint(self.grid_size - 5, self.grid_size)])
            for _ in range(self.num_targets)
        ]
        
        self.coverage_map.fill(0)
        self.found_targets = set()
        
        self._update_coverage()

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.timestep += 1
        rewards = {a: 0.0 for a in self.agents}
        
        # ——— 1. Agent Actions & Kinematics ———
        for agent_id, action in actions.items():
            pos = self.agent_positions[agent_id]
            new_pos = np.copy(pos)
            if action == 0: new_pos[1] += 1
            elif action == 1: new_pos[1] -= 1
            elif action == 2: new_pos[0] -= 1
            elif action == 3: new_pos[0] += 1
            
            # Boundary and obstacle check
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and 
                self.obstacle_map[new_pos[0], new_pos[1]] == 0):
                self.agent_positions[agent_id] = new_pos
            else:
                rewards[agent_id] -= 0.5 # Collision penalty

            rewards[agent_id] -= 0.05 # Idleness / Step penalty

        # ——— 2. Dynamic Evading Targets ———
        for idx in range(self.num_targets):
            t_pos = self.target_positions[idx]
            closest_dist = float('inf')
            for a in self.agents:
                d = np.linalg.norm(self.agent_positions[a] - t_pos)
                if d < closest_dist:
                    closest_dist = d
            
            move_dir = np.array([0, 0])
            if closest_dist < self.sensor_range + 2:
                # Evasion heuristic: random drift favoring open space away from agent
                move_dir = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
            else:
                if np.random.rand() < 0.1: # Slow random walk
                    move_dir = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
                    
            new_t_pos = t_pos + move_dir
            new_t_pos = np.clip(new_t_pos, 0, self.grid_size - 1)
            # Prevent walking into walls
            if self.obstacle_map[new_t_pos[0], new_t_pos[1]] == 0:
                self.target_positions[idx] = new_t_pos

        # ——— 3. Dense Rewards (Coverage & Dispersion) ———
        new_cells = self._update_coverage()
        coverage_reward = new_cells * 0.5
        
        for a in self.agents:
            rewards[a] += coverage_reward
            for other_a in self.agents:
                if a != other_a:
                    dist = np.linalg.norm(self.agent_positions[a] - self.agent_positions[other_a])
                    if dist < 2.0: # Dispersion buffer
                        rewards[a] -= 0.5 

        # ——— 4. Sparse Rewards (Discovery & Continuous Tracking) ———
        currently_tracked = set()
        for a in self.agents:
            for t_idx, t_pos in enumerate(self.target_positions):
                if self._has_line_of_sight(self.agent_positions[a], t_pos, self.sensor_range):
                    currently_tracked.add(t_idx)

        for t_idx in currently_tracked:
            if t_idx not in self.found_targets:
                self.found_targets.add(t_idx)
                for reward_a in self.agents:
                    rewards[reward_a] += 100.0 # First Discovery Bonus
            else:
                for reward_a in self.agents:
                    rewards[reward_a] += 1.0 # Dynamic Tracking Bonus (r_track)

        # ——— 5. Terminations & Truncations ———
        env_done = len(self.found_targets) == self.num_targets
        env_truncated = self.timestep >= self.max_timesteps
        
        if env_done:
            for a in self.agents:
                rewards[a] += 500.0 # Mission Success

        terminations = {a: env_done for a in self.agents}
        truncations = {a: env_truncated for a in self.agents}
        
        if env_done or env_truncated:
            self.agents = []

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def _has_line_of_sight(self, p1, p2, max_dist):
        """Line-of-sight checks using Bresenham's line algorithm."""
        dist = np.linalg.norm(p1 - p2)
        if dist > max_dist: return False
        
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.obstacle_map[x, y] == 1: return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.obstacle_map[x, y] == 1: return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return self.obstacle_map[x, y] == 0

    def _update_coverage(self):
        new_explored = 0
        for pos in self.agent_positions.values():
            x, y = pos[0], pos[1]
            x_min, x_max = max(0, x - self.sensor_range), min(self.grid_size, x + self.sensor_range + 1)
            y_min, y_max = max(0, y - self.sensor_range), min(self.grid_size, y + self.sensor_range + 1)
            
            for cx in range(x_min, x_max):
                for cy in range(y_min, y_max):
                    if self.coverage_map[cx, cy] == 0:
                        if self._has_line_of_sight(pos, np.array([cx, cy]), self.sensor_range):
                            self.coverage_map[cx, cy] = 1
                            new_explored += 1
        return new_explored

    def _get_obs(self, agent_id):
        pos = self.agent_positions[agent_id]
        ego = pos.astype(np.float32) / self.grid_size

        # 16 continuous normalized rays for LiDAR processing
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        lidar = np.ones(self.num_rays, dtype=np.float32)
        for i, angle in enumerate(angles):
            for r in range(1, self.sensor_range + 1):
                rx = int(pos[0] + r * np.cos(angle))
                ry = int(pos[1] + r * np.sin(angle))
                if rx < 0 or rx >= self.grid_size or ry < 0 or ry >= self.grid_size or self.obstacle_map[rx, ry] == 1:
                    lidar[i] = r / self.sensor_range
                    break

        # GAT Communication Network Nodes and Adjacency structure
        comm_nodes = []
        comm_adj = []
        for other_a in self.possible_agents:
            if other_a != agent_id:
                if other_a in self.agent_positions: # Alive
                    other_pos = self.agent_positions[other_a]
                    dist = np.linalg.norm(pos - other_pos)
                    if dist <= self.comm_range:
                        rel = (other_pos - pos) / self.grid_size
                        comm_nodes.append(rel)
                        comm_adj.append(1.0)
                    else:
                        comm_nodes.append(np.zeros(2))
                        comm_adj.append(0.0)
                else: # Dead/Done
                    comm_nodes.append(np.zeros(2))
                    comm_adj.append(0.0)

        comm_nodes = np.array(comm_nodes, dtype=np.float32)
        comm_adj = np.array(comm_adj, dtype=np.float32)

        # Target Indicators
        target_indicators = np.zeros(self.num_targets, dtype=np.float32)
        for t_idx, t_pos in enumerate(self.target_positions):
            if self._has_line_of_sight(pos, t_pos, self.sensor_range):
                target_indicators[t_idx] = 1.0

        return {
            "ego": ego,
            "lidar": lidar,
            "comm_nodes": comm_nodes,
            "comm_adj": comm_adj,
            "target_indicators": target_indicators
        }

    def render(self):
        pass

if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test
    
    env = SwarmSearchEnv()
    try:
        parallel_api_test(env, num_cycles=100)
        print("Advanced Scenario PettingZoo API Test Passed!")
    except Exception as e:
        print(f"API Check issue: {e}")
