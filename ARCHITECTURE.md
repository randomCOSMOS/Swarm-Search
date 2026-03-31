# Swarm Search DEC-POMDP Architecture

## System Overview

This document describes the complete architecture of the Swarm Target Search environment, a decentralized partially-observable Markov decision process (Dec-POMDP) for multi-agent reinforcement learning using swarms of autonomous agents.

---

## Architecture Diagram

```
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                    SWARM SEARCH DEC-POMDP ARCHITECTURE                                ║
╚════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            VISUALIZATION & INTERACTION LAYER                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────────────────────────────┐       ┌──────────────────────────────────┐      │
│  │   swarm_search_ui.html (Browser) │◄──────│ Real-time Simulation Renderer   │      │
│  │  ┌────────────────────────────┐  │       ├──────────────────────────────────┤      │
│  │  │ Canvas  Rendering Engine  │  │       │ • Agent Position Overlay        │      │
│  │  │ • Grid Visualization      │  │       │ • Target Markers                │      │
│  │  │ • Agent/Target Movement   │  │       │ • LiDAR Heatmap                 │      │
│  │  │ • Coverage Heatmap        │  │       │ • Obstacle Map                  │      │
│  │  └────────────────────────────┘  │       └──────────────────────────────────┘      │
│  │                                   │                                                 │
│  └──────────────────────────────────┘                                                 │
│                          ▲                                                              │
│                          │ JavaScript ◄─────┐                                          │
│                          │                    │                                         │
└──────────────────────────┼────────────────────┼──────────────────────────────────────────┘
                           │                    │
┌──────────────────────────┼────────────────────┼──────────────────────────────────────────┐
│                    CORE ENVIRONMENT LAYER (PettingZoo)                                  │
├──────────────────────────┼────────────────────┼──────────────────────────────────────────┤
│                          │                    │                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                     SwarmSearchEnv (ParallelEnv)                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │ STATE MANAGEMENT                                                            │ │ │
│  │  │ ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐ │ │ │
│  │  │ │ Agent Positions  │  │ Target Positions │  │ Coverage Map (Grid x Grid)│ │ │ │
│  │  │ │ (dict: agent→pos)│  │ (list: K targets)│  │ (0/1 Binary State)       │ │ │ │
│  │  │ └──────────────────┘  └──────────────────┘  └────────────────────────────┘ │ │ │
│  │  │ ┌──────────────────────────────────────────────────────────────────────────┐ │ │ │
│  │  │ │ Obstacle Map | Found Targets | Timestep Counter | Communication        │ │ │ │
│  │  │ │              |                |                  | Topology Cache      │ │ │ │
│  │  │ └──────────────────────────────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                   │ │
│  │  ┌──────────────────────────┐         ┌──────────────────────────────────────┐  │ │
│  │  │   STEP PIPELINE          │         │  OBSERVATION GENERATION             │  │ │
│  │  ├──────────────────────────┤         ├──────────────────────────────────────┤  │ │
│  │  │ 1. Kinematics            │         │ For each agent:                      │  │ │
│  │  │    • Boundary checks     │         │  ├─ Ego State (pose/velocity)       │  │ │
│  │  │    • Obstacle collision  │         │  ├─ LiDAR Rays (16-ray casting)    │  │ │
│  │  │    • Position update     │         │  ├─ Comm Neighbors (GAT Graph)     │  │ │
│  │  │                          │         │  ├─ Adjacency Matrix               │  │ │
│  │  │ 2. Target Dynamics       │         │  └─ Target Indicators             │  │ │
│  │  │    • Evasion heuristic   │         │                                     │  │ │
│  │  │    • Random walk         │         │ Returns Dict:                       │  │ │
│  │  │    • Boundary clip       │         │  {ego, lidar, comm_nodes,          │  │ │
│  │  │                          │         │   comm_adj, target_indicators}     │  │ │
│  │  │ 3. Coverage Update       │         │                                     │  │ │
│  │  │    • Ray-casting LOS     │         └──────────────────────────────────────┘  │ │
│  │  │    • Grid cell marking   │                                                   │ │
│  │  │                          │         ┌──────────────────────────────────────┐  │ │
│  │  │ 4. Reward Calculation    │         │  REWARD AGGREGATION                │  │ │
│  │  │    • Dense + Sparse      │         ├──────────────────────────────────────┤  │ │
│  │  │    • Team-wide bonus     │         │ For each agent:                      │  │ │
│  │  │                          │         │  r_total = r_coverage + r_collision  │  │ │
│  │  └──────────────────────────┘         │           + r_tracking + r_success  │  │ │
│  │                                        │                                     │  │ │
│  │                                        │ Coverage: +0.5 per new cell        │  │ │
│  │                                        │ Collision: -0.5                     │  │ │
│  │                                        │ Dispersion: -0.5 if too close      │  │ │
│  │                                        │ Step cost: -0.05                    │  │ │
│  │                                        │ Discovery: +100.0 (first time)      │  │ │
│  │                                        │ Tracking: +1.0 (per timestep)       │  │ │
│  │                                        │ Success: +500.0 (all targets found) │  │ │
│  │                                        └──────────────────────────────────────┘  │ │
│  │                                                                                   │ │
│  └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │ SENSOR & PERCEPTION SYSTEMS                                                    │  │
│  │ ┌───────────────────────────────┐  ┌──────────────────────────────────────────┐ │  │
│  │ │ LiDAR Ray-Casting             │  │ Line-of-Sight Detection                  │ │  │
│  │ │ • 16 rays per agent          │  │ • Bresenham's algorithm                  │ │  │
│  │ │ • Obstacle detection         │  │ • Wall/Obstacle occlusion check          │ │  │
│  │ │ • Continuous range reading   │  │ • Target visibility filtering           │ │  │
│  │ │ • Normalized output [0,1]    │  │ • Communication link validation          │ │  │
│  │ └───────────────────────────────┘  └──────────────────────────────────────────┘ │  │
│  │                                                                                  │  │
│  │ ┌──────────────────────────────────────────────────────────────────────────────┐│  │
│  │ │ Graph Attention Network (GAT) - Communication Topology                      ││  │
│  │ │ • Adjacent agent encoding (relative positions)                              ││  │
│  │ │ • Communication range filtering (R_comm aware)                              ││  │
│  │ │ • Graph adjacency matrix generation                                         ││  │
│  │ └──────────────────────────────────────────────────────────────────────────────┘│  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           ▲
                                           │ Actions: {agent_i: action}
                                           │ shape: (N_agents,) ∈ {0,1,2,3,4}
                                           │
┌──────────────────────────────────────────┴──────────────────────────────────────────────┐
│                     TRAINING & INFERENCE LAYER (Ray RLlib)                              │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ MARL ALGORITHM: MAPPO / MADDPG / QMix                                             │ │
│  │                                                                                     │ │
│  │  Centralized Training ◄────────┐                                                   │ │
│  │  Decentralized Execution (CTDE)│                                                   │ │
│  │                                 │                                                   │ │
│  │  ┌──────────────────────────────┴─────────────────────────────────────────┐       │ │
│  │  │ Multi-Agent Policy Network (PPO)                                       │       │ │
│  │  │ ┌──────────────────┐  ┌────────────────────────────────────────────┐  │       │ │
│  │  │ │ Local Critic     │◄─│ Shared Experience Buffer (Replay Memory)  │  │       │ │
│  │  │ │ (per-agent)      │  │ • Batch size: N_agents × rollout_len      │  │       │ │
│  │  │ └──────────────────┘  └────────────────────────────────────────────┘  │       │ │
│  │  │                                                                         │       │ │
│  │  │ ┌──────────────────────────────────────────────────────────────────┐  │       │ │
│  │  │ │ GNN Actor Network (GAT-enhanced)                                │  │       │ │
│  │  │ │ • Graph convolutions over communication topology               │  │       │ │
│  │  │ │ • Local policy π(a|o_i) with attention heads                 │  │       │ │
│  │  │ │ • Action logits + value estimate                              │  │       │ │
│  │  │ └──────────────────────────────────────────────────────────────────┘  │       │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘       │ │
│  │                                                                                     │ │
│  │  Training Loop:                                                                    │ │
│  │  ┌─────────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │ │
│  │  │ 1. Collect Rollouts     │──│ 2. Compute GAE   │──│ 3. PPO Update         │    │ │
│  │  │    (N_workers parallel) │  │    (per-agent)   │  │ (5 epochs, clip=0.2) │    │ │
│  │  └─────────────────────────┘  └──────────────────┘  └────────────────────────┘    │ │
│  │                                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ Distributed Training Orchestration                                         │  │ │
│  │  │ Nodes: [Master] ◄───► [Worker1] ─┬─ [Worker2] ─┬─ [Worker3] ─┬─ ...      │  │ │
│  │  │ Experiences aggregated ◄──┘      │              │              │          │  │ │
│  │  │                                  ▼              ▼              ▼          │  │ │
│  │  │                         [Gradient Computation & Parameter Sync]           │  │ │
│  │  └─────────────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                                     │ │
│  │  Checkpoints: ./logs/checkpoints/ ← Model snapshots every N episodes              │ │
│  │                                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ INFERENCE PIPELINE (Deployment)                                                    │ │
│  │  [Trained Model] ──► [Gymnasium Wrapper] ──► [Agent Actions] ──► [Env Step]       │ │
│  │                      (Deterministic Policy)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW & DEPENDENCIES                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  numpy ────────────► State arrays (positions, maps)                                    │
│  gymnasium ────────► Space definitions (Box, Discrete, Dict)                           │
│  pettingzoo ───────► ParallelEnv API integration                                       │
│  torch ────────────► Neural network compute (CPU/CUDA)                                 │
│  torch_geometric ──► Graph convolution operations (GAT)                                │
│  ray[rllib] ───────► Distributed RL training, checkpointing, logging                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════════════════╗
║ KEY METRICS TRACKED                                                                    ║
╠════════════════════════════════════════════════════════════════════════════════════════╣
║ • Episode Return (total reward)          • Coverage Ratio (% grid explored)            ║
║ • Target Discovery Rate (targets/episode)• Convergence Speed (episodes to 100% cov)    ║
║ • Policy Loss (PPO objective)            • Value Function Loss                         ║
║ • Agent-Agent Collision Rate             • Communication Efficiency (links active)     ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Breakdown

### 1. Visualization & Interaction Layer
- **swarm_search_ui.html**: Browser-based canvas renderer with real-time visualization
- Displays agent positions, targets, obstacles, LiDAR heatmaps, and coverage maps
- Fully self-contained with hardcoded simulation logic

### 2. Core Environment Layer (PettingZoo)
- **SwarmSearchEnv**: ParallelEnv implementation for multi-agent coordination
- **State Management**:
  - Agent positions (dict: agent_id → [x, y])
  - Target positions (list of [x, y] coords)
  - Coverage map (binary grid tracking explored cells)
  - Obstacle map (static walls)
  - Found targets set (track discovery)
  - Timestep counter

### 3. Step Pipeline
1. **Kinematics**: Agent movement with boundary/obstacle checks
2. **Target Dynamics**: Evasion heuristics with random walk
3. **Coverage Update**: Incremental line-of-sight grid expansion
4. **Reward Calculation**: Dense + sparse reward aggregation

### 4. Sensor & Perception Systems
- **LiDAR Ray-Casting**: 16 rays per agent, obstacle detection, normalized [0,1] output
- **Line-of-Sight (LoS)**: Bresenham's algorithm for target/wall visibility
- **Graph Attention Network (GAT)**: Communication topology with adjacency matrix encoding

### 5. Observation Generation
Each agent receives a dictionary with:
- `ego`: Normalized position [x/grid_size, y/grid_size]
- `lidar`: 16-ray obstacle distances (normalized)
- `comm_nodes`: Relative positions of neighbors within comm_range
- `comm_adj`: Adjacency indicators (1.0 if in range, 0.0 else)
- `target_indicators`: Binary visibility flags for each target

### 6. Reward System

| Reward Type | Value | Trigger |
|---|---|---|
| Coverage Bonus | +0.5 | Per new cell explored |
| Collision Penalty | -0.5 | Agent collision or wall hit |
| Dispersion Penalty | -0.5 | Agents too close (<2.0 units) |
| Step Cost | -0.05 | Every timestep (idleness penalty) |
| Discovery Bonus | +100.0 | First detection of a target |
| Tracking Bonus | +1.0 | Continuous target visibility |
| Mission Success | +500.0 | All targets found |

### 7. Multi-Agent RL Training Layer (Ray RLlib)
- **Algorithm**: MAPPO (Multi-Agent Proximal Policy Optimization)
- **Training Strategy**: Centralized Training, Decentralized Execution (CTDE)
- **Network Architecture**: GAT-enhanced actor networks with per-agent critics
- **Distributed Training**: Multi-worker experience collection with parameter synchronization
- **Checkpointing**: Model snapshots saved to `./logs/checkpoints/` every N episodes

### 8. Action Space
| Action | Meaning |
|---|---|
| 0 | Move Up (y += 1) |
| 1 | Move Down (y -= 1) |
| 2 | Move Left (x -= 1) |
| 3 | Move Right (x += 1) |
| 4 | Hover (no movement) |

---

## Data Flow

**Input (Actions)**: `{agent_id: action ∈ {0,1,2,3,4}}`

**Processing**:
1. Update agent kinematics with boundary/obstacle checks
2. Update target positions with evasion logic
3. Mark newly covered cells via LoS raycasting
4. Check target discovery conditions
5. Compute unified rewards across all agents

**Output (Observations, Rewards, Dones)**:
- `observations`: Dict[agent_id → Dict[str → np.ndarray]]
- `rewards`: Dict[agent_id → float]
- `terminations`: Dict[agent_id → bool] (mission complete)
- `truncations`: Dict[agent_id → bool] (max timesteps reached)
- `infos`: Dict[agent_id → {}]

---

## Key Metrics

| Metric | Description |
|---|---|
| Episode Return | Total accumulated reward per episode |
| Coverage Ratio | Percentage of grid cells explored |
| Target Discovery Rate | Number of targets found per episode |
| Policy Loss | PPO objective convergence metric |
| Value Loss | Critic prediction error |
| Collision Rate | Agent-agent and agent-wall collisions |
| Communication Efficiency | Percentage of active communication links |
| Convergence Speed | Episodes needed to reach target performance levels |

---

## Dependencies

- **numpy**: State and array management
- **gymnasium**: Space definitions (Box, Discrete, Dict)
- **pettingzoo**: ParallelEnv API and multi-agent environment spec
- **torch**: Neural network computation (CPU/CUDA)
- **torch_geometric**: Graph neural network operations (GAT)
- **ray[rllib]**: Distributed reinforcement learning, checkpointing, tensorboard logging

---

## Future Extensions

- [ ] Heterogeneous agent types (different sensor/action capabilities)
- [ ] Dynamic obstacle generation and moving walls
- [ ] Multi-target assignment with coordination constraints
- [ ] Communication bandwidth limitations
- [ ] Energy constraints and recharging stations
- [ ] Partially adversarial target behavior with deception
- [ ] Curriculum learning with difficulty scaling
