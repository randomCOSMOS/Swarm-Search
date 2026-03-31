# Swarm Target Search - Dec-POMDP Formulation & Reward Architecture

## 1. Dec-POMDP Mathematical Formulation
The environment is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) defined by the tuple $\langle I, S, \mathbf{A}, T, R, \boldsymbol{\Omega}, O, \gamma \rangle$:

* **$I$ (Agents):** A finite set of swarm agents $I = \{1, 2, ..., N\}$.
* **$S$ (State Space):** The true global state $s \in S$. 
    * $s = (s_{agents}, s_{targets}, s_{env})$
    * $s_{agents}$: Concatenation of all agent dynamics $(s_1, ..., s_N)$ where $s_i = (x_i, y_i, \theta_i, v_i)$.
    * $s_{targets}$: Hidden locations/status of $K$ targets (static or moving).
    * $s_{env}$: Environmental state, including obstacles and global coverage map.
* **$\mathbf{A}$ (Joint Action Space):** $\mathbf{A} = A_1 \times A_2 \times ... \times A_N$.
    * *Discrete setting (Grid):* $A_i \in \{\uparrow, \downarrow, \leftarrow, \rightarrow, \text{Hover}\}$.
    * *Continuous setting (UAVs/UGVs):* $a_i \in \mathbb{R}^2$ e.g., $[v, \omega]$ (linear and angular velocity).
* **$T$ (Transition Function):** $T(s' | s, \mathbf{a})$. Determines system kinematics. Deterministic for ideal movement, stochastic when factoring in environmental noise (e.g., wind drift) or moving targets.
* **$R$ (Reward Function):** The reward function $R(s, \mathbf{a}, s')$. In cooperative MARL, agents share a team reward to optimize collective behavior, combined with localized dense shaping metrics.
* **$\boldsymbol{\Omega}$ (Joint Observation Space):** $\boldsymbol{\Omega} = \Omega_1 \times \Omega_2 \times ... \times \Omega_N$.
* **$O$ (Observation Function):** $O(\mathbf{o} | s', \mathbf{a})$. The partial, local perception of each agent $\mathbf{o}_i \in \Omega_i$:
    * Ego-state (pose, velocity).
    * Local Field of View (FoV) utilizing a 2D ray-casting array (simulating LiDAR) to realistically account for line-of-sight occlusions behind walls and obstacles.
    * Explicit communication topology via a message-passing vector or a Graph Attention Network (GAT) encoding of neighboring agents within communication range $R_{comm}$.
    * Indicators for targets discovered inside sensor range $R_{sensor}$.
* **$\gamma$ (Discount Factor):** $\gamma \in [0, 1)$ implicitly favors rapid search closure over delayed detection.

## 2. Reward Shaping Architecture
Balancing exploration (spreading out) with exploitation (target detection) without centralized coordination requires careful shaping. We structure the reward as $r_i^t = w_{d} r_{dense}^t + w_{s} r_{sparse}^t$.

### Dense Rewards (Exploration & Safety)
1.  **Exploration Bonus (Area Coverage):** * $r_{cov} = +k$ for every *new* global grid cell revealed by the agent's sensor footprint. This implicitly drives the swarm to spread out, as overlapping search areas yield $0$ marginal reward.
2.  **Dispersion & Collision Penalty:** * $r_{col} = -p_{col}$ for colliding with obstacles or boundaries.
    * $r_{disp} = -p_{disp}$ if the distance to the nearest neighbor $d_{ij} < d_{safe}$ (prevents swarm collapse and search redundancy).
3.  **Step Penalty:**
    * $r_{step} = -c$ per timestep to penalize idleness and prioritize minimizing total search time.

### Sparse Rewards (Objective Focus)
1.  **Target Discovery:** * $r_{target} = +100$ awarded to the team the *first* time a target falls within any agent's $R_{sensor}$.
2.  **Target Tracking (Dynamic Bonus):**
    * $r_{track} = +c$ awarded per timestep that a moving or evading target is kept actively within the collective sensor range.
3.  **Mission Success:** * $r_{success} = +500$ team bonus when all $K$ targets are located before the episode horizon terminates.

## 3. Simulation Framework Recommendation
**Recommendation: PettingZoo (Parallel API) + Ray RLlib**

* **Why PettingZoo?** Maintained by the Farama Foundation, it is the modern Gymnasium standard for multi-agent RL. The `ParallelEnv` API processes all agent actions concurrently, mimicking decentralized execution perfectly.
* **Why Ray RLlib or CleanRL?** PettingZoo bridges seamlessly to production-grade algorithms. For MAPPO/MADDPG, RLlib handles distributed training pipelines, experience replay for MARL, and Centralized Training with Decentralized Execution (CTDE) mechanics natively.