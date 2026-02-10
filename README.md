# Hospital Staffing Optimization

A discrete-time (hourly) hospital staffing decision system with two modes: a **greedy heuristic** (stochastic simulation) and a **MIP optimizer** (deterministic look-ahead). Both modes minimize total cost while prioritizing patient survival. Only ambulance patients can be diverted. Only treated patients (those assigned a nurse) are eligible for discharge. The budget is a hard cap on controllable spending (nurses + diversions); if starting nurse payroll exceeds the budget, the run is infeasible. The death penalty is an uncontrollable cost used to ensure the optimizer avoids deaths.

## Two Modes

### Greedy (`--mode greedy`, default)

Hour-by-hour heuristic. Departures and deaths are stochastic (random). Each hour the greedy algorithm follows this priority:

1. Pay existing nurses (mandatory — cannot fire mid-shift)
2. Hire new nurses for uncovered patients (if budget allows)
3. Divert uncovered ambulance patients (if budget allows)
4. Remaining uncovered patients wait without a nurse → 10% random death chance per hour

### Optimize (`--mode optimize`)

Multi-period Mixed Integer Program (MIP) solved with PuLP/CBC. Looks ahead across all hours simultaneously and uses expected values for stochastic elements (mean departures, 10% expected death rate). Finds the globally optimal staffing plan.

## Optimization Model Formulation

### Sets

- $t \in \{0, 1, \ldots, T-1\}$ — hourly time periods

### Parameters

| Symbol | Description |
|--------|-------------|
| $c_n$ | Cost per nurse per hour (default \$150) |
| $c_d$ | Cost per diverted patient (default \$1,000) |
| $c_\delta$ | Penalty per patient death (default \$10,000,000) |
| $a_t$ | Total patient arrivals at hour $t$ (from `arrivals.csv`) |
| $\bar{a}_t$ | Ambulance arrivals at hour $t$ (from `arrivals.csv`, only these can be diverted) |
| $\bar{d}$ | Expected departures per hour (mean of `departures.csv`) |
| $R$ | Maximum room capacity (default 88) |
| $B$ | Hourly budget cap on controllable spending (optional) |
| $P_0$ | Starting occupancy — patients already in hospital (default 55) |
| $N_0$ | Initial nurses (default 55) |

### Decision Variables

| Variable | Type | Description |
|----------|------|-------------|
| $h_t$ | Integer $\geq 0$ | Nurses to hire at hour $t$ |
| $v_t$ | Integer $\geq 0$ | Ambulance patients to divert at hour $t$ |

### State Variables

| Variable | Description |
|----------|-------------|
| $n_t$ | Total nurses on staff at hour $t$ |
| $\text{dep}_t$ | Actual departures at hour $t$ |
| $x_t$ | Active patients at hour $t$ (before deaths) |
| $w_t$ | Waiting patients without a nurse at hour $t$ |
| $p_t$ | Patients at end of hour $t$ (after deaths) |

### Objective

$$\min \sum_{t=0}^{T-1} \Big( c_n \cdot n_t + c_d \cdot v_t + c_\delta \cdot 0.1 \cdot w_t \Big)$$

The death penalty term $c_\delta \cdot 0.1 \cdot w_t$ represents the expected cost of deaths (10% of waiting patients die each hour). This large penalty ensures the optimizer avoids letting patients wait without nurses.

### Constraints

For each hour $t = 0, \ldots, T-1$:

**Diversion limit** — only ambulance patients can be diverted:

$$v_t \leq \bar{a}_t$$

**Departures** — cannot exceed expected rate, and only treated patients (those with a nurse last hour) can depart:

$$\text{dep}_t \leq \bar{d}, \qquad \text{dep}_t \leq p_{t-1} - 0.9 \cdot w_{t-1}$$

**Patient flow** — active patients after arrivals, departures, and diversions:

$$x_t = p_{t-1} - \text{dep}_t + a_t - v_t$$

**Room capacity**:

$$x_t \leq R$$

**Nurse staffing** — can only hire, never fire:

$$n_t = n_{t-1} + h_t$$

**Waiting patients** — those without a nurse (max constraint):

$$w_t = \max(0, x_t - n_t)$$

**End-of-hour patients** — 10% of waiting patients die:

$$p_t = x_t - 0.1 \cdot w_t$$

**Budget constraint** (when budget $B$ is specified) — hard cap on controllable costs:

$$c_n \cdot n_t + c_d \cdot v_t \leq B$$

**Non-negativity and integrality**:

$$h_t, v_t, n_t \in \mathbb{Z}_{\geq 0}, \qquad p_t, x_t, w_t, \text{dep}_t \geq 0$$

**Initial conditions**: $p_{-1} = P_0$, $n_{-1} = N_0$, $w_{-1} = \max(0, P_0 - N_0)$

## Data

- **`arrivals.csv`** — 24 rows (hours 0–23), columns: Time, Emergency Room Ambulance, Emergency Room Walk in, Surgery, Critical Care, Step Down
- **`departures.csv`** — single column "Leaving" with historical departure counts (randomly sampled in greedy mode, averaged in optimize mode)

## Usage

```bash
# Greedy mode (default), unlimited budget
python hospital_simulation.py

# Greedy mode with budget
python hospital_simulation.py --mode greedy --budget 15000

# Optimization mode with budget
python hospital_simulation.py --mode optimize --budget 15000

# Custom costs
python hospital_simulation.py --mode optimize --budget 12000 --nurse-cost 200 --diversion-cost 1500

# All options
python hospital_simulation.py --help
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | greedy | `greedy` (stochastic heuristic) or `optimize` (MIP look-ahead) |
| `--budget` | unlimited | Hard cap on hourly controllable spending (nurses + diversions) |
| `--nurse-cost` | 150 | Cost per nurse per hour ($c_n$, all nurses, not just new hires) |
| `--diversion-cost` | 1000 | Cost per diverted ambulance patient ($c_d$) |
| `--death-cost` | 10000000 | Penalty per patient death ($c_\delta$, not counted against budget) |
| `--starting-occupancy` | 55 | Starting occupancy — patients already in hospital ($P_0$) |
| `--initial-nurses` | 55 | Starting nurse count ($N_0$) |
| `--max-rooms` | 88 | Hospital room capacity ($R$) |
| `--hours` | 24 | Number of hours to run ($T$) |

## Dependencies

- Python 3.10+
- pandas, numpy, pulp
