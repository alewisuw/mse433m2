# Hospital Staffing Optimization

A discrete-time simulation that makes hourly staffing decisions (hire nurses, divert patients) to minimize costs subject to a hard budget constraint. Patients without nurses wait in a queue and face a 10% hourly mortality risk.

## Model

- **Inputs per hour**: arrivals (from CSV), random departures (from CSV), current state
- **Decisions**: hire nurses, divert patients — both limited by budget
- **Constraint**: budget is a hard cap on controllable costs (nurses + diversions)
- **If budget is exhausted**: patients wait without a nurse → 10% death chance/hour
- **Room cap**: 88 rooms (configurable); overflow patients are diverted automatically

### Cost Function

```
Total Cost = nurse_cost × all_nurses + diversion_cost × diverted + death_cost × deaths
```

## Usage

```bash
# Default (55 patients, 55 nurses, unlimited budget)
python hospital_simulation.py

# With budget constraint
python hospital_simulation.py --budget 10000

# Custom costs
python hospital_simulation.py --budget 12000 --nurse-cost 200 --diversion-cost 1500 --death-cost 5000000

# All options
python hospital_simulation.py --help
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--budget` | unlimited | Hard cap on hourly spending (nurses + diversions) |
| `--nurse-cost` | 150 | Cost per nurse per hour (all nurses, not just new) |
| `--diversion-cost` | 1000 | Cost per diverted patient |
| `--death-cost` | 10000000 | Penalty per patient death |
| `--initial-patients` | 55 | Starting patient count |
| `--initial-nurses` | 55 | Starting nurse count |
| `--max-rooms` | 88 | Hospital room capacity |
| `--hours` | 24 | Simulation duration |
