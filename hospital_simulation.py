"""
Real-Time Hospital Staffing Decision System

Problem:
- Discrete time: 1-hour intervals
- 1:1 nurse-to-patient ratio required
- Nurses are interchangeable
- Patients without a nurse wait and face a 10% death risk per hour
- Only treated patients (those assigned a nurse) are eligible for discharge

Costs:
- Nurse hiring: $150 per nurse per hour
- Patient diversion: $1,000 per patient
- Patient death: $10,000,000 per patient (10% of waiting patients die each hour)

Objective:
Minimize total cost while prioritizing patient survival through staffing and diversion decisions.
Nurses >= patients is NOT a hard constraint; uncovered patients wait and risk death.
"""

import pandas as pd
import random
import numpy as np
import argparse
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus


def optimal_staffing_decision(current_patients, arrivals, departures, current_nurses, 
                             ambulance_arrivals=0, max_rooms=88, 
                             nurse_cost=150, diversion_cost=1000, death_cost=10000000,
                             budget=None):
    """
    Determine optimal staffing decisions for one hour.
    
    Budget is a HARD CAP on total controllable spending (nurses + diversions).
    If the budget is exhausted, remaining uncovered patients wait without a nurse
    and face a 10% chance of dying each hour.
    
    Only ambulance patients can be diverted.
    
    Priority order within budget:
      1. Pay existing nurses (mandatory, cannot fire mid-shift)
      2. Hire additional nurses for uncovered patients
      3. Divert ambulance patients we can't cover
      4. If budget exhausted: patients wait (risk death)
    
    Args:
        current_patients: patients currently in hospital (including any waiting from last hour)
        arrivals: new patient arrivals this hour (all types)
        departures: patients leaving this hour
        current_nurses: nurses currently staffed
        ambulance_arrivals: ambulance arrivals this hour (only these can be diverted)
        max_rooms: maximum room capacity (default 88)
        nurse_cost: cost per nurse per hour (default 150)
        diversion_cost: cost per diverted patient (default 1000)
        death_cost: cost per patient death (default 10000000)
        budget: hard cap on controllable spending per hour (None = unlimited)
        
    Returns:
        dict with keys: hire_nurses, divert_patients, waiting, actual_deaths,
                        active_patients, total_cost
    """
    
    # Step 1: Calculate patients after departures
    patients_after_departures = max(0, current_patients - departures)
    potential_patients = patients_after_departures + arrivals
    
    # Step 2: Room capacity constraint — must divert if over room limit (only ambulances)
    if potential_patients > max_rooms:
        room_overflow_divert = min(potential_patients - max_rooms, ambulance_arrivals)
        potential_patients -= room_overflow_divert
    else:
        room_overflow_divert = 0
    
    # Track remaining divertible ambulances after room overflow
    remaining_ambulances = ambulance_arrivals - room_overflow_divert
    
    # Step 3: Determine nurse gap
    nurses_needed = max(0, potential_patients - current_nurses)
    
    # Step 4: Allocate budget
    # Mandatory cost: pay all existing nurses
    base_nurse_cost = current_nurses * nurse_cost
    
    if budget is not None:
        remaining_budget = budget - base_nurse_cost
        
        if remaining_budget < 0:
            # Infeasible: can't afford existing nurses
            return {
                'status': 'INFEASIBLE',
                'reason': 'Starting nurse payroll exceeds budget',
                'hire_nurses': 0,
                'divert_patients': 0,
                'waiting': None,
                'actual_deaths': None,
                'active_patients': None,
                'total_cost': None
            }
        
        # 4a: Hire as many nurses as budget allows
        if nurses_needed > 0 and remaining_budget > 0:
            affordable_hires = min(nurses_needed, int(remaining_budget // nurse_cost))
            hire_nurses = affordable_hires
            remaining_budget -= hire_nurses * nurse_cost
        else:
            hire_nurses = 0
        
        still_uncovered = nurses_needed - hire_nurses
        
        # 4b: Divert as many uncovered ambulance patients as budget allows
        if still_uncovered > 0 and remaining_budget > 0 and remaining_ambulances > 0:
            affordable_diversions = min(still_uncovered, remaining_ambulances, int(remaining_budget // diversion_cost))
            budget_divert = affordable_diversions
            remaining_budget -= budget_divert * diversion_cost
        else:
            budget_divert = 0
        
        still_uncovered -= budget_divert
        
        # 4c: Remaining uncovered patients wait without a nurse
        waiting = still_uncovered
        
    else:
        # No budget constraint — hire all needed nurses, no waiting
        hire_nurses = nurses_needed
        budget_divert = 0
        waiting = 0
    
    # Total diversions = room overflow + budget-driven
    divert_patients = room_overflow_divert + budget_divert
    
    # Active patients in hospital (includes waiting patients — they occupy rooms)
    active_patients = potential_patients - budget_divert
    
    # Step 5: Deaths — only waiting patients (no nurse) face 10% death risk per hour
    actual_deaths = sum(1 for _ in range(waiting) if random.random() < 0.1)
    
    # Remove dead patients from active count and waiting
    active_patients -= actual_deaths
    waiting -= actual_deaths
    
    # Step 6: Calculate total cost
    total_nurses = current_nurses + hire_nurses
    total_nurse_cost = nurse_cost * total_nurses
    total_diversion_cost = diversion_cost * divert_patients
    total_death_cost = death_cost * actual_deaths
    total_cost = total_nurse_cost + total_diversion_cost + total_death_cost
    
    return {
        'status': 'OK',
        'hire_nurses': hire_nurses,
        'divert_patients': divert_patients,
        'waiting': waiting,
        'actual_deaths': actual_deaths,
        'active_patients': active_patients,
        'total_cost': total_cost
    }


def optimize_staffing(hours, initial_occupancy, initial_nurses, arrivals_per_hour,
                      ambulance_per_hour, expected_departures, max_rooms, nurse_cost,
                      diversion_cost, death_cost, budget=None):
    """
    Solve a multi-period Mixed Integer Program (MIP) for globally optimal staffing.
    
    Uses expected values for stochastic elements:
      - Departures: mean of departure options (deterministic)
      - Deaths: 10% of waiting patients (expected value, not random)
    
    Decision variables per hour:
      - hire[t]: nurses to hire (integer >= 0)
      - divert[t]: ambulance patients to divert (integer >= 0, capped at ambulance arrivals)
    
    State variables per hour:
      - nurses[t]: total nurses on staff
      - patients[t]: patients at end of hour (after deaths)
      - active[t]: patients before deaths
      - waiting[t]: patients without a nurse (face 10% death rate)
      - dep[t]: actual departures (min of expected departures, available patients)
    
    Objective: minimize total cost = Σ (nurse_cost × nurses + diversion_cost × divert
                                        + death_cost × 0.1 × waiting)
    
    Returns:
        dict with 'status', 'total_cost', 'hourly_plan' (list of dicts per hour),
        and 'summary' stats
    """
    T = hours
    d_exp = expected_departures
    
    prob = LpProblem("HospitalStaffing", LpMinimize)
    
    # Decision variables
    h = [LpVariable(f"hire_{t}", lowBound=0, cat='Integer') for t in range(T)]
    div = [LpVariable(f"divert_{t}", lowBound=0, cat='Integer') for t in range(T)]
    
    # State variables
    n = [LpVariable(f"nurses_{t}", lowBound=0, cat='Integer') for t in range(T)]
    p = [LpVariable(f"patients_{t}", lowBound=0) for t in range(T)]
    w = [LpVariable(f"waiting_{t}", lowBound=0) for t in range(T)]
    active = [LpVariable(f"active_{t}", lowBound=0) for t in range(T)]
    dep = [LpVariable(f"dep_{t}", lowBound=0) for t in range(T)]
    no_wait = [LpVariable(f"no_wait_{t}", cat='Binary') for t in range(T)]
    
    for t in range(T):
        p_prev = initial_occupancy if t == 0 else p[t - 1]
        n_prev = initial_nurses if t == 0 else n[t - 1]
        w_prev = max(0, initial_occupancy - initial_nurses) if t == 0 else w[t - 1]
        a_t = arrivals_per_hour[t] if t < len(arrivals_per_hour) else 0
        amb_t = ambulance_per_hour[t] if t < len(ambulance_per_hour) else 0
        
        # Diversions: can only divert ambulance patients
        prob += div[t] <= amb_t
        
        # Departures: can't exceed expected rate; only treated patients can depart
        prob += dep[t] <= d_exp
        prob += dep[t] <= p_prev - 0.9 * w_prev
        
        # Active patients = previous - departures + arrivals - diversions
        prob += active[t] == p_prev - dep[t] + a_t - div[t]
        
        # Room capacity
        prob += active[t] <= max_rooms
        
        # Nurse staffing (can only hire, never fire)
        prob += n[t] == n_prev + h[t]
        
        # Waiting = max(0, active - nurses)
        prob += w[t] >= active[t] - n[t]
        prob += w[t] <= active[t] - n[t] + max_rooms * no_wait[t]
        prob += w[t] <= max_rooms * (1 - no_wait[t])
        
        # End-of-hour patients: deaths remove 10% of waiting
        prob += p[t] == active[t] - 0.1 * w[t]
        
        # Budget: hard cap on controllable spending (nurses + diversions)
        if budget is not None:
            prob += nurse_cost * n[t] + diversion_cost * div[t] <= budget
    
    # Objective: minimize total cost across all hours
    prob += lpSum(
        nurse_cost * n[t] + diversion_cost * div[t] + death_cost * 0.1 * w[t]
        for t in range(T)
    )
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))
    
    status = LpStatus[prob.status]
    if status != 'Optimal':
        return {'status': status, 'total_cost': None, 'hourly_plan': [], 'summary': {}}
    
    # Extract results
    hourly_plan = []
    cumulative_cost = 0
    for t in range(T):
        a_t = arrivals_per_hour[t] if t < len(arrivals_per_hour) else 0
        dep_val = dep[t].varValue
        active_val = active[t].varValue
        n_val = n[t].varValue
        h_val = h[t].varValue
        div_val = div[t].varValue
        w_val = w[t].varValue
        p_val = p[t].varValue
        expected_deaths = 0.1 * w_val
        
        hourly_cost = nurse_cost * n_val + diversion_cost * div_val + death_cost * expected_deaths
        cumulative_cost += hourly_cost
        
        hourly_plan.append({
            'hour': t,
            'arrivals': a_t,
            'departures': round(dep_val, 1),
            'current_patients_end': round(p_val, 1),
            'current_nurses': int(round(n_val)),
            'hire_nurses': int(round(h_val)),
            'divert_patients': int(round(div_val)),
            'waiting_queue': round(w_val, 1),
            'expected_deaths': round(expected_deaths, 2),
            'hourly_cost': hourly_cost,
            'cumulative_cost': cumulative_cost
        })
    
    summary = {
        'total_arrivals': sum(r['arrivals'] for r in hourly_plan),
        'total_departures': sum(r['departures'] for r in hourly_plan),
        'total_hired': sum(r['hire_nurses'] for r in hourly_plan),
        'total_diverted': sum(r['divert_patients'] for r in hourly_plan),
        'total_expected_deaths': sum(r['expected_deaths'] for r in hourly_plan),
        'peak_waiting': max(r['waiting_queue'] for r in hourly_plan),
        'avg_census': np.mean([r['current_patients_end'] for r in hourly_plan]),
        'peak_census': max(r['current_patients_end'] for r in hourly_plan),
        'avg_nurses': np.mean([r['current_nurses'] for r in hourly_plan]),
        'peak_nurses': max(r['current_nurses'] for r in hourly_plan),
        'final_patients': hourly_plan[-1]['current_patients_end'],
        'final_nurses': hourly_plan[-1]['current_nurses'],
        'final_waiting': hourly_plan[-1]['waiting_queue'],
        'total_cost': cumulative_cost,
    }
    
    return {'status': status, 'total_cost': cumulative_cost, 'hourly_plan': hourly_plan, 'summary': summary}


class HospitalSimulation:
    """Simulation of hospital operations over multiple hours"""
    
    def __init__(self, initial_occupancy=55, initial_nurses=55, budget=None, max_rooms=88,
                 nurse_cost=150, diversion_cost=1000, death_cost=10000000):
        self.current_patients = initial_occupancy
        self.current_nurses = initial_nurses
        self.budget = budget
        self.max_rooms = max_rooms
        self.nurse_cost = nurse_cost
        self.diversion_cost = diversion_cost
        self.death_cost = death_cost
        self.waiting_queue = 0  # patients in hospital without a nurse
        self.cumulative_cost = 0
        self.hourly_records = []
        
        # Load data
        self.arrivals_df = pd.read_csv('arrivals.csv')
        self.departures_df = pd.read_csv('departures.csv')
        self.departure_options = self.departures_df['Leaving'].dropna().tolist()
    
    def get_arrivals(self, hour):
        """Get total arrivals and ambulance arrivals for a given hour from CSV"""
        row = self.arrivals_df[self.arrivals_df['Time'] == hour]
        if row.empty:
            return 0, 0
        
        ambulance = int(row['Emergency Room Ambulance'].values[0])
        total = int(
            ambulance +
            row['Emergency Room Walk in'].values[0] +
            row['Surgery'].values[0] +
            row['Critical Care'].values[0] +
            row['Step Down'].values[0]
        )
        return total, ambulance
    
    def get_departures(self):
        """Get random departures from departure options"""
        if not self.departure_options:
            return 0
        return random.choice(self.departure_options)
    
    def simulate_hour(self, hour):
        """Simulate one hour of operations"""
        if self.budget is not None:
            base_nurse_cost = self.current_nurses * self.nurse_cost
            if base_nurse_cost > self.budget:
                return {
                    'status': 'INFEASIBLE',
                    'reason': 'Starting nurse payroll exceeds budget',
                    'hour': hour,
                    'arrivals': 0,
                    'departures': 0,
                    'current_patients_end': self.current_patients,
                    'current_nurses': self.current_nurses,
                    'hire_nurses': 0,
                    'divert_patients': 0,
                    'waiting_queue': self.waiting_queue,
                    'actual_deaths': 0,
                    'hourly_cost': 0,
                    'cumulative_cost': self.cumulative_cost
                }
        # Get inputs for this hour
        arrivals, ambulance_arrivals = self.get_arrivals(hour)
        treated_patients = max(0, self.current_patients - self.waiting_queue)
        departures = min(self.get_departures(), treated_patients)
        
        # Make optimal decision (single unified function handles budget or unlimited)
        result = optimal_staffing_decision(
            self.current_patients, arrivals, departures, 
            self.current_nurses, ambulance_arrivals, self.max_rooms,
            self.nurse_cost, self.diversion_cost, self.death_cost,
            self.budget
        )
        if result.get('status') == 'INFEASIBLE':
            return {
                'status': 'INFEASIBLE',
                'reason': result.get('reason', 'Infeasible'),
                'hour': hour,
                'arrivals': arrivals,
                'departures': departures,
                'current_patients_end': self.current_patients,
                'current_nurses': self.current_nurses,
                'hire_nurses': 0,
                'divert_patients': 0,
                'waiting_queue': self.waiting_queue,
                'actual_deaths': 0,
                'hourly_cost': 0,
                'cumulative_cost': self.cumulative_cost
            }
        
        # Update state
        self.current_nurses += result['hire_nurses']
        self.current_patients = result['active_patients']
        self.waiting_queue = result['waiting']
        self.cumulative_cost += result['total_cost']
        
        # Record this hour
        record = {
            'status': result.get('status', 'OK'),
            'hour': hour,
            'arrivals': arrivals,
            'departures': departures,
            'current_patients_end': self.current_patients,
            'current_nurses': self.current_nurses,
            'hire_nurses': result['hire_nurses'],
            'divert_patients': result['divert_patients'],
            'waiting_queue': self.waiting_queue,
            'actual_deaths': result['actual_deaths'],
            'hourly_cost': result['total_cost'],
            'cumulative_cost': self.cumulative_cost
        }
        self.hourly_records.append(record)
        
        return record
    
    def run_simulation(self, hours=24):
        """Run simulation for specified number of hours"""
        print("="*130)
        print("HOSPITAL STAFFING OPTIMIZATION SIMULATION")
        print("="*130)
        print(f"Initial State: {self.current_patients} patients, {self.current_nurses} nurses, {self.max_rooms} total rooms")
        if self.budget is not None:
            print(f"Hourly Budget: ${self.budget:,.0f}  |  Nurse: ${self.nurse_cost:,.0f}/hr  |  Diversion: ${self.diversion_cost:,.0f}  |  Death: ${self.death_cost:,.0f}")
        print("="*130)
        print(f"{'Hour':>4} | {'Arr':>4} | {'Dep':>4} | {'Patients':>8} | {'Nurses':>7} | "
              f"{'Hire':>4} | {'Divert':>6} | {'Queue':>5} | {'Deaths':>6} | {'Cost':>12} | {'Cumul.$':>14}")
        print("-"*130)
        
        for hour in range(hours):
            record = self.simulate_hour(hour)
            if record.get('status') == 'INFEASIBLE':
                print(f"{hour:4d} | {'-':>4} | {'-':>4} | {record['current_patients_end']:8d} | "
                    f"{record['current_nurses']:7d} | {'-':>4} | {'-':>6} | "
                    f"{record['waiting_queue']:5d} | {'-':>6} | {'-':>12} | {'-':>14}")
                print("\nINFEASIBLE: starting nurse payroll exceeds budget.")
                return
            print(f"{hour:4d} | {record['arrivals']:4d} | {record['departures']:4d} | "
                  f"{record['current_patients_end']:8d} | {record['current_nurses']:7d} | "
                  f"{record['hire_nurses']:4d} | {record['divert_patients']:6d} | "
                  f"{record['waiting_queue']:5d} | {record['actual_deaths']:6d} | "
                  f"${record['hourly_cost']:11,.0f} | ${record['cumulative_cost']:13,.0f}")
        
        print("="*130)
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics"""
        df = pd.DataFrame(self.hourly_records)
        
        print("\nSUMMARY STATISTICS")
        print("-"*130)
        print(f"Total Arrivals:           {df['arrivals'].sum():6.0f}")
        print(f"Total Departures:         {df['departures'].sum():6.0f}")
        print(f"Total Nurses Hired:       {df['hire_nurses'].sum():6.0f}")
        print(f"Total Patients Diverted:  {df['divert_patients'].sum():6.0f}")
        print(f"Total Actual Deaths:      {df['actual_deaths'].sum():6.0f}")
        print(f"Peak Waiting Queue:       {df['waiting_queue'].max():6.0f}")
        print(f"Average Hospital Census:  {df['current_patients_end'].mean():6.2f}")
        print(f"Peak Hospital Census:     {df['current_patients_end'].max():6.0f}")
        print(f"Average Nurse Staffing:   {df['current_nurses'].mean():6.2f}")
        print(f"Peak Nurse Staffing:      {df['current_nurses'].max():6.0f}")
        print(f"\nFinal State: {self.current_patients} patients, {self.current_nurses} nurses, {self.waiting_queue} waiting")
        print(f"TOTAL COST: ${self.cumulative_cost:,.0f}")
        print("="*130)

    def run_optimization(self, hours=24):
        """Run the MIP optimization model and display results"""
        # Build deterministic arrivals list from CSV
        arrivals_per_hour = []
        ambulance_per_hour = []
        for hour in range(hours):
            total, amb = self.get_arrivals(hour)
            arrivals_per_hour.append(total)
            ambulance_per_hour.append(amb)
        
        expected_departures = np.mean(self.departure_options)
        
        print("="*130)
        print("HOSPITAL STAFFING OPTIMIZATION — MIP MODEL (deterministic, expected values)")
        print("="*130)
        print(f"Initial State: {self.current_patients} patients, {self.current_nurses} nurses, {self.max_rooms} total rooms")
        print(f"Expected Departures/hr: {expected_departures:.1f}  |  Death Rate: 10% of waiting patients (expected)")
        if self.budget is not None:
            print(f"Hourly Budget: ${self.budget:,.0f}  |  Nurse: ${self.nurse_cost:,.0f}/hr  |  Diversion: ${self.diversion_cost:,.0f}  |  Death: ${self.death_cost:,.0f}")
        else:
            print(f"Budget: Unlimited  |  Nurse: ${self.nurse_cost:,.0f}/hr  |  Diversion: ${self.diversion_cost:,.0f}  |  Death: ${self.death_cost:,.0f}")
        print("="*130)
        
        result = optimize_staffing(
            hours=hours,
            initial_occupancy=self.current_patients,
            initial_nurses=self.current_nurses,
            arrivals_per_hour=arrivals_per_hour,
            ambulance_per_hour=ambulance_per_hour,
            expected_departures=expected_departures,
            max_rooms=self.max_rooms,
            nurse_cost=self.nurse_cost,
            diversion_cost=self.diversion_cost,
            death_cost=self.death_cost,
            budget=self.budget
        )
        
        if result['status'] != 'Optimal':
            print(f"\nSolver status: {result['status']} — no feasible solution found.")
            print("Try increasing the budget or relaxing constraints.")
            return
        
        print(f"{'Hour':>4} | {'Arr':>4} | {'Dep':>5} | {'Patients':>8} | {'Nurses':>7} | "
              f"{'Hire':>4} | {'Divert':>6} | {'Queue':>5} | {'E[Death]':>8} | {'Cost':>12} | {'Cumul.$':>14}")
        print("-"*130)
        
        for r in result['hourly_plan']:
            print(f"{r['hour']:4d} | {r['arrivals']:4d} | {r['departures']:5.1f} | "
                  f"{r['current_patients_end']:8.1f} | {r['current_nurses']:7d} | "
                  f"{r['hire_nurses']:4d} | {r['divert_patients']:6d} | "
                  f"{r['waiting_queue']:5.1f} | {r['expected_deaths']:8.2f} | "
                  f"${r['hourly_cost']:11,.0f} | ${r['cumulative_cost']:13,.0f}")
        
        print("="*130)
        
        s = result['summary']
        print("\nSUMMARY STATISTICS (Expected Values)")
        print("-"*130)
        print(f"Total Arrivals:           {s['total_arrivals']:6.0f}")
        print(f"Total Departures:         {s['total_departures']:6.1f}")
        print(f"Total Nurses Hired:       {s['total_hired']:6.0f}")
        print(f"Total Patients Diverted:  {s['total_diverted']:6.0f}")
        print(f"Total Expected Deaths:    {s['total_expected_deaths']:6.2f}")
        print(f"Peak Waiting Queue:       {s['peak_waiting']:6.1f}")
        print(f"Average Hospital Census:  {s['avg_census']:6.2f}")
        print(f"Peak Hospital Census:     {s['peak_census']:6.1f}")
        print(f"Average Nurse Staffing:   {s['avg_nurses']:6.2f}")
        print(f"Peak Nurse Staffing:      {s['peak_nurses']:6.0f}")
        print(f"\nFinal State: {s['final_patients']:.1f} patients, {s['final_nurses']} nurses, {s['final_waiting']:.1f} waiting")
        print(f"TOTAL EXPECTED COST: ${s['total_cost']:,.0f}")
        print("="*130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hospital Staffing Optimization Simulation')
    parser.add_argument('--mode', type=str, choices=['greedy', 'optimize'], default='greedy',
                        help='greedy = hour-by-hour heuristic (stochastic), optimize = MIP look-ahead (deterministic expected values). Default: greedy')
    parser.add_argument('--budget', type=float, default=None, 
                        help='Hard cap on total controllable spending per hour: nurses + diversions (default: None = unlimited)')
    parser.add_argument('--starting-occupancy', type=int, default=55,
                        help='Starting occupancy — patients already in hospital (default: 55)')
    parser.add_argument('--initial-nurses', type=int, default=55,
                        help='Initial number of nurses (default: 55)')
    parser.add_argument('--max-rooms', type=int, default=88,
                        help='Maximum room capacity (default: 88)')
    parser.add_argument('--hours', type=int, default=24,
                        help='Number of hours to simulate (default: 24)')
    parser.add_argument('--nurse-cost', type=float, default=150,
                        help='Cost per nurse per hour (default: 150)')
    parser.add_argument('--diversion-cost', type=float, default=1000,
                        help='Cost per diverted patient (default: 1000)')
    parser.add_argument('--death-cost', type=float, default=10000000,
                        help='Cost per patient death (default: 10000000)')
    
    args = parser.parse_args()
    
    sim = HospitalSimulation(
        initial_occupancy=args.starting_occupancy,
        initial_nurses=args.initial_nurses,
        budget=args.budget,
        max_rooms=args.max_rooms,
        nurse_cost=args.nurse_cost,
        diversion_cost=args.diversion_cost,
        death_cost=args.death_cost
    )
    
    if args.mode == 'optimize':
        sim.run_optimization(hours=args.hours)
    else:
        sim.run_simulation(hours=args.hours)
