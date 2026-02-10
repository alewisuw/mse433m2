"""
Real-Time Hospital Staffing Decision System

Problem:
- Discrete time: 1-hour intervals
- 1:1 nurse-to-patient ratio required
- Nurses are interchangeable
- Patients cannot wait without nurses

Costs:
- Nurse hiring: $150 per nurse per hour
- Patient diversion: $1,000 per patient
- Patient death: $10,000,000 per patient (10% chance per hour in hospital)

Objective:
Minimize total hourly cost subject to constraint that nurses >= patients

Note: Deaths only occur for patients waiting WITHOUT a nurse. Patients in the hospital
with nurses (1:1 ratio maintained) do not die.
"""

import pandas as pd
import random
import numpy as np
import argparse


def optimal_staffing_decision(current_patients, arrivals, departures, current_nurses, 
                             waiting_queue=0, max_rooms=88, 
                             nurse_cost=150, diversion_cost=1000, death_cost=10000000,
                             budget=None):
    """
    Determine optimal staffing decisions for one hour.
    
    Budget is a HARD CAP on total controllable spending (nurses + diversions).
    If the budget is exhausted, remaining uncovered patients wait without a nurse
    and face a 10% chance of dying each hour.
    
    Priority order within budget:
      1. Pay existing nurses (mandatory, cannot fire mid-shift)
      2. Hire additional nurses for uncovered patients
      3. Divert patients we can't cover
      4. If budget exhausted: patients wait (risk death)
    
    Args:
        current_patients: patients currently in hospital (including any waiting from last hour)
        arrivals: new patient arrivals this hour
        departures: patients leaving this hour
        current_nurses: nurses currently staffed
        waiting_queue: patients waiting without a nurse from previous hours
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
    
    # Step 2: Room capacity constraint — must divert if over room limit (no cost, physical limit)
    if potential_patients > max_rooms:
        room_overflow_divert = potential_patients - max_rooms
        potential_patients = max_rooms
    else:
        room_overflow_divert = 0
    
    # Step 3: Determine nurse gap
    nurses_needed = max(0, potential_patients - current_nurses)
    
    # Step 4: Allocate budget
    # Mandatory cost: pay all existing nurses
    base_nurse_cost = current_nurses * nurse_cost
    
    if budget is not None:
        remaining_budget = budget - base_nurse_cost
        
        if remaining_budget < 0:
            # Can't even afford existing nurses — no hiring, no diversion
            # All uncovered patients wait
            remaining_budget = 0
        
        # 4a: Hire as many nurses as budget allows
        if nurses_needed > 0 and remaining_budget > 0:
            affordable_hires = min(nurses_needed, int(remaining_budget // nurse_cost))
            hire_nurses = affordable_hires
            remaining_budget -= hire_nurses * nurse_cost
        else:
            hire_nurses = 0
        
        still_uncovered = nurses_needed - hire_nurses
        
        # 4b: Divert as many uncovered patients as budget allows
        if still_uncovered > 0 and remaining_budget > 0:
            affordable_diversions = min(still_uncovered, int(remaining_budget // diversion_cost))
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
        'hire_nurses': hire_nurses,
        'divert_patients': divert_patients,
        'waiting': waiting,
        'actual_deaths': actual_deaths,
        'active_patients': active_patients,
        'total_cost': total_cost
    }


class HospitalSimulation:
    """Simulation of hospital operations over multiple hours"""
    
    def __init__(self, initial_patients=55, initial_nurses=55, budget=None, max_rooms=88,
                 nurse_cost=150, diversion_cost=1000, death_cost=10000000):
        self.current_patients = initial_patients
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
        """Get total arrivals for a given hour from CSV"""
        row = self.arrivals_df[self.arrivals_df['Time'] == hour]
        if row.empty:
            return 0
        
        # Sum all arrival types
        total = int(
            row['Emergency Room Ambulance'].values[0] +
            row['Emergency Room Walk in'].values[0] +
            row['Surgery'].values[0] +
            row['Critical Care'].values[0] +
            row['Step Down'].values[0]
        )
        return total
    
    def get_departures(self):
        """Get random departures from departure options"""
        if not self.departure_options:
            return 0
        return random.choice(self.departure_options)
    
    def simulate_hour(self, hour):
        """Simulate one hour of operations"""
        # Get inputs for this hour
        arrivals = self.get_arrivals(hour)
        departures = min(self.get_departures(), self.current_patients)
        
        # Make optimal decision (single unified function handles budget or unlimited)
        result = optimal_staffing_decision(
            self.current_patients, arrivals, departures, 
            self.current_nurses, self.waiting_queue, self.max_rooms,
            self.nurse_cost, self.diversion_cost, self.death_cost,
            self.budget
        )
        
        # Update state
        self.current_nurses += result['hire_nurses']
        self.current_patients = result['active_patients']
        self.waiting_queue = result['waiting']
        self.cumulative_cost += result['total_cost']
        
        # Record this hour
        record = {
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hospital Staffing Optimization Simulation')
    parser.add_argument('--budget', type=float, default=None, 
                        help='Hard cap on total controllable spending per hour: nurses + diversions (default: None = unlimited)')
    parser.add_argument('--initial-patients', type=int, default=55,
                        help='Initial number of patients (default: 55)')
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
    
    # Run simulation with specified parameters
    sim = HospitalSimulation(
        initial_patients=args.initial_patients,
        initial_nurses=args.initial_nurses,
        budget=args.budget,
        max_rooms=args.max_rooms,
        nurse_cost=args.nurse_cost,
        diversion_cost=args.diversion_cost,
        death_cost=args.death_cost
    )
    sim.run_simulation(hours=args.hours)
