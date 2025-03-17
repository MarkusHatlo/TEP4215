import numpy as np
import matplotlib.pyplot as plt
import random


TH1_hot, TH1_cold = 90, 25
TH2_hot, TH2_cold = 80, 30
TC1_cold, TC1_hot = 15, 70
TC2_cold, TC2_hot = 25, 60

H1_duty = 500
H2_duty = 400
C1_duty = 600
C2_duty = 400

mCp_H1 = H1_duty / (TH1_hot - TH1_cold)
mCp_H2 = H2_duty / (TH2_hot - TH2_cold)
mCp_C1 = C1_duty / (TC1_hot - TC1_cold)
mCp_C2 = C2_duty / (TC2_hot - TC2_cold)

U = 0.075
project_lifetime = 20000
minutes_per_hour = 60
LPS_cost = 0.039
CW_cost = 0.048
LPS_emissions = 0.080
CW_emissions = 0.055
exchanger_capital_base = 15000000
exchanger_capital_per_area = 110000
exchanger_emissions_per_area = 80000

def log_mean_temp_diff(delta_T1, delta_T2):
    if abs(delta_T1 - delta_T2) < 0.001:
        return delta_T1
    if delta_T1 <= 0 or delta_T2 <= 0:
        return None
    return (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)

def evaluate_design(Q11, Q12, Q21, Q22):
    if any(q < 0 for q in [Q11, Q12, Q21, Q22]):
        return None

    TH1A = TH1_hot - Q11 / mCp_H1
    TH1B = TH1A - Q12 / mCp_H1
    TH2A = TH2_hot - Q22 / mCp_H2
    TH2B = TH2A - Q21 / mCp_H2
    TC1A = TC1_cold + Q21 / mCp_C1
    TC1B = TC1A + Q11 / mCp_C1
    TC2A = TC2_cold + Q12 / mCp_C2
    TC2B = TC2A + Q22 / mCp_C2
    
    if Q11 + Q12 > H1_duty or Q21 + Q22 > H2_duty:
        return None
    if Q11 + Q21 > C1_duty or Q12 + Q22 > C2_duty:
        return None
    
    if Q11 > 0 and (TH1_hot <= TC1B or TH1A <= TC1A):
        return None
    if Q12 > 0 and (TH1A <= TC2B or TH1B <= TC2A):
        return None
    if Q21 > 0 and (TH2A <= TC1B or TH2B <= TC1A):
        return None
    if Q22 > 0 and (TH2_hot <= TC2B or TH2A <= TC2A):
        return None
    
    areas = []
    exchanger_count = 0
    
    if Q11 > 0:
        lmtd = log_mean_temp_diff(TH1_hot - TC1B, TH1A - TC1A)
        if lmtd is None:
            return None
        areas.append(Q11 / (U * lmtd))
        exchanger_count += 1
    
    if Q12 > 0:
        lmtd = log_mean_temp_diff(TH1A - TC2B, TH1B - TC2A)
        if lmtd is None:
            return None
        areas.append(Q12 / (U * lmtd))
        exchanger_count += 1
    
    if Q21 > 0:
        lmtd = log_mean_temp_diff(TH2A - TC1B, TH2B - TC1A)
        if lmtd is None:
            return None
        areas.append(Q21 / (U * lmtd))
        exchanger_count += 1
    
    if Q22 > 0:
        lmtd = log_mean_temp_diff(TH2_hot - TC2B, TH2A - TC2A)
        if lmtd is None:
            return None
        areas.append(Q22 / (U * lmtd))
        exchanger_count += 1
    
    LPS_required = (C1_duty + C2_duty - Q11 - Q12 - Q21 - Q22)
    CW_required = (H1_duty + H2_duty - Q11 - Q12 - Q21 - Q22)
    
    total_area = sum(areas)
    capital_cost = exchanger_count * exchanger_capital_base + exchanger_capital_per_area * total_area
    operating_cost = (LPS_required * LPS_cost + CW_required * CW_cost) * minutes_per_hour * project_lifetime
    total_cost = capital_cost + operating_cost
    capital_emissions = total_area * exchanger_emissions_per_area
    operating_emissions = (LPS_required * LPS_emissions + CW_required * CW_emissions) * minutes_per_hour * project_lifetime
    total_emissions = capital_emissions + operating_emissions
    
    return {
        'Q11': Q11,
        'Q12': Q12,
        'Q21': Q21,
        'Q22': Q22,
        'capital_cost': capital_cost,
        'operating_cost': operating_cost,
        'total_cost': total_cost,
        'capital_emissions': capital_emissions,
        'operating_emissions': operating_emissions,
        'total_emissions': total_emissions,
        'LPS': LPS_required,
        'CW': CW_required,
        'exchanger_count': exchanger_count,
        'total_area': total_area
    }

def objective_function(Q11, Q12, Q21, Q22, weight=0.5):
    design = evaluate_design(Q11, Q12, Q21, Q22)
    if design is None:
        return float('inf'), None
    
    obj_value = weight * design['total_cost'] + (1 - weight) * design['total_emissions']
    return obj_value, design

def particle_swarm_optimization(n_particles=30, max_iterations=1000, weight=0.5):
    bounds = [(0, min(H1_duty, C1_duty)), (0, min(H1_duty, C2_duty)), 
              (0, min(H2_duty, C1_duty)), (0, min(H2_duty, C2_duty))]
    
    w1 = 2.8
    w2 = 1.3
    d1 = 0.0165
    d2 = 0.158
    d3 = 0.180
    
    particles = np.zeros((n_particles, 4))
    particles[0] = [500, 0, 0, 400]
    
    for i in range(1, n_particles):
        for j in range(4):
            if i % 4 == j:
                particles[i, j] = random.uniform(bounds[j][0], 0.8 * bounds[j][1])
            else:
                particles[i, j] = random.uniform(bounds[j][0], bounds[j][1])
    
    velocities = np.zeros((n_particles, 4))
    for i in range(n_particles):
        for j in range(4):
            max_velocity = 0.25 * (bounds[j][1] - bounds[j][0])
            velocities[i, j] = random.uniform(-max_velocity, max_velocity)
    
    personal_best_positions = particles.copy()
    personal_best_values = np.full(n_particles, float('inf'))
    personal_best_designs = [None] * n_particles
    
    global_best_value = float('inf')
    global_best_position = np.zeros(4)
    global_best_design = None
    
    for i in range(n_particles):
        obj_value, design = objective_function(particles[i, 0], particles[i, 1], 
                                               particles[i, 2], particles[i, 3], weight)
        
        if obj_value < personal_best_values[i]:
            personal_best_values[i] = obj_value
            personal_best_positions[i] = particles[i].copy()
            personal_best_designs[i] = design
            
            if obj_value < global_best_value and design is not None:
                global_best_value = obj_value
                global_best_position = particles[i].copy()
                global_best_design = design
    
    history = []
    if global_best_design is not None:
        history.append(global_best_design)
    
    consecutive_no_improvement = 0
    
    for iteration in range(max_iterations):
        inertia = d1 / (d2 + (iteration / max_iterations) * d3)
        
        for i in range(n_particles):
            obj_value, design = objective_function(particles[i, 0], particles[i, 1], 
                                                   particles[i, 2], particles[i, 3], weight)
            
            if obj_value < personal_best_values[i] and design is not None:
                personal_best_values[i] = obj_value
                personal_best_positions[i] = particles[i].copy()
                personal_best_designs[i] = design
                
                if obj_value < global_best_value:
                    global_best_value = obj_value
                    global_best_position = particles[i].copy()
                    global_best_design = design
                    history.append(global_best_design)
        
        for i in range(n_particles):
            for j in range(4):
                r1, r2 = random.random(), random.random()
                
                cognitive_component = w1 * r1 * (personal_best_positions[i, j] - particles[i, j])
                social_component = w2 * r2 * (global_best_position[j] - particles[i, j])
                
                velocities[i, j] = inertia * velocities[i, j] + cognitive_component + social_component
                
                particles[i, j] += velocities[i, j]
                
                particles[i, j] = max(bounds[j][0], min(bounds[j][1], particles[i, j]))
        
        if iteration > 10 and len(history) >= 2:
            if abs(history[-1]['total_cost'] - history[-2]['total_cost']) < 1e-6 and \
               abs(history[-1]['total_emissions'] - history[-2]['total_emissions']) < 1e-6:
                consecutive_no_improvement += 1
                if consecutive_no_improvement > 20:
                    print(f"Early stopping at iteration {iteration}: No improvement for 20 consecutive iterations")
                    break
            else:
                consecutive_no_improvement = 0
                
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best objective = {global_best_value}")
            if global_best_design is not None:
                print(f"Current best design: Q11={global_best_design['Q11']:.2f}, Q12={global_best_design['Q12']:.2f}, "
                      f"Q21={global_best_design['Q21']:.2f}, Q22={global_best_design['Q22']:.2f}")
    
    print(f"PSO completed after {iteration+1} iterations")
    print(f"Best design found: Q11={global_best_design['Q11']:.2f}, Q12={global_best_design['Q12']:.2f}, "
          f"Q21={global_best_design['Q21']:.2f}, Q22={global_best_design['Q22']:.2f}")
    print(f"Total Cost: {global_best_design['total_cost']:.2f}, "
          f"Total Emissions: {global_best_design['total_emissions']:.2f}")
    
    return history

def generate_design_candidates(n_samples=1000):
    best_designs = particle_swarm_optimization(n_particles=30, max_iterations=1000, weight=0.5)
    
    bounds = [(0, min(H1_duty, C1_duty)), (0, min(H1_duty, C2_duty)), 
              (0, min(H2_duty, C1_duty)), (0, min(H2_duty, C2_duty))]
    
    all_designs = best_designs.copy()
    
    for _ in range(n_samples - len(best_designs)):
        Q_values = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(4)]
        result = evaluate_design(*Q_values)
        if result is not None:
            all_designs.append(result)
    
    return all_designs

if __name__ == "__main__":
    problem1_solution = evaluate_design(500, 0, 0, 400)
    print("Problem 1 Solution:")
    print(f"Q11={problem1_solution['Q11']}, Q12={problem1_solution['Q12']}, "
          f"Q21={problem1_solution['Q21']}, Q22={problem1_solution['Q22']}")
    print(f"Total Cost: {problem1_solution['total_cost']}")
    print(f"Total Emissions: {problem1_solution['total_emissions']}")
    
    designs = generate_design_candidates(1000)
    print(f"Generated {len(designs)} feasible designs")
    
    capital_costs = [d['capital_cost'] for d in designs]
    operating_costs = [d['operating_cost'] for d in designs]
    total_costs = [d['total_cost'] for d in designs]
    total_emissions = [d['total_emissions'] for d in designs]
    
    min_cost_idx = np.argmin(total_costs)
    min_emissions_idx = np.argmin(total_emissions)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(operating_costs, capital_costs, alpha=0.7, label='Feasible Designs')
    
    plt.scatter(operating_costs[min_cost_idx], capital_costs[min_cost_idx], 
                color='red', s=100, marker='*', label='Min. Lifetime Cost')
    plt.scatter(operating_costs[min_emissions_idx], capital_costs[min_emissions_idx], 
                color='green', s=100, marker='*', label='Min. Lifetime Emissions')
    plt.scatter(problem1_solution['operating_cost'], problem1_solution['capital_cost'], 
                color='blue', s=100, marker='*', label='Problem 1 Solution')
    
    plt.xlabel('Operating Cost (NOK)')
    plt.ylabel('Capital Cost (NOK)')
    plt.title('Pareto Chart: Capital Cost vs Operating Cost')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(total_costs, total_emissions, alpha=0.7, label='Feasible Designs')
    
    plt.scatter(total_costs[min_cost_idx], total_emissions[min_cost_idx], 
                color='red', s=100, marker='*', label='Min. Lifetime Cost')
    plt.scatter(total_costs[min_emissions_idx], total_emissions[min_emissions_idx], 
                color='green', s=100, marker='*', label='Min. Lifetime Emissions')
    plt.scatter(problem1_solution['total_cost'], problem1_solution['total_emissions'], 
                color='blue', s=100, marker='*', label='Problem 1 Solution')
    
    plt.xlabel('Lifetime Costs (NOK)')
    plt.ylabel('Lifetime GHG Emissions (kgCO2e)')
    plt.title('Pareto Chart: Lifetime GHG Emissions vs Lifetime Costs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nOptimal Designs:")
    print("\nMinimum Cost Design:")
    print(f"Q11={designs[min_cost_idx]['Q11']:.2f}, Q12={designs[min_cost_idx]['Q12']:.2f}, "
          f"Q21={designs[min_cost_idx]['Q21']:.2f}, Q22={designs[min_cost_idx]['Q22']:.2f}")
    print(f"Total Cost: {designs[min_cost_idx]['total_cost']:.2f}")
    print(f"Total Emissions: {designs[min_cost_idx]['total_emissions']:.2f}")
    
    print("\nMinimum Emissions Design:")
    print(f"Q11={designs[min_emissions_idx]['Q11']:.2f}, Q12={designs[min_emissions_idx]['Q12']:.2f}, "
          f"Q21={designs[min_emissions_idx]['Q21']:.2f}, Q22={designs[min_emissions_idx]['Q22']:.2f}")
    print(f"Total Cost: {designs[min_emissions_idx]['total_cost']:.2f}")
    print(f"Total Emissions: {designs[min_emissions_idx]['total_emissions']:.2f}")