import numpy as np
import matplotlib.pyplot as plt
import random

# Constants - stream temperatures and duties
TH1_hot, TH1_cold = 90, 25
TH2_hot, TH2_cold = 80, 30
TC1_cold, TC1_hot = 15, 70
TC2_cold, TC2_hot = 25, 60

H1_duty = 500
H2_duty = 400
C1_duty = 600
C2_duty = 400

# Calculate mCp values
mCp_H1 = H1_duty / (TH1_hot - TH1_cold)
mCp_H2 = H2_duty / (TH2_hot - TH2_cold)
mCp_C1 = C1_duty / (TC1_hot - TC1_cold)
mCp_C2 = C2_duty / (TC2_hot - TC2_cold)

# Cost and emissions factors
U = 0.075
project_lifetime = 20000
minutes_per_hour = 60
LPS_cost, CW_cost = 0.039, 0.048
LPS_emissions, CW_emissions = 0.080, 0.055
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
    # Check for negative heat duties
    if any(q < 0 for q in [Q11, Q12, Q21, Q22]):
        return None

    # Calculate stream temperatures
    TH1A = TH1_hot - Q11 / mCp_H1
    TH1B = TH1A - Q12 / mCp_H1
    TH2A = TH2_hot - Q22 / mCp_H2
    TH2B = TH2A - Q21 / mCp_H2
    TC1A = TC1_cold + Q21 / mCp_C1
    TC1B = TC1A + Q11 / mCp_C1
    TC2A = TC2_cold + Q12 / mCp_C2
    TC2B = TC2A + Q22 / mCp_C2
    
    # Check energy balance constraints
    if Q11 + Q12 > H1_duty or Q21 + Q22 > H2_duty or Q11 + Q21 > C1_duty or Q12 + Q22 > C2_duty:
        return None
    
    # Check temperature feasibility
    if (Q11 > 0 and (TH1_hot <= TC1B or TH1A <= TC1A)) or \
       (Q12 > 0 and (TH1A <= TC2B or TH1B <= TC2A)) or \
       (Q21 > 0 and (TH2A <= TC1B or TH2B <= TC1A)) or \
       (Q22 > 0 and (TH2_hot <= TC2B or TH2A <= TC2A)):
        return None
    
    # Calculate areas and count exchangers
    areas = []
    exchanger_count = 0
    
    for Q, hot_in, hot_out, cold_in, cold_out in [
        (Q11, TH1_hot, TH1A, TC1A, TC1B),
        (Q12, TH1A, TH1B, TC2A, TC2B),
        (Q21, TH2A, TH2B, TC1A, TC1B),
        (Q22, TH2_hot, TH2A, TC2A, TC2B)
    ]:
        if Q > 0:
            lmtd = log_mean_temp_diff(hot_in - cold_out, hot_out - cold_in)
            if lmtd is None:
                return None
            areas.append(Q / (U * lmtd))
            exchanger_count += 1
    
    # Calculate utilities, costs and emissions
    total_heat_recovery = Q11 + Q12 + Q21 + Q22
    LPS_required = (C1_duty + C2_duty - total_heat_recovery)
    CW_required = (H1_duty + H2_duty - total_heat_recovery)
    
    total_area = sum(areas)
    capital_cost = exchanger_count * exchanger_capital_base + exchanger_capital_per_area * total_area
    operating_cost = (LPS_required * LPS_cost + CW_required * CW_cost) * minutes_per_hour * project_lifetime
    total_cost = capital_cost + operating_cost
    
    capital_emissions = total_area * exchanger_emissions_per_area
    operating_emissions = (LPS_required * LPS_emissions + CW_required * CW_emissions) * minutes_per_hour * project_lifetime
    total_emissions = capital_emissions + operating_emissions
    
    return {
        'Q11': Q11, 'Q12': Q12, 'Q21': Q21, 'Q22': Q22,
        'capital_cost': capital_cost, 'operating_cost': operating_cost, 'total_cost': total_cost,
        'capital_emissions': capital_emissions, 'operating_emissions': operating_emissions, 
        'total_emissions': total_emissions, 'LPS': LPS_required, 'CW': CW_required,
        'exchanger_count': exchanger_count, 'total_area': total_area
    }

def objective_function(Q_values, weight=0.5):
    """Calculate weighted objective value combining cost and emissions"""
    design = evaluate_design(*Q_values)
    if design is None:
        return float('inf'), None
    
    obj_value = weight * design['total_cost'] + (1 - weight) * design['total_emissions']
    return obj_value, design

def optimize_network(n_particles=20, max_iterations=500, weight=0.5):
    """Simplified particle swarm optimization"""
    bounds = [(0, min(H1_duty, C1_duty)), (0, min(H1_duty, C2_duty)), 
              (0, min(H2_duty, C1_duty)), (0, min(H2_duty, C2_duty))]
    
    # Initialize particles
    particles = np.zeros((n_particles, 4))
    particles[0] = [500, 0, 0, 400]  # Start with known solution
    
    # Initialize remaining particles with random values
    for i in range(1, n_particles):
        particles[i] = [random.uniform(b[0], b[1]) for b in bounds]
    
    # Initialize velocities
    velocities = np.random.uniform(-0.1, 0.1, (n_particles, 4))
    
    # Initialize best positions and values
    personal_best = particles.copy()
    personal_best_values = np.full(n_particles, float('inf'))
    personal_best_designs = [None] * n_particles
    
    global_best_value = float('inf')
    global_best_position = np.zeros(4)
    global_best_design = None
    history = []
    
    # Evaluate initial positions
    for i in range(n_particles):
        obj_value, design = objective_function(particles[i], weight)
        if obj_value < global_best_value and design is not None:
            global_best_value = obj_value
            global_best_position = particles[i].copy()
            global_best_design = design
            personal_best_values[i] = obj_value
            personal_best[i] = particles[i].copy()
            personal_best_designs[i] = design
    
    if global_best_design:
        history.append(global_best_design)
    
    # Main optimization loop
    no_improvement = 0
    for iteration in range(max_iterations):
        # Update inertia weight (decreases over time)
        w = 0.9 - 0.5 * (iteration / max_iterations)
        
        for i in range(n_particles):
            # Update velocities and positions
            for j in range(4):
                r1, r2 = random.random(), random.random()
                
                # Cognitive and social components
                velocities[i, j] = w * velocities[i, j] + \
                                  2.0 * r1 * (personal_best[i, j] - particles[i, j]) + \
                                  2.0 * r2 * (global_best_position[j] - particles[i, j])
                
                # Update position
                particles[i, j] += velocities[i, j]
                
                # Enforce bounds
                particles[i, j] = max(bounds[j][0], min(bounds[j][1], particles[i, j]))
            
            # Evaluate new position
            obj_value, design = objective_function(particles[i], weight)
            if obj_value < personal_best_values[i] and design is not None:
                personal_best_values[i] = obj_value
                personal_best[i] = particles[i].copy()
                personal_best_designs[i] = design
                
                if obj_value < global_best_value:
                    global_best_value = obj_value
                    global_best_position = particles[i].copy()
                    global_best_design = design
                    history.append(global_best_design)
                    no_improvement = 0
                    continue
        
        # Early stopping if no improvement
        no_improvement += 1
        if no_improvement > 20:
            break
    
    return history

def generate_designs(n_samples=1000):
    """Generate a large number of feasible design candidates"""
    # Get initial designs from optimization
    best_designs = optimize_network(n_particles=30, max_iterations=300, weight=0.5)
    print(f"Found {len(best_designs)} designs through optimization")
    
    bounds = [(0, min(H1_duty, C1_duty)), (0, min(H1_duty, C2_duty)), 
              (0, min(H2_duty, C1_duty)), (0, min(H2_duty, C2_duty))]
    
    all_designs = best_designs.copy()
    attempts = 0
    max_attempts = n_samples * 10  # Limit total attempts to prevent infinite loops
    
    # Generate additional random designs to reach n_samples
    while len(all_designs) < n_samples and attempts < max_attempts:
        # Generate a random design within bounds
        Q_values = [random.uniform(b[0], b[1]) for b in bounds]
        
        # More strategic sampling - favor designs with higher heat recovery
        if random.random() < 0.7:  # 70% chance to generate designs with higher heat exchange
            total_bound = sum([b[1] for b in bounds])
            target_exchange = random.uniform(0.3 * total_bound, 0.8 * total_bound)
            scale_factor = target_exchange / (sum(Q_values) + 0.001)  # Avoid division by zero
            Q_values = [min(q * scale_factor, bounds[i][1]) for i, q in enumerate(Q_values)]
        
        result = evaluate_design(*Q_values)
        attempts += 1
        
        if result is not None:
            all_designs.append(result)
        
        if attempts % 1000 == 0:
            print(f"Generated {len(all_designs)} designs after {attempts} attempts")
    
    print(f"Generated a total of {len(all_designs)} feasible designs")
    return all_designs

def visualize_results(designs):
    """Create visualization of results"""
    # Extract data for plotting
    capital_costs = [d['capital_cost'] for d in designs]
    operating_costs = [d['operating_cost'] for d in designs]
    total_costs = [d['total_cost'] for d in designs]
    total_emissions = [d['total_emissions'] for d in designs]
    
    min_cost_idx = np.argmin(total_costs)
    min_emissions_idx = np.argmin(total_emissions)
    
    # Create Pareto chart
    plt.figure(figsize=(10, 6))
    plt.scatter(total_costs, total_emissions, alpha=0.7, label='Designs')
    
    plt.scatter(total_costs[min_cost_idx], total_emissions[min_cost_idx], 
                color='red', s=100, marker='*', label='Min Cost')
    plt.scatter(total_costs[min_emissions_idx], total_emissions[min_emissions_idx], 
                color='green', s=100, marker='*', label='Min Emissions')
    
    plt.xlabel('Lifetime Costs (NOK)')
    plt.ylabel('Lifetime GHG Emissions (kgCO2e)')
    plt.title('Pareto Chart: Emissions vs Costs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Print optimal designs
    print("\nOptimal Designs:")
    for label, idx in [("Minimum Cost", min_cost_idx), ("Minimum Emissions", min_emissions_idx)]:
        print(f"\n{label}:")
        print(f"Heat duties: Q11={designs[idx]['Q11']:.1f}, Q12={designs[idx]['Q12']:.1f}, "
              f"Q21={designs[idx]['Q21']:.1f}, Q22={designs[idx]['Q22']:.1f}")
        print(f"Total Cost: {designs[idx]['total_cost']:,.0f} NOK")
        print(f"Total Emissions: {designs[idx]['total_emissions']:,.0f} kgCO2e")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    # First check the basic solution
    base_design = evaluate_design(500, 0, 0, 400)
    print("Base design:")
    print(f"Total Cost: {base_design['total_cost']:,.0f} NOK")
    print(f"Total Emissions: {base_design['total_emissions']:,.0f} kgCO2e")
    
    # Generate a large set of designs
    designs = generate_designs(1000)
    
    # Visualize results
    visualize_results(designs)