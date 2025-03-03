import numpy as np
import matplotlib.pyplot as plt
import random

# Constants from the problem
TH1_hot, TH1_cold = 90, 25
TH2_hot, TH2_cold = 80, 30
TC1_cold, TC1_hot = 15, 70
TC2_cold, TC2_hot = 25, 60

H1_duty = 500  # MJ/min
H2_duty = 400  # MJ/min
C1_duty = 600  # MJ/min
C2_duty = 400  # MJ/min

# Calculate heat capacity flow rates
mCp_H1 = H1_duty / (TH1_hot - TH1_cold)  # = 7.69
mCp_H2 = H2_duty / (TH2_hot - TH2_cold)  # = 8.00
mCp_C1 = C1_duty / (TC1_hot - TC1_cold)  # = 10.91
mCp_C2 = C2_duty / (TC2_hot - TC2_cold)  # = 11.43

# Economic parameters
U = 0.075  # MJ/m2-min-C
project_lifetime = 20000  # hours
minutes_per_hour = 60
LPS_cost = 0.039  # NOK/MJ
CW_cost = 0.048  # NOK/MJ
LPS_emissions = 0.080  # kgCO2e/MJ
CW_emissions = 0.055  # kgCO2e/MJ
exchanger_capital_base = 15000000  # NOK
exchanger_capital_per_area = 110000  # NOK/m2
exchanger_emissions_per_area = 80000  # kgCO2e/m2

def log_mean_temp_diff(delta_T1, delta_T2):
    """Calculate log mean temperature difference safely"""
    if abs(delta_T1 - delta_T2) < 0.001:
        return delta_T1  # If they're very close, avoid numerical issues
    if delta_T1 <= 0 or delta_T2 <= 0:
        return None  # Invalid temperature difference
    return (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)

def evaluate_design(Q11, Q12, Q21, Q22):
    """
    Evaluate a design for feasibility and calculate objectives
    
    Returns dict with results or None if infeasible
    """
    # Check if duties are non-negative
    if any(q < 0 for q in [Q11, Q12, Q21, Q22]):
        return None

    # Calculate intermediate temperatures
    TH1A = TH1_hot - Q11 / mCp_H1
    TH1B = TH1A - Q12 / mCp_H1
    TH2A = TH2_hot - Q22 / mCp_H2
    TH2B = TH2A - Q21 / mCp_H2
    TC1A = TC1_cold + Q21 / mCp_C1
    TC1B = TC1A + Q11 / mCp_C1
    TC2A = TC2_cold + Q12 / mCp_C2
    TC2B = TC2A + Q22 / mCp_C2
    
    # Check energy balances - duties don't exceed stream requirements
    if Q11 + Q12 > H1_duty or Q21 + Q22 > H2_duty:
        return None
    if Q11 + Q21 > C1_duty or Q12 + Q22 > C2_duty:
        return None
    
    # Check 2nd law of thermodynamics (hot stream must be hotter than cold stream)
    if Q11 > 0 and (TH1_hot <= TC1B or TH1A <= TC1A):
        return None
    if Q12 > 0 and (TH1A <= TC2B or TH1B <= TC2A):
        return None
    if Q21 > 0 and (TH2A <= TC1B or TH2B <= TC1A):
        return None
    if Q22 > 0 and (TH2_hot <= TC2B or TH2A <= TC2A):
        return None
    
    # Calculate areas of exchangers that are used
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
    
    # Calculate utility requirements
    LPS_required = (C1_duty + C2_duty - Q11 - Q12 - Q21 - Q22)  # MJ/min
    CW_required = (H1_duty + H2_duty - Q11 - Q12 - Q21 - Q22)  # MJ/min
    
    # Calculate costs
    total_area = sum(areas)
    
    # Capital cost (NOK)
    capital_cost = exchanger_count * exchanger_capital_base + exchanger_capital_per_area * total_area
    
    # Operating cost (NOK for project lifetime)
    operating_cost = (LPS_required * LPS_cost + CW_required * CW_cost) * minutes_per_hour * project_lifetime
    
    # Total lifetime cost (NOK)
    total_cost = capital_cost + operating_cost
    
    # GHG emissions
    capital_emissions = total_area * exchanger_emissions_per_area  # kgCO2e
    operating_emissions = (LPS_required * LPS_emissions + CW_required * CW_emissions) * minutes_per_hour * project_lifetime  # kgCO2e
    total_emissions = capital_emissions + operating_emissions  # kgCO2e
    
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

def latin_hypercube_sampling(n_samples, dimension, bounds):
    """
    Generate Latin Hypercube samples
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    dimension : int
        Dimension of the design space
    bounds : list of tuples
        List of (min, max) pairs for each dimension
    
    Returns:
    --------
    numpy.ndarray
        Array of samples with shape (n_samples, dimension)
    """
    # Initialize result array
    result = np.zeros((n_samples, dimension))
    
    # Generate samples for each dimension
    for i in range(dimension):
        # Create bins
        bins = np.linspace(0, 1, n_samples + 1)
        
        # Generate points in each bin
        points = np.random.uniform(bins[:-1], bins[1:])
        
        # Shuffle points
        np.random.shuffle(points)
        
        # Scale to the actual range
        min_val, max_val = bounds[i]
        result[:, i] = min_val + points * (max_val - min_val)
    
    return result

def generate_design_candidates(n_samples=1000):
    """Generate and evaluate HEN design candidates"""
    # Define bounds for each Q value
    bounds = [(0, min(H1_duty, C1_duty)), (0, min(H1_duty, C2_duty)), 
              (0, min(H2_duty, C1_duty)), (0, min(H2_duty, C2_duty))]
    
    # Generate samples using Latin Hypercube Sampling
    samples = latin_hypercube_sampling(n_samples, 4, bounds)
    
    # Evaluate each sample
    feasible_designs = []
    for i in range(n_samples):
        Q11, Q12, Q21, Q22 = samples[i]
        result = evaluate_design(Q11, Q12, Q21, Q22)
        if result is not None:
            feasible_designs.append(result)
    
    # Also add designs with some exchangers not used
    # For each pattern of exchanger use
    for pattern in range(1, 16):  # 2^4 - 1 = 15 possible non-empty patterns
        for _ in range(n_samples // 15):  # Distribute samples
            # Create binary pattern (1 = exchanger used, 0 = not used)
            binary = [(pattern >> i) & 1 for i in range(4)]
            
            # Generate random Q values within bounds
            Q_values = []
            for i, use in enumerate(binary):
                if use:
                    min_val, max_val = bounds[i]
                    Q_values.append(random.uniform(0, max_val))
                else:
                    Q_values.append(0)
            
            # Evaluate design
            result = evaluate_design(*Q_values)
            if result is not None:
                feasible_designs.append(result)
    
    return feasible_designs

# Problem 1 solution (placeholder - replace with your actual solution)
# This represents a solution from the pinch method
problem1_solution = evaluate_design(300, 200, 200, 100)

# Generate candidates
designs = generate_design_candidates(1000)
print(f"Generated {len(designs)} feasible designs")

# Extract data for plotting
capital_costs = [d['capital_cost'] for d in designs]
operating_costs = [d['operating_cost'] for d in designs]
total_costs = [d['total_cost'] for d in designs]
total_emissions = [d['total_emissions'] for d in designs]

# Find special points
min_cost_idx = np.argmin(total_costs)
min_emissions_idx = np.argmin(total_emissions)

# Plot pareto diagram
plt.figure(figsize=(10, 8))
plt.scatter(operating_costs, capital_costs, alpha=0.7, label='Feasible Designs')

# Highlight special points
plt.scatter(operating_costs[min_cost_idx], capital_costs[min_cost_idx], 
            color='red', s=100, marker='*', label='Min. Lifetime Cost')
plt.scatter(operating_costs[min_emissions_idx], capital_costs[min_emissions_idx], 
            color='green', s=100, marker='*', label='Min. Lifetime Emissions')

# Highlight Problem 1 solution if available
if problem1_solution is not None:
    plt.scatter(problem1_solution['operating_cost'], problem1_solution['capital_cost'], 
                color='blue', s=100, marker='*', label='Problem 1 Solution')

plt.xlabel('Operating Cost (NOK)')
plt.ylabel('Capital Cost (NOK)')
plt.title('Pareto Chart: Capital Cost vs Operating Cost')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pareto_capital_vs_operating.png')
plt.show()

# Print details of special points
print("\nPoint with minimum lifetime cost:")
min_cost = designs[min_cost_idx]
print(f"Capital cost: {min_cost['capital_cost']:,.2f} NOK")
print(f"Operating cost: {min_cost['operating_cost']:,.2f} NOK")
print(f"Total cost: {min_cost['total_cost']:,.2f} NOK")
print(f"Q values: Q11={min_cost['Q11']:.2f}, Q12={min_cost['Q12']:.2f}, Q21={min_cost['Q21']:.2f}, Q22={min_cost['Q22']:.2f}")

print("\nPoint with minimum lifetime emissions:")
min_emissions = designs[min_emissions_idx]
print(f"Capital cost: {min_emissions['capital_cost']:,.2f} NOK")
print(f"Operating cost: {min_emissions['operating_cost']:,.2f} NOK")
print(f"Total cost: {min_emissions['total_cost']:,.2f} NOK")
print(f"Q values: Q11={min_emissions['Q11']:.2f}, Q12={min_emissions['Q12']:.2f}, Q21={min_emissions['Q21']:.2f}, Q22={min_emissions['Q22']:.2f}")