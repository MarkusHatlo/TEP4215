import numpy as np
import matplotlib.pyplot as plt

# Process data from the problem
# Hot stream data
T_h_in = 90  # °C
T_h_out_desired = 35  # °C
MC_h = 10  # MJ/min-°C

# Cold stream data
T_c_in = 30  # °C
T_c_out_desired = 85  # °C
MC_c = 14  # MJ/min-°C

# Utilities data
cost_LPS = 0.039  # NOK/MJ
cost_ChW = 0.048  # NOK/MJ
GHG_LPS = 0.080  # kgCO2e/MJ
GHG_ChW = 0.055  # kgCO2e/MJ

# Heat exchanger data
U = 0.075  # MJ/m²-min-°C
A_min = 25  # m²
A_max = 1000  # m²
cap_cost_fixed = 15000000  # NOK
cap_cost_per_area = 110000  # NOK/m²
GHG_per_area = 80000  # kgCO2e/m²

# Process lifetime
lifetime_hours = 80000  # hours
lifetime_minutes = lifetime_hours * 60  # minutes

# Heat requirement calculation for baseline case
q_h_baseline = MC_h * (T_h_in - T_h_out_desired)  # MJ/min (heat to be removed from hot stream)
q_c_baseline = MC_c * (T_c_out_desired - T_c_in)  # MJ/min (heat to be added to cold stream)

print(f"Baseline heat removal needed from hot stream: {q_h_baseline} MJ/min")
print(f"Baseline heat addition needed for cold stream: {q_c_baseline} MJ/min")

# Function to simulate the co-current heat exchanger for a given area
def cocurrent_hx(A, N=100):
    # Split the exchanger into N sections
    dA = A / N
    
    # Initialize temperatures
    T_h = np.zeros(N+1)
    T_c = np.zeros(N+1)
    
    # Set inlet temperatures
    T_h[0] = T_h_in
    T_c[0] = T_c_in
    
    # Simulate heat exchange through the sections
    for i in range(1, N+1):
        # Heat transfer in section i
        Q_i = U * dA * (T_h[i-1] - T_c[i-1])
        
        # Energy balances
        T_h[i] = T_h[i-1] - Q_i / MC_h
        T_c[i] = T_c[i-1] + Q_i / MC_c
    
    # Heat exchanged
    Q_exchanged = MC_h * (T_h[0] - T_h[N])
    
    # Check if target temperatures are achieved
    T_h_out = T_h[N]
    T_c_out = T_c[N]
    
    # Calculate remaining utility requirements
    Q_ChW = max(0, MC_h * (T_h_out - T_h_out_desired))  # Additional cooling needed
    Q_LPS = max(0, MC_c * (T_c_out_desired - T_c_out))  # Additional heating needed
    
    return {
        'T_h_out': T_h_out,
        'T_c_out': T_c_out,
        'Q_exchanged': Q_exchanged,
        'Q_ChW': Q_ChW,
        'Q_LPS': Q_LPS
    }

# Function to calculate objective functions for a given area
def calculate_objectives(A):
    # Simulate the heat exchanger
    results = cocurrent_hx(A)
    
    # Extract results
    Q_ChW = results['Q_ChW']
    Q_LPS = results['Q_LPS']
    
    # Calculate capital cost of the heat exchanger
    cap_cost = cap_cost_fixed + cap_cost_per_area * A
    
    # Calculate operating costs
    op_cost_per_min = cost_LPS * Q_LPS + cost_ChW * Q_ChW
    total_op_cost = op_cost_per_min * lifetime_minutes
    
    # Calculate total cost (Objective 1)
    total_cost = cap_cost + total_op_cost
    
    # Calculate embedded GHG
    embedded_GHG = GHG_per_area * A
    
    # Calculate operating GHG
    op_GHG_per_min = GHG_LPS * Q_LPS + GHG_ChW * Q_ChW
    total_op_GHG = op_GHG_per_min * lifetime_minutes
    
    # Calculate total GHG (Objective 2)
    total_GHG = embedded_GHG + total_op_GHG
    
    return {
        'total_cost': total_cost,
        'total_GHG': total_GHG,
        'cap_cost': cap_cost,
        'op_cost': total_op_cost,
        'embedded_GHG': embedded_GHG,
        'op_GHG': total_op_GHG,
        'T_h_out': results['T_h_out'],
        'T_c_out': results['T_c_out'],
        'Q_exchanged': results['Q_exchanged'],
        'Q_ChW': Q_ChW,
        'Q_LPS': Q_LPS
    }

# Create arrays of heat exchanger areas to evaluate
areas = np.linspace(A_min, A_max, 100)

# Calculate objectives for each area
results = [calculate_objectives(A) for A in areas]

# Extract data for plotting
total_costs = [r['total_cost'] for r in results]
total_GHG = [r['total_GHG'] for r in results]
cap_costs = [r['cap_cost'] for r in results]
op_costs = [r['op_cost'] for r in results]
embedded_GHGs = [r['embedded_GHG'] for r in results]
op_GHGs = [r['op_GHG'] for r in results]
T_h_outs = [r['T_h_out'] for r in results]
T_c_outs = [r['T_c_out'] for r in results]
Q_exchanged = [r['Q_exchanged'] for r in results]
Q_ChWs = [r['Q_ChW'] for r in results]
Q_LPSs = [r['Q_LPS'] for r in results]

# Find the optimal areas
min_cost_idx = np.argmin(total_costs)
min_GHG_idx = np.argmin(total_GHG)

optimal_area_cost = areas[min_cost_idx]
optimal_area_GHG = areas[min_GHG_idx]

# Print the optimal solutions
print("\nOptimal solution for minimizing cost:")
print(f"Optimal heat exchanger area: {optimal_area_cost:.2f} m²")
print(f"Total cost: {total_costs[min_cost_idx]/1e6:.2f} million NOK")
print(f"Capital cost: {cap_costs[min_cost_idx]/1e6:.2f} million NOK")
print(f"Operating cost: {op_costs[min_cost_idx]/1e6:.2f} million NOK")
print(f"Hot stream outlet temperature: {T_h_outs[min_cost_idx]:.2f} °C")
print(f"Cold stream outlet temperature: {T_c_outs[min_cost_idx]:.2f} °C")
print(f"Heat exchanged: {Q_exchanged[min_cost_idx]:.2f} MJ/min")
print(f"Remaining ChW requirement: {Q_ChWs[min_cost_idx]:.2f} MJ/min")
print(f"Remaining LPS requirement: {Q_LPSs[min_cost_idx]:.2f} MJ/min")
print(f"Total GHG emissions: {total_GHG[min_cost_idx]/1e6:.2f} million kgCO₂e")

print("\nOptimal solution for minimizing GHG emissions:")
print(f"Optimal heat exchanger area: {optimal_area_GHG:.2f} m²")
print(f"Total cost: {total_costs[min_GHG_idx]/1e6:.2f} million NOK")
print(f"Total GHG emissions: {total_GHG[min_GHG_idx]/1e6:.2f} million kgCO₂e")
print(f"Embedded GHG: {embedded_GHGs[min_GHG_idx]/1e6:.2f} million kgCO₂e")
print(f"Operating GHG: {op_GHGs[min_GHG_idx]/1e6:.2f} million kgCO₂e")
print(f"Hot stream outlet temperature: {T_h_outs[min_GHG_idx]:.2f} °C")
print(f"Cold stream outlet temperature: {T_c_outs[min_GHG_idx]:.2f} °C")
print(f"Remaining ChW requirement: {Q_ChWs[min_GHG_idx]:.2f} MJ/min")
print(f"Remaining LPS requirement: {Q_LPSs[min_GHG_idx]:.2f} MJ/min")

# Calculate baseline costs and emissions
baseline_chw_cost = cost_ChW * q_h_baseline * lifetime_minutes
baseline_lps_cost = cost_LPS * q_c_baseline * lifetime_minutes
baseline_total_cost = baseline_chw_cost + baseline_lps_cost
baseline_chw_ghg = GHG_ChW * q_h_baseline * lifetime_minutes
baseline_lps_ghg = GHG_LPS * q_c_baseline * lifetime_minutes
baseline_total_ghg = baseline_chw_ghg + baseline_lps_ghg

print("\nBaseline (without heat integration):")
print(f"Total cost: {baseline_total_cost/1e6:.2f} million NOK")
print(f"Total GHG emissions: {baseline_total_ghg/1e6:.2f} million kgCO₂e")

# Calculate cost savings and GHG reduction
cost_savings = baseline_total_cost - total_costs[min_cost_idx]
ghg_reduction = baseline_total_ghg - total_GHG[min_GHG_idx]

print("\nComparison to baseline:")
print(f"Cost savings with cost optimization: {cost_savings/1e6:.2f} million NOK ({cost_savings/baseline_total_cost*100:.2f}%)")
print(f"GHG reduction with GHG optimization: {ghg_reduction/1e6:.2f} million kgCO₂e ({ghg_reduction/baseline_total_ghg*100:.2f}%)")

# Plotting
plt.figure(figsize=(16, 12))

# Plot 1: Total Cost vs Area
plt.subplot(2, 2, 1)
plt.plot(areas, [c/1e6 for c in total_costs], 'b-', linewidth=2)
plt.axvline(x=optimal_area_cost, color='r', linestyle='--')
plt.axhline(y=total_costs[min_cost_idx]/1e6, color='r', linestyle='--')
plt.scatter([optimal_area_cost], [total_costs[min_cost_idx]/1e6], color='r', s=100)
plt.text(optimal_area_cost+10, total_costs[min_cost_idx]/1e6, f'Optimal: {optimal_area_cost:.0f} m²', color='r', fontsize=12)
plt.title('Total Cost vs Heat Exchanger Area', fontsize=14)
plt.xlabel('Heat Exchanger Area (m²)', fontsize=12)
plt.ylabel('Total Cost (Million NOK)', fontsize=12)
plt.grid(True)

# Plot 2: Total GHG vs Area
plt.subplot(2, 2, 2)
plt.plot(areas, [g/1e6 for g in total_GHG], 'g-', linewidth=2)
plt.axvline(x=optimal_area_GHG, color='r', linestyle='--')
plt.axhline(y=total_GHG[min_GHG_idx]/1e6, color='r', linestyle='--')
plt.scatter([optimal_area_GHG], [total_GHG[min_GHG_idx]/1e6], color='r', s=100)
plt.text(optimal_area_GHG+10, total_GHG[min_GHG_idx]/1e6, f'Optimal: {optimal_area_GHG:.0f} m²', color='r', fontsize=12)
plt.title('Total GHG Emissions vs Heat Exchanger Area', fontsize=14)
plt.xlabel('Heat Exchanger Area (m²)', fontsize=12)
plt.ylabel('Total GHG Emissions (Million kgCO₂e)', fontsize=12)
plt.grid(True)

# Plot 3: Cost Breakdown
plt.subplot(2, 2, 3)
plt.plot(areas, [c/1e6 for c in cap_costs], 'b--', label='Capital Cost', linewidth=2)
plt.plot(areas, [c/1e6 for c in op_costs], 'b-.', label='Operating Cost', linewidth=2)
plt.plot(areas, [c/1e6 for c in total_costs], 'b-', label='Total Cost', linewidth=2)
plt.axvline(x=optimal_area_cost, color='r', linestyle='--')
plt.scatter([optimal_area_cost], [total_costs[min_cost_idx]/1e6], color='r', s=100)
plt.title('Cost Components vs Heat Exchanger Area', fontsize=14)
plt.xlabel('Heat Exchanger Area (m²)', fontsize=12)
plt.ylabel('Cost (Million NOK)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Plot 4: GHG Breakdown
plt.subplot(2, 2, 4)
plt.plot(areas, [g/1e6 for g in embedded_GHGs], 'g--', label='Embedded GHG', linewidth=2)
plt.plot(areas, [g/1e6 for g in op_GHGs], 'g-.', label='Operating GHG', linewidth=2)
plt.plot(areas, [g/1e6 for g in total_GHG], 'g-', label='Total GHG', linewidth=2)
plt.axvline(x=optimal_area_GHG, color='r', linestyle='--')
plt.scatter([optimal_area_GHG], [total_GHG[min_GHG_idx]/1e6], color='r', s=100)
plt.title('GHG Components vs Heat Exchanger Area', fontsize=14)
plt.xlabel('Heat Exchanger Area (m²)', fontsize=12)
plt.ylabel('GHG Emissions (Million kgCO₂e)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig('heat_exchanger_optimization.png')
plt.show()

# Main results for quick reference
print("\nSummary of Optimal Solutions:")
print(f"Economic optimum: {optimal_area_cost:.0f} m² heat exchanger")
print(f"Environmental optimum: {optimal_area_GHG:.0f} m² heat exchanger")