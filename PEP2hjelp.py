import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
T1 = 30  # Feed temperature [°C]
TBP = 73.5  # Bubble point temperature [°C]
F1 = 20  # Feed flow rate [kmol/min]
x1_ace = 0.5  # Acetone mole fraction in feed
x1_tol = 0.5  # Toluene mole fraction in feed
x5_ace = 0.99  # Acetone mole fraction in distillate
x5_tol = 0.01  # Toluene mole fraction in distillate
x6_ace = 0.01  # Acetone mole fraction in bottoms
x6_tol = 0.99  # Toluene mole fraction in bottoms
C_LPS = 0.039  # Cost of low-pressure steam [NOK/MJ]
C_CW = 0.010  # Cost of cooling water [NOK/MJ]
lifetime = 8000 * 60  # Project lifetime [min] (8000 hours converted to minutes)

def evaluate_system(T2):
    """
    Evaluates the system performance for a given preheating temperature T2
    Returns the cost per kmol of toluene produced
    """
    # Calculate feed enthalpy
    h1 = 0.1463 * T1 - 1.6939
    
    # Calculate enthalpy after preheating
    if T2 < TBP:
        h2 = 0.1463 * T2 - 1.6939
        # Below bubble point - no vapor phase
        F3 = F1
        F4 = 0
        x3_ace = x1_ace
        x3_tol = x1_tol
    else:
        # At or above bubble point - vapor-liquid equilibrium
        h2 = -0.0631 * T2**2 + 14.863 * T2 - 743
        
        # Calculate vapor and liquid compositions using VLE relationships
        y4_ace = -2.51e-4 * T2**2 + 2.43e-2 * T2 + 0.42
        y4_tol = 1 - y4_ace
        x3_ace = 2.82e-4 * T2**2 - 6.45e-2 * T2 + 3.72
        x3_tol = 1 - x3_ace
        
        # Solve for vapor fraction using material balance
        # F1*x1_ace = F3*x3_ace + F4*y4_ace and F1 = F3 + F4
        # Rearranging: F4/F1 = (x1_ace - x3_ace)/(y4_ace - x3_ace)
        vapor_fraction = (x1_ace - x3_ace)/(y4_ace - x3_ace)
        
        # Check if vapor fraction is physically valid
        vapor_fraction = max(0, min(1, vapor_fraction))
        
        F4 = F1 * vapor_fraction
        F3 = F1 - F4
    
    # Calculate heat duty for preheater
    Qhx = F1 * (h2 - h1)
    
    # Calculate reboiler and condenser duties
    Qreb = F3 * (-0.12 * T2 - 20.9 * x3_tol + 44)
    Qcond = F3 * (0.0268 * T2 - 28.5 * x3_tol + 35)
    
    # Calculate capital cost
    Ccap = 100000000 + 20000000 * F3
    
    # Calculate operating cost over lifetime
    Cop = lifetime * (Qhx * C_LPS + Qreb * C_LPS + Qcond * C_CW)
    
    # Calculate toluene production from column mass balance
    # F3*x3_ace = F5*x5_ace + F6*x6_ace
    # F3*x3_tol = F5*x5_tol + F6*x6_tol
    # F3 = F5 + F6
    
    # Solving these equations:
    F6 = F3 * (x3_ace - x5_ace) / (x6_ace - x5_ace)
    F5 = F3 - F6
    
    # Total toluene production
    Mtol = lifetime * F6 * x6_tol
    
    # Calculate objective function
    J = (Ccap + Cop) / Mtol
    
    result = {
        'T2': T2,
        'F3': F3,
        'F4': F4,
        'F5': F5,
        'F6': F6,
        'x3_ace': x3_ace,
        'x3_tol': x3_tol,
        'Qhx': Qhx,
        'Qreb': Qreb,
        'Qcond': Qcond,
        'Ccap': Ccap,
        'Cop': Cop,
        'Mtol': Mtol,
        'J': J
    }
    
    return J, result

def objective_function(T2):
    """Wrapper function for optimization"""
    J, _ = evaluate_system(T2)
    return J

# Function to perform optimization and generate the single plot
def optimize_and_plot():
    # Create arrays to evaluate the system at different temperatures
    T2_range = np.linspace(30, 93, 200)
    results = []
    J_values = []
    
    for T2 in T2_range:
        J, result = evaluate_system(T2)
        results.append(result)
        J_values.append(J)
    
    # Find minimum using scipy optimizer
    result = minimize_scalar(objective_function, bounds=(30, 93), method='bounded')
    optimal_T2 = result.x
    min_cost = result.fun
    
    # Get detailed results at optimal point
    _, optimal_details = evaluate_system(optimal_T2)
    
    # Print optimization results
    print("=== Optimization Results ===")
    print(f"Optimal preheating temperature: {optimal_T2:.2f}°C")
    print(f"Minimum cost per kmol of toluene: {min_cost:.2f} NOK/kmol")
    print(f"Flash vapor flow rate: {optimal_details['F4']:.2f} kmol/min")
    print(f"Column feed flow rate: {optimal_details['F3']:.2f} kmol/min")
    print(f"Toluene product flow rate: {optimal_details['F6']:.2f} kmol/min")
    print(f"Acetone mole fraction in column feed: {optimal_details['x3_ace']:.4f}")
    print(f"Preheater duty: {optimal_details['Qhx']:.2f} MJ/min")
    print(f"Reboiler duty: {optimal_details['Qreb']:.2f} MJ/min")
    print(f"Condenser duty: {optimal_details['Qcond']:.2f} MJ/min")
    print(f"Capital cost: {optimal_details['Ccap']/1e6:.2f} million NOK")
    print(f"Operating cost: {optimal_details['Cop']/1e6:.2f} million NOK")
    print(f"Total toluene production: {optimal_details['Mtol']/1e6:.2f} million kmol")
    
    # Create a single plot
    plt.figure(figsize=(10, 6))
    
    # Plot the objective function vs T2
    plt.plot(T2_range, J_values, 'b-', linewidth=2)
    
    # Add vertical lines for bubble point and optimal temperature
    plt.axvline(x=TBP, color='r', linestyle='--', label='Bubble Point')
    plt.axvline(x=optimal_T2, color='g', linestyle='-', label=f'Optimal T2: {optimal_T2:.2f}°C')
    
    # Add a point to mark the minimum cost location
    plt.plot(optimal_T2, min_cost, 'ro', markersize=10, label=f'Minimum: {min_cost:.2f} NOK/kmol')
    
    # Add labels and title
    plt.xlabel('Preheating Temperature T2 (°C)', fontsize=12)
    plt.ylabel('Cost per kmol Toluene (NOK/kmol)', fontsize=12)
    plt.title('Objective Function vs Preheating Temperature', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add annotations for the minimum point
    plt.annotate(f'({optimal_T2:.2f}°C, {min_cost:.2f} NOK/kmol)',
                xy=(optimal_T2, min_cost),
                xytext=(optimal_T2+5, min_cost+10000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return optimal_T2, min_cost, optimal_details

# Run the optimization and generate the plot
if __name__ == "__main__":
    optimal_T2, min_cost, optimal_details = optimize_and_plot()