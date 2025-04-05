import numpy as np
import matplotlib.pyplot as plt

# Data from the problem
hot_curve_data = [
    [500, 50],
    [325, 45],
    [300, 40],
    [100, 30],
    [100, 15],
    [75, 10],
    [35, 5],
    [30, 0]
]

cold_curve_data = [
    [420, 60],
    [300, 57],
    [275, 55],
    [250, 52],
    [200, 50],
    [120, 45],
    [110, 35],
    [90, 30],
    [75, 25],
    [40, 20],
    [25, 15]
]

hot_curve = np.array(hot_curve_data)
cold_curve = np.array(cold_curve_data)

# Plot the original composite curves
plt.figure(figsize=(10, 6))
plt.plot(hot_curve[:, 1], hot_curve[:, 0], 'r-o', label='Hot Curve')
plt.plot(cold_curve[:, 1], cold_curve[:, 0], 'b-o', label='Cold Curve')
plt.xlabel('Enthalpy (MW)')
plt.ylabel('Temperature (°C)')
plt.title('Original Composite Curves')
plt.grid(True)
plt.legend()
plt.savefig('original_composite_curves.png')

# Calculate utility requirements
hot_utility = cold_curve[0, 1] - hot_curve[0, 1]  # MW
cold_utility = cold_curve[-1, 1] - hot_curve[-1, 1]  # MW

print(f"Hot utility required: {hot_utility:.2f} MW")
print(f"Cold utility required: {cold_utility:.2f} MW")

# Finding suitable utilities
# Determine minimum hot utility temperature needed
hot_utility_min_temp = cold_curve[0, 0] + 10  # Adding ΔTmin
print(f"Minimum hot utility temperature needed: {hot_utility_min_temp:.2f}°C")

# Determine minimum cold utility temperature needed
cold_utility_max_temp = hot_curve[-1, 0] - 10  # Subtracting ΔTmin
print(f"Maximum cold utility temperature needed: {cold_utility_max_temp:.2f}°C")

# Select reasonable utilities based on the requirements
# For hot utility: High pressure steam at 430°C (above minimum needed)
# For cold utility: Cooling water at 20°C (below maximum allowed)

hot_utility_temp = 430  # °C
cold_utility_temp = 20  # °C

print(f"Selected hot utility: Steam at {hot_utility_temp}°C for {hot_utility:.2f} MW")
print(f"Selected cold utility: Cooling water at {cold_utility_temp}°C for {cold_utility:.2f} MW")

# Create balanced composite curves with utilities
# For the hot utility, we add points to extend the hot curve
# For the cold utility, we add points to extend the cold curve

# Create extended hot curve with utilities
balanced_hot_curve = np.vstack([
    [hot_utility_temp, 0],  # Hot utility start
    [hot_utility_temp, hot_utility],  # Hot utility end
    hot_curve  # Original hot curve
])

# Create extended cold curve with utilities
extended_cold_enthalpy = cold_curve[-1, 1] + cold_utility
balanced_cold_curve = np.vstack([
    cold_curve,  # Original cold curve
    [cold_utility_temp, extended_cold_enthalpy]  # Cold utility end
])

# Plot the balanced composite curves
plt.figure(figsize=(10, 6))
plt.plot(balanced_hot_curve[:, 1], balanced_hot_curve[:, 0], 'r-o', label='Hot Curve with Utilities')
plt.plot(balanced_cold_curve[:, 1], balanced_cold_curve[:, 0], 'b-o', label='Cold Curve with Utilities')
plt.xlabel('Enthalpy (MW)')
plt.ylabel('Temperature (°C)')
plt.title('Balanced Composite Curves with Utilities')
plt.grid(True)
plt.legend()
plt.savefig('balanced_composite_curves.png')

# Calculate the exergy of the utilities relative to ambient temperature (25°C)
T0 = 25 + 273.15  # Ambient temperature in Kelvin
hot_utility_temp_K = hot_utility_temp + 273.15  # K
cold_utility_temp_K = cold_utility_temp + 273.15  # K

# Calculate Carnot factors
hot_carnot = 1 - T0/hot_utility_temp_K
cold_carnot = T0/cold_utility_temp_K - 1

# Calculate exergy of utilities
hot_exergy = hot_utility * hot_carnot
cold_exergy = cold_utility * (-1) * cold_carnot  # Cold utility exergy is typically negative

print(f"Hot utility Carnot factor: {hot_carnot:.4f}")
print(f"Cold utility Carnot factor: {cold_carnot:.4f}")
print(f"Hot utility exergy: {hot_exergy:.2f} MW")
print(f"Cold utility exergy: {cold_exergy:.2f} MW")

# Convert to annual costs using the utility cost estimation procedure
# Assuming 8000 hours of operation per year
hours_per_year = 8000

# Base utility costs from Lecture 19 (estimated)
base_cost_per_GJ = 32  # USD/GJ for exergy

# Convert MW to GJ/hr (1 MW = 3.6 GJ/hr)
hot_exergy_GJ_hr = hot_exergy * 3.6
cold_exergy_GJ_hr = abs(cold_exergy) * 3.6  # Using absolute value for cost calculation

# Calculate annual costs
hot_annual_cost = hot_exergy_GJ_hr * base_cost_per_GJ * hours_per_year
cold_annual_cost = cold_exergy_GJ_hr * base_cost_per_GJ * hours_per_year
total_annual_cost = hot_annual_cost + cold_annual_cost

print(f"Hot utility annual cost: {hot_annual_cost:,.2f} USD/year")
print(f"Cold utility annual cost: {cold_annual_cost:,.2f} USD/year")
print(f"Total utility annual cost: {total_annual_cost:,.2f} USD/year")

# Create a summary table of results
print("\nSUMMARY OF RESULTS (BASE CASE):")
print("-" * 50)
print(f"{'Utility':<20} {'Temperature (°C)':<20} {'Amount (MW)':<15} {'Cost (USD/year)':<20}")
print("-" * 50)
print(f"{'Hot Utility (Steam)':<20} {hot_utility_temp:<20.2f} {hot_utility:<15.2f} {hot_annual_cost:<20,.2f}")
print(f"{'Cold Utility (Water)':<20} {cold_utility_temp:<20.2f} {cold_utility:<15.2f} {cold_annual_cost:<20,.2f}")
print("-" * 50)
print(f"{'Total':<20} {'':<20} {hot_utility + cold_utility:<15.2f} {total_annual_cost:<20,.2f}")