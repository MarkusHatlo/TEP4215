#import numpy as np
import matplotlib.pyplot as plt


hot_temps = [50, 60, 70, 90]
hot_q = [0, 150, 233, 500]


cold_temps = [10, 20, 30, 40, 60, 100]  
cold_q = [0, 100, 225, 250, 333, 400]  


plt.figure(figsize=(8,8))

plt.plot(hot_q, hot_temps, marker='o', linestyle='-', color="purple", label="Hot Composite Curve")
plt.plot(cold_q, cold_temps, marker='o', linestyle='-', color="blue", label="Cold Composite Curve")
plt.plot(1000,100,color="white")


plt.xlabel("Q [MJ/min]")
plt.ylabel("Temp [°C]")
plt.title("Composite Curves")
plt.legend()
plt.locator_params(axis='both', nbins=15)  # Increase the number of bins (more grid lines)
plt.grid()
#plt.minorticks_on()  # Enable minor ticks
#plt.grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor grid (dotted, thinner)
plt.show()
