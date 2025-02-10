import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_composite_curves(streams, delta_Tmin=0):
    """
    Calculate hot and cold composite curves given stream data and delta_Tmin.
    """
    temperature_points = sorted(set(streams['Inlet T'].tolist() + streams['Final T'].tolist()), reverse=True)
    
    hot_composite = []
    cold_composite = []
    
    for i in range(len(temperature_points) - 1):
        T1, T2 = temperature_points[i], temperature_points[i+1]
        
        hot_duty = sum(
            stream['mCp'] * (T1 - T2) for _, stream in streams.iterrows()
            if stream['Stream Type'] == 'Hot' and stream['Inlet T'] >= T1 and stream['Final T'] <= T2
        )
        
        cold_duty = sum(
            stream['mCp'] * (T1 - T2) for _, stream in streams.iterrows()
            if stream['Stream Type'] == 'Cold' and stream['Final T'] >= T1 and stream['Inlet T'] <= T2
        )
        
        hot_composite.append((T2 + delta_Tmin / 2, hot_duty))
        cold_composite.append((T2 - delta_Tmin / 2, cold_duty))
    
    return hot_composite, cold_composite

def plot_composite_curves(hot_composite, cold_composite, delta_Tmin):
    """
    Plot the composite curves.
    """
    hot_temps, hot_duties = zip(*hot_composite)
    cold_temps, cold_duties = zip(*cold_composite)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(hot_duties), hot_temps, label='Hot Composite', color='red')
    plt.plot(np.cumsum(cold_duties), cold_temps, label='Cold Composite', color='blue')
    plt.xlabel('Cumulative Heat Duty (kW)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Composite Curves (ΔTmin={delta_Tmin}°C)')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Define stream data based on PEP03 Table 1
    streams_data = {
        'Stream': ['H1', 'H2', 'C1', 'C2'],
        'mCp': [2.5, 3.0, 4.0, 2.6],  # kW/°C
        'Inlet T': [250, 180, 50, 40],
        'Final T': [150, 100, 220, 150],
        'Stream Type': ['Hot', 'Hot', 'Cold', 'Cold']
    }
    
    streams_df = pd.DataFrame(streams_data)
    
    # Solve for ΔTmin = 0, 10, 20
    for delta_Tmin in [0, 10, 20]:
        hot_composite, cold_composite = calculate_composite_curves(streams_df, delta_Tmin)
        plot_composite_curves(hot_composite, cold_composite, delta_Tmin)

if __name__ == "__main__":
    main()
