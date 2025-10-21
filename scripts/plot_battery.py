import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_battery_from_log(log_file='results/log/battery_log.csv'):
    try:
        # Load the data
        data = np.loadtxt(log_file, delimiter=',', skiprows=1)
        # Transpose to get timesteps x agents
        battery_history = data.T

        num_agents = battery_history.shape[0]

        # Plot
        for i in range(num_agents):
            plt.plot(battery_history[i], label=f'Agent {i+1}')

        plt.xlabel('Timestep')
        plt.ylabel('Battery Level')
        plt.title('Battery Levels Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.savefig('battery_plot_from_log.png', dpi=300, bbox_inches='tight')
        print("Battery plot saved as battery_plot_from_log.png")

        # Optionally show
        if len(sys.argv) > 1 and sys.argv[1] == '--show':
            plt.show()

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found. Run the heuristic first to generate the log.")
    except Exception as e:
        print(f"Error loading or plotting data: {e}")

if __name__ == "__main__":
    plot_battery_from_log()