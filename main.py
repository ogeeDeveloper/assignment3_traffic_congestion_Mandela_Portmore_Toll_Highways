import argparse
import yaml
import sys
from src.traffic_optimizer import TrafficOptimizer
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Traffic Optimization System')
    parser.add_argument('--vehicles', type=int,
                        help='Number of vehicles to simulate')
    parser.add_argument('--time-steps', type=int,
                        help='Number of time steps to simulate')
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='Path to configuration file')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Initialize optimizer with configuration
    optimizer = TrafficOptimizer(config)
    optimizer.initialize_network()

    # Get parameters from args or config
    total_vehicles = args.vehicles or config['simulation']['default_vehicles']
    time_steps = args.time_steps or config['simulation']['default_time_steps']

    # Run Knapsack optimization
    vehicles = config['optimization']['knapsack_sample_vehicles']
    knapsack_result = optimizer.knapsack_optimization(
        vehicles, max_capacity=2000)
    print("Knapsack Optimization Result:", knapsack_result)

    # Run custom distribution algorithm
    custom_distribution = optimizer.custom_traffic_distribution(total_vehicles)
    print("\nCustom Distribution Result:", custom_distribution)

    # Simulate traffic flow
    flow_simulation = optimizer.simulate_traffic_flow(
        custom_distribution, time_steps)
    print("\nFlow Simulation First Timestep:", flow_simulation[0])

    # Visualize results
    plt = optimizer.visualize_network(custom_distribution)
    plt.show()

    return {
        'knapsack_result': knapsack_result,
        'custom_distribution': custom_distribution,
        'flow_simulation': flow_simulation
    }


if __name__ == "__main__":
    results = main()
