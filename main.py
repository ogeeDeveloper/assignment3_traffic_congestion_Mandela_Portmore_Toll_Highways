import argparse
import yaml
import sys
from src.traffic_optimizer import TrafficOptimizer
from pathlib import Path
import logging


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            validate_config(config)
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def validate_config(config: dict) -> None:
    """Validate configuration structure"""
    required_sections = [
        'system',
        'simulation',
        'road_network',
        'vehicle_priorities',
        'time_periods',
        'visualization'
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}")

    # Validate time periods
    if 'time_periods' in config:
        required_periods = ['morning_peak', 'evening_peak', 'night', 'normal']
        required_period_fields = ['hours', 'inflow_mod', 'outflow_mod']

        for period in required_periods:
            if period not in config['time_periods']:
                raise ValueError(f"Missing required time period: {period}")

            period_config = config['time_periods'][period]
            for field in required_period_fields:
                if field not in period_config:
                    raise ValueError(f"Missing {field} in {
                                     period} time period configuration")

    # Validate simulation parameters
    required_sim_params = [
        'flow_rate_per_hour',
        'min_utilization',
        'peak_utilization',
        'base_exit_rate'
    ]
    for param in required_sim_params:
        if param not in config['simulation']:
            raise ValueError(f"Missing '{param}' in simulation configuration")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Traffic Optimization System for Mandela and Portmore Toll Highways'
    )
    parser.add_argument(
        '--vehicles',
        type=int,
        help='Number of vehicles to simulate'
    )
    parser.add_argument(
        '--time-steps',
        type=int,
        help='Number of time steps to simulate'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args()


def create_sample_vehicles(config: dict) -> list:
    """
    Create a list of sample vehicles with different types and priorities.

    This function generates a predefined list of vehicles with varying attributes
    such as type, weight, and value. These sample vehicles are used for
    demonstration or testing purposes in the traffic optimization system.

    Parameters:
    config (dict): A dictionary containing the configuration settings for the
                   traffic optimization system. This parameter is currently
                   unused in the function but may be utilized in future
                   implementations to customize the sample vehicles.

    Returns:
    list: A list of dictionaries, where each dictionary represents a vehicle
          with the following keys:
          - 'id' (int): A unique identifier for the vehicle.
          - 'type' (str): The type of vehicle (e.g., 'emergency', 'public_transport').
          - 'weight' (int): The weight assigned to the vehicle, representing its
                            impact on traffic.
          - 'value' (int): The priority value of the vehicle in the system.

    """
    return [
        {
            'id': 1,
            'type': 'emergency',
            'weight': 1,
            'value': 5
        },
        {
            'id': 2,
            'type': 'public_transport',
            'weight': 2,
            'value': 4
        },
        {
            'id': 3,
            'type': 'commercial',
            'weight': 2,
            'value': 3
        },
        {
            'id': 4,
            'type': 'private',
            'weight': 1,
            'value': 2
        }
    ]


def main():
    try:
        # Parse arguments and setup logging
        args = parse_arguments()
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug logging enabled")

        # Load and validate configuration
        logging.info("Loading configuration from: %s", args.config)
        config = load_config(args.config)
        logging.info("Configuration loaded and validated successfully")

        # Initialize optimizer
        logging.info("Initializing traffic optimizer...")
        optimizer = TrafficOptimizer(config)
        optimizer.initialize_network()

        # Get simulation parameters
        total_vehicles = args.vehicles or config['simulation']['default_vehicles']
        time_steps = args.time_steps or config['simulation']['default_time_steps']
        logging.info(f"Running simulation with {
                     total_vehicles} vehicles for {time_steps} time steps")

        # Create and run knapsack optimization
        logging.info("Starting knapsack optimization...")
        vehicles = [
            {
                'id': 1,
                'type': 'emergency',
                'weight': 1,
                'value': 5
            },
            {
                'id': 2,
                'type': 'public_transport',
                'weight': 2,
                'value': 4
            },
            {
                'id': 3,
                'type': 'commercial',
                'weight': 2,
                'value': 3
            },
            {
                'id': 4,
                'type': 'private',
                'weight': 1,
                'value': 2
            }
        ]

        knapsack_result = optimizer.knapsack_optimization(
            vehicles, max_capacity=2000)

        # Print knapsack results
        print("\nKnapsack Optimization Result:")
        total_value = 0
        total_weight = 0
        for vehicle in knapsack_result:
            print(f"- Vehicle {vehicle['id']} (Type: {vehicle['type']}, "
                  f"Weight: {vehicle['weight']}, Value: {vehicle['value']})")
            total_value += vehicle['value']
            total_weight += vehicle['weight']
        print(f"Total Value: {total_value}, Total Weight: {total_weight}")

        # Run traffic distribution
        logging.info("Starting traffic distribution...")
        custom_distribution = optimizer.custom_traffic_distribution(
            total_vehicles)

        # Print distribution results
        print("\nRoad Network Distribution:")
        total_distributed = 0
        for road_id, count in custom_distribution.items():
            road = optimizer.road_network[road_id]
            utilization = (count / road.capacity) * 100
            total_distributed += count
            print(f"- {road_id}: {count:,} vehicles "
                  f"({utilization:.1f}% utilized) "
                  f"[Priority: {road.priority}]"
                  f"{' 🚑' if road.emergency_lanes else ''}")
        print(f"Total Distributed: {total_distributed:,} vehicles")

        # Run traffic flow simulation
        logging.info("Starting traffic flow simulation...")
        flow_simulation = optimizer.simulate_traffic_flow(
            custom_distribution, time_steps)

        performance_summary = optimizer.monitor.get_summary()

        print("\nPerformance Metrics:")
        print(
            f"- Average Network Utilization: {performance_summary['avg_utilization']:.1f}%")
        print(
            f"- Maximum Congestion Level: {performance_summary['max_congestion']:.2f}")
        print(
            f"- Number of Bottlenecks: {performance_summary['bottleneck_count']}")

        # Calculate and print simulation statistics
        initial_queue = flow_simulation[0]['overflow_queue']
        final_queue = flow_simulation[-1]['overflow_queue']
        vehicles_processed = initial_queue - final_queue
        avg_utilization = sum(step['utilization_percentage']
                              for step in flow_simulation) / len(flow_simulation)
        peak_utilization = max(step['utilization_percentage']
                               for step in flow_simulation)
        min_utilization = min(step['utilization_percentage']
                              for step in flow_simulation)

        print("\nSimulation Results:")
        print(f"- Initial queue size: {initial_queue:,}")
        print(f"- Final queue size: {final_queue:,}")
        print(f"- Vehicles processed: {vehicles_processed:,}")
        print(f"- Average network utilization: {avg_utilization:.1f}%")
        print(f"- Peak utilization: {peak_utilization:.1f}%")
        print(f"- Minimum utilization: {min_utilization:.1f}%")

        # Generate time period statistics
        print("\nTime Period Analysis:")
        for period_name, period_data in config['time_periods'].items():
            period_steps = [
                step for step in flow_simulation
                if step['time_hour'] % 24 in period_data['hours']
            ]
            if period_steps:
                avg_period_util = sum(step['utilization_percentage']
                                      for step in period_steps) / len(period_steps)
                print(f"- {period_name.replace('_', ' ').title()}: "
                      f"{avg_period_util:.1f}% average utilization")

        # Visualize results
        logging.info("Generating visualization...")
        plt = optimizer.visualize_network(custom_distribution, flow_simulation)
        plt.show()

        # Return complete results
        results = {
            'knapsack_result': {
                'selected_vehicles': knapsack_result,
                'total_value': total_value,
                'total_weight': total_weight
            },
            'distribution': {
                'allocation': custom_distribution,
                'total_distributed': total_distributed
            },
            'simulation': {
                'initial_queue': initial_queue,
                'final_queue': final_queue,
                'vehicles_processed': vehicles_processed,
                'statistics': {
                    'avg_utilization': avg_utilization,
                    'peak_utilization': peak_utilization,
                    'min_utilization': min_utilization
                }
            },
            'flow_data': flow_simulation
        }

        logging.info("Simulation completed successfully")
        return results

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        sys.exit(1)
    except KeyError as ke:
        logging.error(f"Missing configuration key: {str(ke)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        logging.debug("Error details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        results = main()
        logging.info("Simulation completed successfully")
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        sys.exit(1)
