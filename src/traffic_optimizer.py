import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define road segment structure
RoadSegment = namedtuple('RoadSegment', [
    'id', 'capacity', 'current_load', 'travel_time',
    'priority', 'emergency_lanes'
])


class TrafficPerformanceMonitor:
    """Monitors and tracks system performance metrics"""

    def __init__(self):
        self.metrics = {
            'congestion_levels': [],
            'travel_times': [],
            'bottlenecks': set(),
            'emergency_response': [],
            'utilization_history': []
        }

    def update_metrics(self, timestep_data: Dict):
        """
        Update performance metrics with current timestep data

        Args:
            timestep_data: Dictionary containing current timestep information
                {
                    'distribution': Dict[str, int],  # Current vehicle distribution
                    'total_in_network': int,         # Total vehicles in network
                    'utilization_percentage': float, # Current utilization
                    ...
                }
        """
        try:
            # Record utilization
            self.metrics['utilization_history'].append(
                timestep_data['utilization_percentage']
            )

            # Calculate and record congestion
            congestion = timestep_data['total_in_network'] / sum(
                timestep_data['distribution'].values()
            ) if timestep_data['distribution'].values() else 0
            self.metrics['congestion_levels'].append(congestion)

            # Identify bottlenecks
            self.identify_bottlenecks(timestep_data['distribution'])

            logging.debug(f"Updated metrics - Utilization: {timestep_data['utilization_percentage']:.1f}%, "
                          f"Congestion: {congestion:.2f}")

        except KeyError as e:
            logging.error(f"Missing required data in timestep_data: {e}")
            raise
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")
            raise

    def get_summary(self) -> Dict:
        """
        Get summary of performance metrics

        Returns:
            Dictionary containing summarized metrics
        """
        try:
            return {
                'avg_utilization': sum(self.metrics['utilization_history']) / len(self.metrics['utilization_history'])
                if self.metrics['utilization_history'] else 0,
                'max_congestion': max(self.metrics['congestion_levels'])
                if self.metrics['congestion_levels'] else 0,
                'bottleneck_count': len(self.metrics['bottlenecks']),
                'total_measurements': len(self.metrics['utilization_history'])
            }
        except Exception as e:
            logging.error(f"Error generating metrics summary: {e}")
            return {}

    def calculate_congestion(self, state: Dict) -> float:
        """Calculate current congestion level"""
        total_capacity = sum(road.capacity for road in state.values())
        total_load = sum(road.current_load for road in state.values())
        return total_load / total_capacity

    def identify_bottlenecks(self, distribution: Dict[str, int]):
        """
        Identify current bottlenecks in the network

        Args:
            distribution: Dictionary mapping road IDs to current vehicle counts
        """
        BOTTLENECK_THRESHOLD = 0.9
        try:
            for road_id, current_load in distribution.items():
                if isinstance(current_load, (int, float)) and current_load > 0:
                    self.metrics['bottlenecks'].add(road_id)
        except Exception as e:
            logging.error(f"Error identifying bottlenecks: {e}")


class ValidationSystem:
    """Validates traffic distribution and triggers adjustments"""

    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config['system']['adjustment_threshold']

    def validate_distribution(self, distribution: Dict, actual_traffic: Dict) -> bool:
        """Validate current distribution against actual traffic"""
        deviation = self.calculate_deviation(distribution, actual_traffic)
        return deviation <= self.threshold

    def calculate_deviation(self, distribution: Dict, actual: Dict) -> float:
        """Calculate deviation between expected and actual distribution"""
        total_deviation = 0
        for road_id in distribution:
            expected = distribution[road_id]
            actual_value = actual.get(road_id, 0)
            total_deviation += abs(expected - actual_value)
        return total_deviation / len(distribution)


class AdaptiveController:
    """Controls adaptive behavior of the system"""

    def __init__(self, config: Dict):
        self.config = config
        self.learning_rate = config['system']['learning_rate']
        self.historical_data = []

    def adjust_parameters(self, performance_metrics: Dict):
        """Adjust system parameters based on performance"""
        if performance_metrics['congestion'] > self.config['simulation']['peak_utilization']:
            self.update_flow_rates()
            self.rebalance_distribution()

    def update_flow_rates(self):
        """Update flow rates based on current performance"""
        pass

    def rebalance_distribution(self):
        """Rebalance traffic distribution"""
        pass


class TrafficOptimizer:
    """Main traffic optimization system"""

    def __init__(self, config: Dict):
        self.config = config
        self.road_network = {}
        self.monitor = TrafficPerformanceMonitor()
        self.validator = ValidationSystem(config)
        self.controller = AdaptiveController(config)
        self.simulation_history = []
        self.flow_rate_per_hour = config['simulation']['flow_rate_per_hour']
        logging.info("Traffic Optimizer initialized with configuration")

    def initialize_network(self):
        """Initialize road network from configuration"""
        try:
            self.road_network = {
                road_id: RoadSegment(
                    id=road_config['id'],
                    capacity=road_config['capacity'],
                    current_load=0,
                    travel_time=road_config['travel_time'],
                    priority=road_config['priority'],
                    emergency_lanes=road_config['emergency_lanes']
                )
                for road_id, road_config in self.config['road_network'].items()
            }
            logging.info(f"Road network initialized with {
                         len(self.road_network)} segments")
        except KeyError as e:
            logging.error(f"Missing required configuration key: {e}")
            raise

    def knapsack_optimization(self, vehicles: List[Dict], max_capacity: int) -> List[Dict]:
        """Enhanced knapsack optimization with priority handling"""
        logging.info(f"Starting knapsack optimization for {
                     len(vehicles)} vehicles")

        def get_priority_value(vehicle: Dict) -> float:
            """Calculate priority-adjusted value for vehicle"""
            try:
                priority_multipliers = self.config['vehicle_priorities']
                base_value = vehicle['value']
                # Default to private if type not specified
                vehicle_type = vehicle.get('type', 'private')
                multiplier = priority_multipliers.get(vehicle_type, 1.0)
                return base_value * multiplier
            except KeyError:
                logging.warning(
                    f"Using default value for vehicle {vehicle['id']}")
                return vehicle['value']

        try:
            n = len(vehicles)
            # Initialize DP table
            K = [[0 for _ in range(max_capacity + 1)] for _ in range(n + 1)]

            # Build solution table
            for i in range(n + 1):
                for w in range(max_capacity + 1):
                    if i == 0 or w == 0:
                        K[i][w] = 0
                    elif vehicles[i-1]['weight'] <= w:
                        priority_value = get_priority_value(vehicles[i-1])
                        K[i][w] = max(priority_value +
                                      K[i-1][w-vehicles[i-1]['weight']],
                                      K[i-1][w])
                    else:
                        K[i][w] = K[i-1][w]

            # Reconstruct and return solution
            solution = self.reconstruct_solution(K, vehicles, max_capacity)
            logging.info(f"Knapsack optimization completed successfully. "
                         f"Selected {len(solution)} vehicles")
            return solution

        except Exception as e:
            logging.error(f"Error in knapsack optimization: {str(e)}")
            raise

    def reconstruct_solution(self, K: List[List[int]], vehicles: List[Dict], max_capacity: int) -> List[Dict]:
        """
        Reconstruct the knapsack solution from the dynamic programming table.

        Args:
            K: Dynamic programming table
            vehicles: List of vehicles with weights and values
            max_capacity: Maximum capacity

        Returns:
            List of selected vehicles
        """
        logging.debug("Reconstructing knapsack solution")
        n = len(vehicles)
        w = max_capacity
        selected_vehicles = []

        # Traverse DP table to find selected items
        for i in range(n, 0, -1):
            if K[i][w] != K[i-1][w]:
                selected_vehicles.append(vehicles[i-1])
                w -= vehicles[i-1]['weight']
                logging.debug(f"Selected vehicle {vehicles[i-1]['id']} "
                              f"(Type: {vehicles[i-1]['type']}, "
                              f"Weight: {vehicles[i-1]['weight']}, "
                              f"Value: {vehicles[i-1]['value']})")

        logging.info(f"Selected {len(selected_vehicles)} vehicles "
                     f"with total weight {max_capacity - w}")
        return selected_vehicles

    def custom_traffic_distribution(self, total_vehicles: int) -> Dict[str, int]:
        """Enhanced traffic distribution with real-time adaptation"""

        def get_road_condition_factor(road_id: str) -> float:
            """Calculate road condition factor"""
            road = self.road_network[road_id]
            base_factor = 1.0

            # Adjust for road priority
            priority_factors = {
                'highway': 1.2,
                'interchange': 0.8
            }
            base_factor *= priority_factors.get(road.priority, 1.0)

            # Adjust for emergency lanes
            if road.emergency_lanes:
                base_factor *= 1.1

            return base_factor

        logging.info(f"Starting custom traffic distribution for {
                     total_vehicles} vehicles")
        total_capacity = sum(
            road.capacity for road in self.road_network.values())

        if total_vehicles > total_capacity:
            logging.warning(
                f"Input vehicles ({total_vehicles}) exceed network capacity ({total_capacity})")
            self.overflow_vehicles = total_vehicles - total_capacity
        else:
            self.overflow_vehicles = 0

        # Calculate adjusted capacities
        adjusted_capacities = {
            road_id: road.capacity * get_road_condition_factor(road_id)
            for road_id, road in self.road_network.items()
        }
        total_adjusted_capacity = sum(adjusted_capacities.values())

        # Distribute vehicles
        distribution = {}
        remaining = min(total_vehicles, total_capacity)

        for road_id, adjusted_capacity in adjusted_capacities.items():
            share = (adjusted_capacity / total_adjusted_capacity) * remaining
            distribution[road_id] = int(share)

        # Handle remaining vehicles
        remaining = remaining - sum(distribution.values())
        if remaining > 0:
            priority_roads = sorted(
                self.road_network.keys(),
                key=lambda x: self.road_network[x].priority == 'highway',
                reverse=True
            )

            for road_id in priority_roads:
                if remaining <= 0:
                    break
                space_left = self.road_network[road_id].capacity - \
                    distribution[road_id]
                add_amount = min(remaining, space_left)
                distribution[road_id] += add_amount
                remaining -= add_amount

        logging.info(f"Distributed {
                     sum(distribution.values())} vehicles across network")
        return distribution

    def simulate_traffic_flow(self, distribution: Dict[str, int], time_steps: int = 24) -> List[Dict]:
        """
        Enhanced traffic flow simulation with comprehensive monitoring and error handling.

        Args:
            distribution (Dict[str, int]): Initial distribution of vehicles across roads
            time_steps (int): Number of time steps to simulate

        Returns:
            List[Dict]: List of timestep data including distribution and metrics
        """
        logging.info(f"Starting traffic flow simulation for {
                     time_steps} time_steps")
        flow_data = []
        current_dist = distribution.copy()
        overflow_queue = getattr(self, 'overflow_vehicles', 0)

        def get_time_modifiers(hour: int) -> Tuple[float, float]:
            """Get time-based flow modifiers for the given hour"""
            try:
                for period_name, period_config in self.config['time_periods'].items():
                    if hour in period_config['hours']:
                        return period_config['inflow_mod'], period_config['outflow_mod']
                # Default modifiers if no period matches
                logging.warning(
                    f"No specific time period found for hour {hour}")
                return 1.0, 1.0
            except Exception as e:
                logging.error(f"Error getting time modifiers: {e}")
                return 1.0, 1.0

        for t in range(time_steps):
            try:
                # Get time-based modifiers
                inflow_mod, outflow_mod = get_time_modifiers(t % 24)
                total_capacity = sum(
                    road.capacity for road in self.road_network.values())

                # Process outflow (vehicles leaving the network)
                new_distribution = {}
                total_exits = 0
                available_capacity = 0

                # Calculate outflow for each road
                for road_id, road in self.road_network.items():
                    current_load = current_dist[road_id]
                    base_exit_rate = int(
                        road.capacity * self.config['simulation']['base_exit_rate'])
                    actual_exit = min(
                        int(base_exit_rate * outflow_mod), current_load)

                    # Update road load after exits
                    new_load = current_load - actual_exit
                    new_distribution[road_id] = new_load
                    total_exits += actual_exit

                    # Calculate available capacity for inflow
                    space = road.capacity - new_load
                    available_capacity += space

                # Process inflow from overflow queue
                if overflow_queue > 0 and available_capacity > 0:
                    # Determine target utilization based on time of day
                    hour = t % 24
                    if hour in self.config['time_periods']['morning_peak']['hours']:
                        target_util = self.config['simulation']['peak_utilization']
                    elif hour in self.config['time_periods']['night']['hours']:
                        target_util = self.config['simulation']['min_utilization']
                    else:
                        target_util = 0.7  # Normal hours target

                    # Calculate current utilization
                    current_util = sum(
                        new_distribution.values()) / total_capacity

                    # Add vehicles if below target utilization
                    if current_util < target_util:
                        space_for_target = int((target_util * total_capacity) -
                                               sum(new_distribution.values()))
                        vehicles_to_add = min(
                            space_for_target,
                            int(available_capacity * inflow_mod),
                            overflow_queue
                        )

                        # Distribute new vehicles proportionally to available capacity
                        if vehicles_to_add > 0:
                            for road_id, road in self.road_network.items():
                                if vehicles_to_add <= 0:
                                    break

                                space = road.capacity - \
                                    new_distribution[road_id]
                                if space > 0:
                                    # Calculate proportional share for this road
                                    add_amount = min(
                                        space,
                                        int(vehicles_to_add *
                                            (road.capacity / total_capacity))
                                    )
                                    new_distribution[road_id] += add_amount
                                    overflow_queue -= add_amount
                                    vehicles_to_add -= add_amount
                                    logging.debug(
                                        f"Added {add_amount} vehicles to {
                                            road_id} "
                                        f"(Space: {space}, Capacity: {
                                            road.capacity})"
                                    )

                # Update current distribution
                current_dist = new_distribution

                # Calculate metrics
                current_load = sum(current_dist.values())
                utilization = (current_load / total_capacity) * 100

                # Record timestep data
                timestep_data = {
                    'time_hour': t,
                    'distribution': current_dist.copy(),
                    'overflow_queue': overflow_queue,
                    'total_in_network': current_load,
                    'time_modifiers': (inflow_mod, outflow_mod),
                    'utilization_percentage': utilization,
                    'total_exits': total_exits,
                    'available_capacity': available_capacity,
                    'metrics': {
                        'road_utilization': {
                            road_id: (
                                load / self.road_network[road_id].capacity * 100)
                            for road_id, load in current_dist.items()
                        },
                        'network_load': current_load,
                        'network_capacity': total_capacity,
                        'queue_size': overflow_queue
                    }
                }

                # Store timestep data
                flow_data.append(timestep_data)

                # Update performance monitor
                self.monitor.update_metrics(timestep_data)

                # Log progress
                logging.info(
                    f"Hour {t}: {current_load:,} vehicles in network "
                    f"({utilization:.1f}% utilized), "
                    f"{overflow_queue:,} in queue "
                    f"(Flow modifiers: {inflow_mod:.1f}, {outflow_mod:.1f})"
                )

                # Additional debug logging
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    logging.debug(f"Road utilizations: " +
                                  ", ".join([f"{road_id}: {load/self.road_network[road_id].capacity*100:.1f}%"
                                             for road_id, load in current_dist.items()]))

            except KeyError as ke:
                logging.error(f"Missing configuration key at hour {t}: {ke}")
                raise
            except ValueError as ve:
                logging.error(
                    f"Invalid value in calculation at hour {t}: {ve}")
                raise
            except Exception as e:
                logging.error(
                    f"Unexpected error in simulation at hour {t}: {e}")
                raise

        # Generate final performance summary
        try:
            performance_summary = {
                'total_processed': flow_data[0]['overflow_queue'] - flow_data[-1]['overflow_queue'],
                'avg_utilization': sum(step['utilization_percentage'] for step in flow_data) / len(flow_data),
                'peak_utilization': max(step['utilization_percentage'] for step in flow_data),
                'min_utilization': min(step['utilization_percentage'] for step in flow_data),
                'final_queue': flow_data[-1]['overflow_queue']
            }
            logging.info("\nSimulation Summary:")
            for metric, value in performance_summary.items():
                if metric.endswith('utilization'):
                    logging.info(
                        f"- {metric.replace('_', ' ').title()}: {value:.1f}%")
                else:
                    logging.info(
                        f"- {metric.replace('_', ' ').title()}: {value:,}")

        except Exception as e:
            logging.error(f"Error generating performance summary: {e}")

        return flow_data

    def visualize_network(self, distribution: Dict[str, int], flow_data: List[Dict] = None):
        """Enhanced network visualization"""
        logging.info("Generating network visualization")

        viz_config = self.config['visualization']

        # Create figure with subplots
        fig = plt.figure(figsize=(
            viz_config['figure_size']['width'],
            viz_config['figure_size']['height']
        ))
        gs = plt.GridSpec(2, 2, figure=fig)

        # Network graph
        ax1 = fig.add_subplot(gs[0, :])
        G = nx.Graph()

        # Add nodes with enhanced information
        max_capacity = max(
            road.capacity for road in self.road_network.values())
        for road_id, load in distribution.items():
            road = self.road_network[road_id]
            capacity = road.capacity
            utilization = (load / capacity) * 100
            emergency_status = "🚑" if road.emergency_lanes else ""
            G.add_node(road_id,
                       load=load,
                       utilization=utilization,
                       priority=road.priority,
                       emergency=emergency_status)

        # Add edges with road type information
        edges = [
            ('mandela_north', 'interchange_1', {'type': 'highway'}),
            ('mandela_south', 'interchange_1', {'type': 'highway'}),
            ('portmore_east', 'interchange_2', {'type': 'highway'}),
            ('portmore_west', 'interchange_2', {'type': 'highway'}),
            ('interchange_1', 'interchange_2', {'type': 'connector'})
        ]
        G.add_edges_from(edges)

        # Draw network with enhanced styling
        pos = nx.spring_layout(G)

        # Color nodes based on utilization
        node_colors = [G.nodes[node]['utilization'] for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(G, pos,
                                       node_color=node_colors,
                                       node_size=viz_config['node_size'],
                                       cmap=plt.cm.get_cmap(viz_config['colormap']))

        # Draw edges with different styles based on type
        highway_edges = [(u, v) for (u, v, d) in G.edges(data=True)
                         if d['type'] == 'highway']
        connector_edges = [(u, v) for (u, v, d) in G.edges(data=True)
                           if d['type'] == 'connector']

        nx.draw_networkx_edges(G, pos, edgelist=highway_edges,
                               width=2, edge_color='blue')
        nx.draw_networkx_edges(G, pos, edgelist=connector_edges,
                               width=1.5, edge_color='green', style='dashed')

        # Add detailed labels
        labels = {
            node: f"{node}{G.nodes[node]['emergency']}\n"
            f"{distribution[node]}/{self.road_network[node].capacity}\n"
            f"{G.nodes[node]['utilization']:.1f}%\n"
            f"({self.road_network[node].priority})"
            for node in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, labels,
                                font_size=viz_config['font_size'])

        plt.colorbar(nodes, label='Utilization %')
        ax1.set_title('Traffic Distribution Network')
        ax1.axis('off')

        if flow_data:
            # Traffic Load Analysis
            ax2 = fig.add_subplot(gs[1, 0])
            times = [d['time_hour'] for d in flow_data]
            loads = [d['total_in_network'] for d in flow_data]
            queues = [d['overflow_queue'] for d in flow_data]

            ax2.plot(times, loads, 'b-', label='Vehicles in Network')
            ax2.plot(times, queues, 'r--', label='Queue Size')
            ax2.fill_between(times, loads, alpha=0.3, color='blue')

            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Number of Vehicles')
            ax2.set_title('Network Load and Queue Size Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Road Utilization Analysis
            ax3 = fig.add_subplot(gs[1, 1])
            for road_id in self.road_network:
                utilization = [
                    d['distribution'][road_id] /
                    self.road_network[road_id].capacity * 100
                    for d in flow_data
                ]
                ax3.plot(times, utilization,
                         label=f"{road_id} ({self.road_network[road_id].priority})")

            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Utilization (%)')
            ax3.set_title('Road Utilization Over Time')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)

            # Add peak hours highlighting
            for ax in [ax2, ax3]:
                # Morning peak
                ax.axvspan(6, 9, alpha=0.2, color='yellow',
                           label='Morning Peak')
                # Evening peak
                ax.axvspan(16, 19, alpha=0.2, color='orange',
                           label='Evening Peak')

        plt.tight_layout()
        return plt
