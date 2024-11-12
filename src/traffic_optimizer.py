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
RoadSegment = namedtuple(
    'RoadSegment', ['id', 'capacity', 'current_load', 'travel_time'])


class TrafficOptimizer:
    def __init__(self, config: dict):
        self.config = config
        self.road_network = {}
        self.total_vehicles = 0
        self.simulation_history = []
        self.flow_rate_per_hour = config.get('simulation', {}).get(
            'flow_rate_per_hour', 0.25)  # 25% of capacity flows per hour
        logging.info("Traffic Optimizer initialized with configuration")

    def initialize_network(self):
        """Initialize the road network from configuration"""
        try:
            self.road_network = {
                road_id: RoadSegment(
                    id=road_config['id'],
                    capacity=road_config['capacity'],
                    current_load=0,
                    travel_time=road_config['travel_time']
                )
                for road_id, road_config in self.config['road_network'].items()
            }
            logging.info(f"Road network initialized with {
                         len(self.road_network)} segments")
        except KeyError as e:
            logging.error(f"Missing required configuration key: {e}")
            raise

    def custom_traffic_distribution(self, total_vehicles: int) -> Dict[str, int]:
        """
        Enhanced custom dynamic programming algorithm for traffic distribution.
        """
        logging.info(f"Starting custom traffic distribution for {
                     total_vehicles} vehicles")
        roads = list(self.road_network.keys())

        # Calculate total network capacity
        total_capacity = sum(
            road.capacity for road in self.road_network.values())

        if total_vehicles > total_capacity:
            logging.warning(
                f"Input vehicles ({total_vehicles}) exceed network capacity ({total_capacity})")
            self.overflow_vehicles = total_vehicles - total_capacity
        else:
            self.overflow_vehicles = 0

        # Distribute vehicles proportionally to road capacity
        distribution = {}
        remaining_vehicles = min(total_vehicles, total_capacity)

        # First pass: proportional distribution
        for road_id, road in self.road_network.items():
            share = (road.capacity / total_capacity) * remaining_vehicles
            distribution[road_id] = int(share)

        # Second pass: optimize distribution considering road types
        # Main highways get priority for remaining capacity
        remaining = remaining_vehicles - sum(distribution.values())
        if remaining > 0:
            priority_roads = ['mandela_north', 'mandela_south',
                              'portmore_east', 'portmore_west']
            for road_id in priority_roads:
                if road_id in self.road_network:
                    space_left = self.road_network[road_id].capacity - \
                        distribution[road_id]
                    if space_left > 0:
                        add_amount = min(remaining, space_left)
                        distribution[road_id] += add_amount
                        remaining -= add_amount
                        if remaining == 0:
                            break

        # Final pass: distribute any remaining vehicles to interchanges
        if remaining > 0:
            interchange_roads = [r for r in roads if 'interchange' in r]
            for road_id in interchange_roads:
                space_left = self.road_network[road_id].capacity - \
                    distribution[road_id]
                if space_left > 0:
                    add_amount = min(remaining, space_left)
                    distribution[road_id] += add_amount
                    remaining -= add_amount
                    if remaining == 0:
                        break

        logging.info(f"Distributed {
                     sum(distribution.values())} vehicles across network")
        if self.overflow_vehicles > 0:
            logging.warning(
                f"{self.overflow_vehicles} vehicles in overflow queue")

        return distribution

    def knapsack_optimization(self, vehicles: List[Dict], max_capacity: int) -> List[Dict]:
        """
        Implement the Knapsack algorithm for vehicle distribution.
        Args:
            vehicles: List of vehicles with weights (size/impact on traffic)
            max_capacity: Maximum road capacity
        Returns:
            Optimal distribution of vehicles
        """
        logging.info(f"Starting knapsack optimization for {
                     len(vehicles)} vehicles")
        n = len(vehicles)
        K = [[0 for _ in range(max_capacity + 1)] for _ in range(n + 1)]

        # Build table K[][] in bottom up manner
        for i in range(n + 1):
            for w in range(max_capacity + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif vehicles[i-1]['weight'] <= w:
                    K[i][w] = max(vehicles[i-1]['value'] +
                                  K[i-1][w-vehicles[i-1]['weight']],
                                  K[i-1][w])
                else:
                    K[i][w] = K[i-1][w]

        return self.reconstruct_solution(K, vehicles, max_capacity)

    def reconstruct_solution(self, K: List[List[int]], vehicles: List[Dict], max_capacity: int) -> List[Dict]:
        """
        Reconstruct the solution from the knapsack DP table.
        """
        n = len(vehicles)
        w = max_capacity
        selected_vehicles = []

        for i in range(n, 0, -1):
            if K[i][w] != K[i-1][w]:
                selected_vehicles.append(vehicles[i-1])
                w -= vehicles[i-1]['weight']
                logging.debug(f"Selected vehicle {
                              vehicles[i-1]['id']} for distribution")

        return selected_vehicles

    def simulate_traffic_flow(self, distribution: Dict[str, int], time_steps: int = 24) -> List[Dict]:
        """
        Expert-level traffic flow simulation with realistic vehicle movement patterns.
        """
        logging.info(f"Starting traffic flow simulation for {
                     time_steps} time_steps")
        flow_data = []
        current_dist = distribution.copy()
        overflow_queue = getattr(self, 'overflow_vehicles', 0)

        # Constants for flow control
        BASE_EXIT_RATE = 0.15  # Base rate at which vehicles exit the network
        MIN_UTILIZATION = 0.4  # Minimum utilization during low traffic
        PEAK_UTILIZATION = 0.9  # Maximum utilization during peak hours

        def get_time_modifier(hour: int) -> Tuple[float, float]:
            """
            Return flow modifiers based on time of day.
            Returns (exit_modifier, entry_modifier)
            """
            if 6 <= hour <= 9:  # Morning peak (more entering than leaving)
                return 0.8, 1.5
            elif 16 <= hour <= 19:  # Evening peak (more leaving than entering)
                return 1.5, 0.8
            elif 0 <= hour <= 5:  # Night time (balanced but low volume)
                return 0.5, 0.5
            else:  # Normal hours (balanced)
                return 1.0, 1.0

        def calculate_flow_rates(road: RoadSegment, current_load: int,
                                 exit_mod: float, entry_mod: float) -> Tuple[int, int]:
            """Calculate exit and entry rates for a road segment"""
            # Calculate base rates
            base_exit = int(road.capacity * BASE_EXIT_RATE * exit_mod)
            base_entry = int(road.capacity * BASE_EXIT_RATE * entry_mod)

            # Adjust for current load
            max_exit = min(base_exit, current_load)
            space_available = road.capacity - (current_load - max_exit)
            max_entry = min(base_entry, space_available)

            return max_exit, max_entry

        for t in range(time_steps):
            exit_mod, entry_mod = get_time_modifier(t % 24)
            total_capacity = sum(
                road.capacity for road in self.road_network.values())
            new_distribution = {}
            total_exits = 0
            available_entry_capacity = 0

            # Phase 1: Process exits
            for road_id, road in self.road_network.items():
                current_load = current_dist[road_id]
                exit_rate, entry_capacity = calculate_flow_rates(
                    road, current_load, exit_mod, entry_mod)

                # Remove exiting vehicles
                new_load = current_load - exit_rate
                new_distribution[road_id] = new_load
                total_exits += exit_rate
                available_entry_capacity += entry_capacity

            # Phase 2: Process entries from overflow queue
            if overflow_queue > 0 and available_entry_capacity > 0:
                # Calculate target utilization based on time of day
                target_util = PEAK_UTILIZATION if exit_mod > 1.0 else (
                    MIN_UTILIZATION if exit_mod < 0.7 else 0.7)

                # Calculate how many vehicles we want to add
                current_util = sum(new_distribution.values()) / total_capacity
                if current_util < target_util:
                    space_for_target = int((target_util * total_capacity) -
                                           sum(new_distribution.values()))
                    vehicles_to_add = min(
                        space_for_target,
                        available_entry_capacity,
                        overflow_queue
                    )

                    # Distribute new vehicles
                    for road_id, road in self.road_network.items():
                        if vehicles_to_add <= 0:
                            break

                        current_load = new_distribution[road_id]
                        target_load = int(road.capacity * target_util)
                        space_available = target_load - current_load

                        if space_available > 0:
                            add_amount = min(space_available,
                                             int(vehicles_to_add * (road.capacity / total_capacity)))
                            new_distribution[road_id] += add_amount
                            overflow_queue -= add_amount
                            vehicles_to_add -= add_amount

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
                'time_modifier': (exit_mod, entry_mod),
                'utilization_percentage': utilization,
                'total_capacity': total_capacity,
                'total_exits': total_exits,
                'available_entry_capacity': available_entry_capacity
            }
            flow_data.append(timestep_data)

            logging.info(
                f"Hour {t}: {current_load} vehicles in network "
                f"({utilization:.1f}% utilized), "
                f"{overflow_queue} in queue (Flow modifiers: {
                    exit_mod:.1f}, {entry_mod:.1f})"
            )

        return flow_data

    def visualize_network(self, distribution: Dict[str, int], flow_data: List[Dict] = None):
        """Enhanced visualization with utilization trends"""
        logging.info("Generating network visualization")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, figure=fig)

        # Network graph
        ax1 = fig.add_subplot(gs[0, :])
        G = nx.Graph()

        # Add nodes with current load information
        max_capacity = max(
            road.capacity for road in self.road_network.values())
        for road_id, load in distribution.items():
            capacity = self.road_network[road_id].capacity
            utilization = (load / capacity) * 100
            G.add_node(road_id, load=load, utilization=utilization)

        # Add edges
        edges = [
            ('mandela_north', 'interchange_1'),
            ('mandela_south', 'interchange_1'),
            ('portmore_east', 'interchange_2'),
            ('portmore_west', 'interchange_2'),
            ('interchange_1', 'interchange_2')
        ]
        G.add_edges_from(edges)

        # Draw network
        pos = nx.spring_layout(G)
        node_colors = [G.nodes[node]['utilization'] for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                       node_size=2000, cmap=plt.cm.RdYlBu_r)
        nx.draw_networkx_edges(G, pos)

        # Add labels with current load and capacity
        labels = {
            node: f'{node}\n{
                distribution[node]}/{self.road_network[node].capacity}\n{G.nodes[node]["utilization"]:.1f}%'
            for node in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.colorbar(nodes, label='Utilization %')
        ax1.set_title('Traffic Distribution Network')
        ax1.axis('off')

        if flow_data:
            # Network load over time
            ax2 = fig.add_subplot(gs[1, 0])
            times = [d['time_hour'] for d in flow_data]
            loads = [d['total_in_network'] for d in flow_data]
            queues = [d['overflow_queue'] for d in flow_data]

            ax2.plot(times, loads, 'b-', label='Vehicles in Network')
            ax2.plot(times, queues, 'r--', label='Queue Size')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Number of Vehicles')
            ax2.set_title('Network Load and Queue Size Over Time')
            ax2.legend()

            # Road utilization over time
            ax3 = fig.add_subplot(gs[1, 1])
            for road_id in self.road_network:
                utilization = [
                    d['distribution'][road_id] /
                    self.road_network[road_id].capacity * 100
                    for d in flow_data
                ]
                ax3.plot(times, utilization, label=road_id)

            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Utilization (%)')
            ax3.set_title('Road Utilization Over Time')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return plt
