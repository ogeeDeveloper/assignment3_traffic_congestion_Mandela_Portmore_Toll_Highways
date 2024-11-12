# Traffic Congestion Analysis System

## Mandela and Portmore Toll Highways

### Overview

An advanced traffic optimization system that uses dynamic programming approaches (Knapsack Algorithm and Custom DP) to manage and reduce congestion on the Mandela and Portmore Toll Highways. The system provides real-time traffic distribution optimization, visualization, and analysis tools.

### 🌟 Key Features

- Dynamic traffic flow optimization
- Real-time visualization of traffic patterns
- Custom dynamic programming algorithm
- Knapsack algorithm implementation
- Comprehensive reporting and analysis
- Configurable simulation parameters
- Interactive network visualization

### 📋 Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements.txt)
- Windows/Linux/MacOS

### 🛠️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/ogeeDeveloper/assignment3_traffic_congestion_Mandela_Portmore_Toll_Highways.git
    cd assignment3_traffic_congestion_Mandela_Portmore_Toll_Highways
   ```

2. Create and activate virtual environment
   ```bash
    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    # On Windows
    venv\Scripts\activate
    # On Linux/Mac
    source venv/bin/activate
   ```

3. Install required packages
   ``` bash
   pip install -r requirements.txt
   ```


### 🚀 Quick Start

1. Basic usage:
   ``` python
    from src.traffic_optimizer import TrafficOptimizer
    import yaml

    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize optimizer
    optimizer = TrafficOptimizer(config)
    optimizer.initialize_network()

    # Run optimization
    distribution = optimizer.custom_traffic_distribution(total_vehicles=5000)

    # Visualize results
    optimizer.visualize_network(distribution)
   ```

2. Command-line usage:
   ``` bash
    # Run with default settings
    python main.py

    # Run with custom parameters
    python main.py --vehicles 6000 --time-steps 48
   ```

### ⚙️ Configuration

Modify `config/config.yaml` to adjust:

- Road network parameters
- Optimization settings
- Visualization preferences
- Simulation parameters

Example configuration:
``` yaml
simulation:
  default_vehicles: 5000
  default_time_steps: 24

road_network:
  mandela_north:
    id: "MN"
    capacity: 2000
    travel_time: 30
```

### 📊 Features in Detail

1. **Traffic Optimization**
    - Knapsack algorithm for vehicle distribution
    - Custom dynamic programming solution
    - Real-time optimization
2. **Visualization**
    - Network graph visualization
    - Traffic flow patterns
    - Interactive diagrams
3. **Analysis Tools**
    - Traffic flow simulation
    - Performance metrics
    - Comprehensive reporting

### 📝 API Documentation

## TrafficOptimizer Class

``` python
optimizer = TrafficOptimizer(config)
```

Key methods:

- `initialize_network()`: Set up road network
- `custom_traffic_distribution(total_vehicles)`: Optimize traffic
- `simulate_traffic_flow(distribution, time_steps)`: Simulate flow
- `visualize_network(distribution)`: Create visualization
- `generate_report()`: Generate analysis report


### 📈 Performance Metrics

The system tracks:

- Average travel time
- Road utilization rates
- Congestion levels
- Traffic flow efficiency