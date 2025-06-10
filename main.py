import time
from agents import RandomAgent, GreedyAgent, HeuristicAgent, ExpectimaxAgent
from game_engine import Game2048


# main.py - Example usage
def main():
    # Create agents
    agents = [
        RandomAgent(),
        GreedyAgent(),
        HeuristicAgent(),
        ExpectimaxAgent(depth=3),
        ExpectimaxAgent(depth=4)
    ]
    
    # Create simulator
    simulator = GameSimulator()
    
    # Run simulations
    num_games = 100  # Start with smaller number for testing
    
    for agent in agents:
        print(f"\n{'='*50}")
        print(f"Testing {agent.name}")
        print('='*50)
        
        # Time the simulation
        start_time = time.time()
        results = simulator.simulate_agent(agent, num_games)
        end_time = time.time()
        
        # Get statistics
        stats = simulator.get_statistics(agent.name)
        
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"Average score: {stats['score_mean']:.0f}")
        print(f"Max score: {stats['score_max']:.0f}")
        print(f"Average max tile: {stats['max_tile_mean']:.0f}")
        print(f"Average decision time: {stats['avg_decision_time_ms']:.1f} ms")
        print(f"Move frequencies: {stats['move_frequencies']}")

if __name__ == "__main__":
    main()