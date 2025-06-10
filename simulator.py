
# simulator.py
import pandas as pd
import numpy as np
from typing import Dict, List
import time
from dataclasses import dataclass
from agents import Agent
from game_engine import Game2048
from collections import defaultdict

@dataclass
class GameResult:
    score: int
    max_tile: int
    moves: int
    duration: float
    move_frequencies: Dict[str, int]

class GameSimulator:
    def __init__(self):
        self.results = {}
    
    def simulate_agent(self, agent: Agent, num_games: int = 1000) -> List[GameResult]:
        """Simulate multiple games for an agent"""
        results = []
        
        print(f"Simulating {num_games} games for {agent.name}...")
        
        for game_num in range(num_games):
            if game_num % 100 == 0:
                print(f"  Game {game_num}/{num_games}")
            
            result = self._simulate_single_game(agent)
            results.append(result)
        
        self.results[agent.name] = results
        return results
    
    def _simulate_single_game(self, agent: Agent) -> GameResult:
        """Simulate a single game"""
        game = Game2048()
        moves = 0
        move_frequencies = defaultdict(int)
        decision_times = []
        
        start_time = time.time()
        
        while not game.is_game_over() and moves < 100000:  # Prevent infinite games
            start_move_time = time.time()
            move = agent.get_move(game)
            decision_time = time.time() - start_move_time
            decision_times.append(decision_time)
            
            if game.move(move):
                moves += 1
                move_frequencies[move.name] += 1
        
        total_time = time.time() - start_time
        
        return GameResult(
            score=game.score,
            max_tile=game.get_max_tile(),
            moves=moves,
            duration=total_time,
            move_frequencies=dict(move_frequencies)
        )
    
    def get_statistics(self, agent_name: str) -> Dict:
        """Get comprehensive statistics for an agent"""
        if agent_name not in self.results:
            return {}
        
        results = self.results[agent_name]
        scores = [r.score for r in results]
        max_tiles = [r.max_tile for r in results]
        moves = [r.moves for r in results]
        durations = [r.duration for r in results]
        
        # Calculate average decision time per move
        total_moves = sum(moves)
        total_decision_time = sum(durations)
        avg_decision_time = (total_decision_time / total_moves * 1000) if total_moves > 0 else 0
        
        # Aggregate move frequencies
        all_move_freqs = defaultdict(int)
        for result in results:
            for move, freq in result.move_frequencies.items():
                all_move_freqs[move] += freq
        
        return {
            'agent_name': agent_name,
            'num_games': len(results),
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_median': np.median(scores),
            'score_max': np.max(scores),
            'max_tile_mean': np.mean(max_tiles),
            'max_tile_mode': max(set(max_tiles), key=max_tiles.count),
            'moves_mean': np.mean(moves),
            'avg_decision_time_ms': avg_decision_time,
            'move_frequencies': dict(all_move_freqs),
            'raw_results': results
        }
