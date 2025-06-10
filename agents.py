# agents.py
import time
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from game_engine import Game2048, Direction

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_move(self, game: Game2048) -> Direction:
        pass

class RandomAgent(Agent):
    def __init__(self):
        super().__init__("Random")
    
    def get_move(self, game: Game2048) -> Direction:
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else Direction.UP

class GreedyAgent(Agent):
    def __init__(self):
        super().__init__("Greedy")
    
    def get_move(self, game: Game2048) -> Direction:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return Direction.UP
        
        best_move = legal_moves[0]
        best_score = -1
        
        for move in legal_moves:
            temp_game = game.copy()
            temp_game._move_without_adding_tile(move)
            if temp_game.score > best_score:
                best_score = temp_game.score
                best_move = move
        
        return best_move

class HeuristicAgent(Agent):
    def __init__(self):
        super().__init__("Heuristic")
    
    def get_move(self, game: Game2048) -> Direction:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return Direction.UP
        
        best_move = legal_moves[0]
        best_score = float('-inf')
        
        for move in legal_moves:
            temp_game = game.copy()
            temp_game._move_without_adding_tile(move)
            score = self._evaluate_board(temp_game)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _evaluate_board(self, game: Game2048) -> float:
        """Heuristic evaluation combining multiple factors"""
        empty_weight = 2.7
        mono_weight = 1.0
        smooth_weight = 0.1
        max_weight = 1.0
        
        empty_score = game.get_empty_cells() * empty_weight
        mono_score = self._monotonicity(game.board) * mono_weight
        smooth_score = self._smoothness(game.board) * smooth_weight
        max_score = math.log2(game.get_max_tile()) * max_weight if game.get_max_tile() > 0 else 0
        
        return empty_score + mono_score + smooth_score + max_score
    
    def _monotonicity(self, board) -> float:
        """Calculate monotonicity of the board"""
        totals = [0, 0, 0, 0]  # up, down, left, right
        
        for x in range(4):
            current = 0
            next_val = current + 1
            
            while next_val < 4:
                while next_val < 4 and board[x][next_val] == 0:
                    next_val += 1
                
                if next_val >= 4:
                    next_val -= 1
                
                current_val = board[x][current] if board[x][current] != 0 else 1
                next_val_val = board[x][next_val] if board[x][next_val] != 0 else 1
                
                if current_val > next_val_val:
                    totals[0] += math.log2(next_val_val) - math.log2(current_val)
                elif next_val_val > current_val:
                    totals[1] += math.log2(current_val) - math.log2(next_val_val)
                
                current = next_val
                next_val += 1
        
        for y in range(4):
            current = 0
            next_val = current + 1
            
            while next_val < 4:
                while next_val < 4 and board[next_val][y] == 0:
                    next_val += 1
                
                if next_val >= 4:
                    next_val -= 1
                
                current_val = board[current][y] if board[current][y] != 0 else 1
                next_val_val = board[next_val][y] if board[next_val][y] != 0 else 1
                
                if current_val > next_val_val:
                    totals[2] += math.log2(next_val_val) - math.log2(current_val)
                elif next_val_val > current_val:
                    totals[3] += math.log2(current_val) - math.log2(next_val_val)
                
                current = next_val
                next_val += 1
        
        return max(totals[0], totals[1]) + max(totals[2], totals[3])
    
    def _smoothness(self, board) -> float:
        """Calculate smoothness of the board"""
        smoothness = 0
        
        for x in range(4):
            for y in range(4):
                if board[x][y] != 0:
                    val = math.log2(board[x][y])
                    
                    # Check right neighbor
                    if x < 3 and board[x+1][y] != 0:
                        target_val = math.log2(board[x+1][y])
                        smoothness -= abs(val - target_val)
                    
                    # Check down neighbor
                    if y < 3 and board[x][y+1] != 0:
                        target_val = math.log2(board[x][y+1])
                        smoothness -= abs(val - target_val)
        
        return smoothness

class ExpectimaxAgent(Agent):
    def __init__(self, depth=4):
        super().__init__(f"Expectimax-{depth}")
        self.depth = depth
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_move(self, game: Game2048) -> Direction:
        self.cache.clear()  # Clear cache for each move
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return Direction.UP
        
        best_move = legal_moves[0]
        best_score = float('-inf')
        
        for move in legal_moves:
            temp_game = game.copy()
            temp_game._move_without_adding_tile(move)
            score = self._expectimax(temp_game, self.depth - 1, False)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _expectimax(self, game: Game2048, depth: int, is_player_turn: bool) -> float:
        """Optimized expectimax with memoization and pruning"""
        # Base case
        if depth == 0 or game.is_game_over():
            return self._evaluate_board_fast(game)
        
        # Cache key based on board state and depth
        board_key = hash(game.board.tobytes()) + depth * 1000000 + (1 if is_player_turn else 0)
        if board_key in self.cache:
            self.cache_hits += 1
            return self.cache[board_key]
        
        self.cache_misses += 1
        
        if is_player_turn:
            # Player's turn - maximize
            max_score = float('-inf')
            legal_moves = game.get_legal_moves()
            
            for move in legal_moves:
                temp_game = game.copy()
                temp_game._move_without_adding_tile(move)
                score = self._expectimax(temp_game, depth - 1, False)
                max_score = max(max_score, score)
            
            result = max_score if max_score != float('-inf') else 0
        else:
            # Random tile placement - expectation
            empty_cells = [(i, j) for i in range(4) for j in range(4) if game.board[i, j] == 0]
            
            if not empty_cells:
                result = self._evaluate_board_fast(game)
            else:
                total_score = 0
                # Sample only a subset of empty cells for performance
                sample_size = min(len(empty_cells), 4)  # Limit to 4 cells max
                sampled_cells = random.sample(empty_cells, sample_size)
                
                for i, j in sampled_cells:
                    # Try placing 2 (90% probability)
                    temp_game = game.copy()
                    temp_game.board[i, j] = 2
                    score_2 = self._expectimax(temp_game, depth - 1, True)
                    
                    # Try placing 4 (10% probability)
                    temp_game.board[i, j] = 4
                    score_4 = self._expectimax(temp_game, depth - 1, True)
                    
                    total_score += 0.9 * score_2 + 0.1 * score_4
                
                result = total_score / len(sampled_cells)
        
        self.cache[board_key] = result
        return result
    
    def _evaluate_board_fast(self, game: Game2048) -> float:
        """Fast heuristic evaluation for terminal nodes"""
        if game.is_game_over():
            return -1000
        
        # Simplified evaluation for speed
        empty_cells = game.get_empty_cells()
        max_tile = game.get_max_tile()
        
        # Corner bonus
        corner_bonus = 0
        corners = [game.board[0,0], game.board[0,3], game.board[3,0], game.board[3,3]]
        if max_tile in corners:
            corner_bonus = 1000
        
        return empty_cells * 10 + math.log2(max_tile) * 5 + corner_bonus + game.score * 0.1
