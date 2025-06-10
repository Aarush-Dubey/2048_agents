# game_engine.py
import numpy as np
import random
from typing import List, Tuple, Optional
from enum import Enum

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
    
    def copy(self):
        """Create a deep copy of the game state"""
        new_game = Game2048(self.size)
        new_game.board = self.board.copy()
        new_game.score = self.score
        return new_game
    
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4
    
    def get_legal_moves(self) -> List[Direction]:
        """Get all legal moves from current state"""
        legal_moves = []
        for direction in Direction:
            if self.can_move(direction):
                legal_moves.append(direction)
        return legal_moves
    
    def can_move(self, direction: Direction) -> bool:
        """Check if a move is legal"""
        temp_game = self.copy()
        old_board = temp_game.board.copy()
        temp_game._move_without_adding_tile(direction)
        return not np.array_equal(old_board, temp_game.board)
    
    def move(self, direction: Direction) -> bool:
        """Make a move and add a random tile"""
        if not self.can_move(direction):
            return False
        
        self._move_without_adding_tile(direction)
        self.add_random_tile()
        return True
    
    def _move_without_adding_tile(self, direction: Direction):
        """Internal method to move without adding tile (for simulation)"""
        if direction == Direction.LEFT:
            self._move_left()
        elif direction == Direction.RIGHT:
            self._move_right()
        elif direction == Direction.UP:
            self._move_up()
        elif direction == Direction.DOWN:
            self._move_down()
    
    def _move_left(self):
        for i in range(self.size):
            self.board[i] = self._merge_line(self.board[i])
    
    def _move_right(self):
        for i in range(self.size):
            self.board[i] = self._merge_line(self.board[i][::-1])[::-1]
    
    def _move_up(self):
        for j in range(self.size):
            column = self.board[:, j]
            self.board[:, j] = self._merge_line(column)
    
    def _move_down(self):
        for j in range(self.size):
            column = self.board[:, j]
            self.board[:, j] = self._merge_line(column[::-1])[::-1]
    
    def _merge_line(self, line):
        """Merge a single line (row or column)"""
        # Remove zeros
        non_zero = line[line != 0]
        
        # Merge adjacent equal tiles
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        result = np.zeros(self.size, dtype=int)
        result[:len(merged)] = merged
        return result
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return len(self.get_legal_moves()) == 0
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value"""
        return int(np.max(self.board))
    
    def get_empty_cells(self) -> int:
        """Get number of empty cells"""
        return int(np.sum(self.board == 0))
