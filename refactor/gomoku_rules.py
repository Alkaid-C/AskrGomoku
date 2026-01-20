"""
Gomoku Game Rules and Logic.

Contains the stable game engine:
- Gomoku board implementation
- Renju opening sequences
- Helper functions for board state manipulation and analysis
"""

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Optional


# ============================================================================ 
# Renju Opening Sequences
# ============================================================================ 

def _generate_renju_openings() -> List[Tuple[Tuple[int, int], ...]]:
    """
    Generate all 184 Renju opening sequences.
    """
    openings = []
    # Second move: [-1,1] x [-1,1] excluding (0,0)
    for r2 in range(-1, 2):
        for c2 in range(-1, 2):
            if (r2, c2) == (0, 0):
                continue
            # Third move: [-2,2] x [-2,2] excluding first and second moves
            for r3 in range(-2, 3):
                for c3 in range(-2, 3):
                    if (r3, c3) == (0, 0) or (r3, c3) == (r2, c2):
                        continue
                    openings.append(((0, 0), (r2, c2), (r3, c3)))
    return openings


RENJU_OPENING_SEQUENCES = _generate_renju_openings()


# ============================================================================ 
# Enums
# ============================================================================ 

class Player(Enum):
    """Enum representing the two players."""
    BLACK = 1
    WHITE = 2


class GameState(Enum):
    """Enum representing the game state after a move."""
    CONTINUE = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3


# ============================================================================ 
# Gomoku Board Engine
# ============================================================================ 

class GomokuBoard:
    """
    Gomoku board engine with canonical output for neural networks.

    Internal state maintains absolute colors (black_pieces, white_pieces) as numpy arrays.
    Public API exposes board in canonical form (current player, opponent).
    """

    def __init__(self, opening_id: int = -1):
        """
        Initialize a 15x15 Gomoku board.

        Args:
            opening_id: If -1, start with empty board. If >= 0, start with the
                        specified Renju opening (index into RENJU_OPENING_SEQUENCES)
                        with a random offset applied.
        """
        self.black_pieces = np.zeros((15, 15), dtype=np.uint8)
        self.white_pieces = np.zeros((15, 15), dtype=np.uint8)
        self.who_to_play = Player.BLACK
        self.occupied_count = 0  # Track number of stones on board

        if opening_id >= 0:
            # Apply Renju opening with random offset
            # Offset: first move can be anywhere in center ±3 (rows/cols 4-10)
            offset_r = random.randint(-3, 3)
            offset_c = random.randint(-3, 3)
            base_r, base_c = 7 + offset_r, 7 + offset_c

            for rel_r, rel_c in RENJU_OPENING_SEQUENCES[opening_id]:
                self.Move((base_r + rel_r, base_c + rel_c))

    def GetLegalMoves(self) -> Tuple[np.ndarray, Player]:
        """
        Returns legal moves mask and current player to move.

        Returns:
            Tuple of (legal_mask, next_player) where:
            - legal_mask: 15x15 numpy array with 1 for legal moves, 0 for illegal
            - next_player: absolute color (BLACK or WHITE) to move
        """
        # Vectorized: legal where neither black nor white has a piece
        legal_mask = ((self.black_pieces == 0) & (self.white_pieces == 0)).astype(np.uint8)
        return (legal_mask, self.who_to_play)

    def GetBoardState(self) -> Tuple[np.ndarray, np.ndarray, Player]:
        """
        Returns canonical board state for neural network input.

        Returns:
            Tuple of (c0, c1, next_player) where:
            - c0: next player's pieces (15x15 numpy array, REFERENCE)
            - c1: opponent's pieces (15x15 numpy array, REFERENCE)
            - next_player: absolute color (BLACK or WHITE)
        """
        if self.who_to_play == Player.BLACK:
            return (self.black_pieces, self.white_pieces, self.who_to_play)
        else:  # WHITE
            return (self.white_pieces, self.black_pieces, self.who_to_play)

    def Move(self, position: Tuple[int, int]) -> GameState:
        """
        Place a stone at the given position for the current player.

        Args:
            position: Tuple (row, col) where 0 <= row, col < 15

        Returns:
            GameState: BLACK_WIN, WHITE_WIN, DRAW, or CONTINUE
        """
        row, col = position
        current_player = self.who_to_play

        # Place the stone for current player
        if current_player == Player.BLACK:
            self.black_pieces[row, col] = 1
            current_pieces = self.black_pieces
        else:
            self.white_pieces[row, col] = 1
            current_pieces = self.white_pieces

        # Update occupied count
        self.occupied_count += 1

        # Evaluate terminal state
        state = GameState.CONTINUE
        if self._check_win(row, col, current_pieces):
            state = GameState.BLACK_WIN if current_player == Player.BLACK else GameState.WHITE_WIN
        elif self._is_board_full():
            state = GameState.DRAW

        # Toggle player (regardless of outcome)
        self.who_to_play = Player.WHITE if current_player == Player.BLACK else Player.BLACK

        return state

    def Render(self) -> str:
        """
        Render the board as an ASCII string.
        """
        chars = np.full((15, 15), '.', dtype='U1')
        chars[self.black_pieces == 1] = 'x'
        chars[self.white_pieces == 1] = 'o'
        return '\n'.join(''.join(row) for row in chars)

    def _check_win(self, row: int, col: int, pieces: np.ndarray) -> bool:
        """
        Check if placing a stone at (row, col) results in a win.
        """
        # Four directions: horizontal, vertical, diagonal, anti-diagonal
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        for dr, dc in directions:
            count = 1  # Count the placed stone itself

            # Count contiguous stones in forward direction
            r, c = row + dr, col + dc
            while 0 <= r < 15 and 0 <= c < 15 and pieces[r, c] == 1:
                count += 1
                r += dr
                c += dc

            # Count contiguous stones in backward direction
            r, c = row - dr, col - dc
            while 0 <= r < 15 and 0 <= c < 15 and pieces[r, c] == 1:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False

    def _is_board_full(self) -> bool:
        """
        Check if the board is completely full.
        """
        return self.occupied_count == 225


# ============================================================================ 
# Helper Functions (Logic & Analysis)
# ============================================================================ 

def encode_observation(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """
    Encode board state into a numpy array.

    Args:
        c0: Current player's pieces [15, 15] numpy array
        c1: Opponent's pieces [15, 15] numpy array

    Returns:
        obs: [3, 15, 15] numpy array (uint8)
    """
    board_mask = np.ones((15, 15), dtype=np.uint8)
    return np.stack([c0, c1, board_mask], axis=0)


def idx_to_pos(idx: int) -> Tuple[int, int]:
    """Convert flat index to (row, col)."""
    return (idx // 15, idx % 15)

def pos_to_idx(row: int, col: int) -> int:
    """Convert (row, col) to flat index."""
    return row * 15 + col

def board_from_observation(obs: np.ndarray, next_player: Player) -> GomokuBoard:
    """
    Reconstruct a GomokuBoard from an observation and player to move.
    """
    board = GomokuBoard()
    current_pieces = obs[0]
    opponent_pieces = obs[1]

    if next_player == Player.BLACK:
        board.black_pieces = current_pieces.copy()
        board.white_pieces = opponent_pieces.copy()
    else:
        board.white_pieces = current_pieces.copy()
        board.black_pieces = opponent_pieces.copy()

    board.who_to_play = next_player
    board.occupied_count = int(np.sum(current_pieces) + np.sum(opponent_pieces))

    return board

def get_local_candidate_moves(obs: np.ndarray, legal_mask: np.ndarray, radius: int = 2) -> List[int]:
    """
    Get legal moves within Manhattan distance of existing stones.
    """
    occupied = ((obs[0] == 1) | (obs[1] == 1))
    candidates = []

    # Get all legal positions
    legal_positions = np.argwhere(legal_mask == 1)

    for pos in legal_positions:
        r, c = pos[0], pos[1]
        # Check if any stone within Manhattan distance
        found_neighbor = False
        for dr in range(-radius, radius + 1):
            if found_neighbor:
                break
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 15 and 0 <= nc < 15 and occupied[nr, nc]:
                    candidates.append(r * 15 + c)
                    found_neighbor = True
                    break

    return candidates

def is_winning_move(board_c0: np.ndarray, row: int, col: int) -> bool:
    """
    Check if placing current player's piece at (row, col) creates 5-in-a-row.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < 15 and 0 <= c < 15 and board_c0[r, c] == 1:
            count += 1
            r += dr
            c += dc

        r, c = row - dr, col - dc
        while 0 <= r < 15 and 0 <= c < 15 and board_c0[r, c] == 1:
            count += 1
            r -= dr
            c -= dc

        if count >= 5:
            return True

    return False

def find_all_win_in_1(obs: np.ndarray, legal_mask: np.ndarray) -> List[int]:
    """
    Search for all win-in-1 moves for the current player.
    """
    board_c0 = obs[0]
    winning_moves = []
    legal_positions = np.argwhere(legal_mask == 1)

    for pos in legal_positions:
        row, col = pos[0], pos[1]
        if is_winning_move(board_c0, row, col):
            winning_moves.append(row * 15 + col)

    return winning_moves

def find_blocking_moves(obs: np.ndarray, legal_mask: np.ndarray) -> Optional[List[int]]:
    """
    Find moves that block opponent's win-in-1 threats.
    Returns None if there are multiple independent threats (dual threat).
    """
    board_opponent = obs[1]
    blocking_positions = []
    legal_positions = np.argwhere(legal_mask == 1)

    for pos in legal_positions:
        row, col = pos[0], pos[1]
        if is_winning_move(board_opponent, row, col):
            blocking_positions.append(row * 15 + col)
            if len(blocking_positions) >= 2:
                return None

    return blocking_positions if blocking_positions else None
