import gym
import gym_chess
import numpy as np


def parse_unicode_state(unicode_str):
    # Mapping from Unicode pieces to integers
    piece_map = {
        '♙': 1,  # White pawn
        '♖': 2,  # White rook
        '♘': 3,  # White knight
        '♗': 4,  # White bishop
        '♕': 5,  # White queen
        '♔': 6,  # White king
        '♟': -1,  # Black pawn
        '♜': -2,  # Black rook
        '♞': -3,  # Black knight
        '♝': -4,  # Black bishop
        '♛': -5,  # Black queen
        '♚': -6,  # Black king
        '⭘': 0,  # Empty square (if using ⭘ for empty)
        ' ': 0,  # Empty square (if using space for empty)
    }

    state = []

    # Split the unicode_str into rows and process each row
    rows = unicode_str.splitlines()
    for row in rows:
        for char in row:
            # Append the numerical value corresponding to the piece in the square
            state.append(piece_map.get(char, 0))  # Default to 0 if unknown character

    # Ensure the state is a 1D array with 64 elements
    return np.array(state[:64])  # Cut off any excess data (if any) to keep the board size 64


def encode_state(env):
    """
    Convert the environment's current state to a numerical representation.
    """
    # Render the board to Unicode and convert it to a numerical state
    unicode_str = env.render(mode='unicode')
    return parse_unicode_state(unicode_str)

