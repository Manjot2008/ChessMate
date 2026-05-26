import chess
import numpy as np

piece_map = {
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,

    'p': -1,
    'n': -2,
    'b': -3,
    'r': -4,
    'q': -5,
    'k': -6
}

def board_to_matrix(board):

    matrix = np.zeros((8,8))

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece:

            row = 7 - (square // 8)
            col = square % 8

            matrix[row][col] = piece_map[str(piece)]

    return matrix

board = chess.Board()

matrix = board_to_matrix(board)

print(matrix)