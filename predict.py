import chess
import torch
import numpy as np

from model import ChessNet

# =========================
# PIECE ENCODING
# =========================

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

# =========================
# BOARD TO MATRIX
# =========================

def board_to_matrix(board):

    matrix = np.zeros((8,8))

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece:

            row = 7 - (square // 8)
            col = square % 8

            matrix[row][col] = piece_map[str(piece)]

    return matrix

# =========================
# REBUILD MOVE VOCAB
# =========================

import pandas as pd

df = pd.read_csv("games.csv")

all_moves = []

for game in df['moves'][:500]:

    moves = game.split()

    all_moves.extend(moves)

unique_moves = sorted(list(set(all_moves)))

move_to_id = {
    move: idx for idx, move in enumerate(unique_moves)
}

id_to_move = {
    idx: move for move, idx in move_to_id.items()
}

# =========================
# LOAD MODEL
# =========================

model = ChessNet(len(unique_moves))

model.load_state_dict(torch.load("chessmate_model.pth"))

model.eval()

# =========================
# CREATE BOARD
# =========================

board = chess.Board()

# Example moves
board.push_san("c4")


# Convert board
matrix = board_to_matrix(board)

X = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

# Prediction
with torch.no_grad():

    output = model(X)

    predicted_id = torch.argmax(output).item()

predicted_move = id_to_move[predicted_id]

print("\nChessMate Suggests:")
print(predicted_move)