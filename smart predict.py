import chess
import torch
import numpy as np
import pandas as pd

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
# LOAD MOVES
# =========================

df = pd.read_csv("games.csv")

all_moves = []

for game in df['moves'][:1500]:

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

board.push_san("e4")
board.push_san("e5")

# =========================
# PREDICT
# =========================

matrix = board_to_matrix(board)

X = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():

    output = model(X)

    probabilities = torch.softmax(output, dim=1)

# Top predictions
top_moves = torch.topk(probabilities, 20)

print("\nTop Legal Predictions:\n")

found = False

for idx in top_moves.indices[0]:

    move = id_to_move[idx.item()]

    try:

        board.parse_san(move)

        print(move)

        found = True

    except:

        continue

if not found:

    print("No legal move found.")