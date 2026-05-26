import chess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ChessNet

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("games.csv")

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
# MOVE ENCODING
# =========================

all_moves = []

for game in df['moves'][:1500]:

    moves = game.split()

    all_moves.extend(moves)

unique_moves = sorted(list(set(all_moves)))

move_to_id = {
    move: idx for idx, move in enumerate(unique_moves)
}

# =========================
# CREATE TRAINING DATA
# =========================

X = []
y = []

games = df['moves'][:1500]

for game in games:

    board = chess.Board()

    moves = game.split()

    for move in moves:

        try:

            matrix = board_to_matrix(board)

            X.append(matrix)

            y.append(move_to_id[move])

            board.push_san(move)

        except:
            break

# Convert to tensors
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

print("Training samples:", len(X))

# =========================
# MODEL
# =========================

model = ChessNet(len(unique_moves))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================

epochs = 10
batch_size = 256

for epoch in range(epochs):

    total_loss = 0

    for i in range(0, len(X), batch_size):

        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "chessmate_model.pth")

print("\nModel trained successfully!")