import pandas as pd

# Load dataset
df = pd.read_csv("games.csv")

all_moves = []

# Collect all moves
for game in df['moves']:

    moves = game.split()

    all_moves.extend(moves)

# Unique moves
unique_moves = sorted(list(set(all_moves)))

# Create move-to-id mapping
move_to_id = {
    move: idx for idx, move in enumerate(unique_moves)
}

id_to_move = {
    idx: move for move, idx in move_to_id.items()
}

print("Total unique moves:", len(unique_moves))

print("\nFirst 20 move mappings:\n")

for move, idx in list(move_to_id.items())[:20]:

    print(f"{move} --> {idx}")