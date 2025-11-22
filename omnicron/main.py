from manager import GameMemory
import numpy as np

mem = GameMemory()

# Write an example record
board = np.zeros((3,3), dtype=np.int8)
move = np.array([1,1], dtype=np.int8)

mem.write("tic_tac_toe", outcome=2, board=board, move=move)

# Query best move
best = mem.get_best_move("tic_tac_toe", board)
print("best move:", best)