import numpy as np
import gym
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from stockfish import Stockfish
from gym_chess.alphazero.board_encoding import encode_board, decode_board, encode_move, decode_move

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 12)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)
        return x

def train(model, optimizer, criterion, board, move):
    optimizer.zero_grad()
    input = torch.tensor(encode_board(board), dtype=torch.float32)
    target = torch.tensor(encode_move(move, board), dtype=torch.long)
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, board):
    input = torch.tensor(encode_board(board), dtype=torch.float32)
    output = model(input)
    move_num = output.argmax().item()
    return decode_move(board, move_num)

def play(model, board, opponent):
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = evaluate(model,board)
            board.push(move)
            print(f"{board}")
        else:
            move = opponent.play(board, chess.engine.Limit(time=2.0))
            board.push(move)
            print(f"{board}")

model = ChessModel()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

board = chess.Board()
opponent = Stockfish()

for epoch in range(1000):
    board = chess.Board()
    move = opponent.play(board, chess.engine.Limit(time=2.0))
    loss = train(model, optimizer, criterion, board, move)
    print(f"Epoch {epoch+1}/{1000}, Loss: {loss:.4f}")

play(model, board, opponent)