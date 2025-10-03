import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm

from flip7_game import Flip7Game


def create_ds(dataset_size = 50_000):
    game = Flip7Game()
    observations = []
    value_diffs = []
    for _ in tqdm(range(dataset_size)):
        if game.round_over:
            game.reset()

        observation = np.array(list(game.card_counts_in_hand.values()))
        game.in_()
        value_dif = np.array([game.get_score_difference()])
        observations.append(observation)
        value_diffs.append(value_dif)

    tensor_x = torch.Tensor(observations)
    tensor_y = torch.Tensor(value_diffs)
    return TensorDataset(tensor_x, tensor_y)

def train(model: nn.Module, dataset: Dataset, test_dataset: Dataset, num_epochs: int = 100, batch_size: int = 128):
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        model.train()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_running_loss = 0
        for batch in data_loader:
            input_state, label = batch
            prediction = model(input_state)
            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        print(f"Epoch {epoch} loss: {train_running_loss / len(data_loader)}")

        if epoch % 10 == 0:
            model.eval()
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            test_running_loss = 0
            for batch in data_loader:
                input_state, label = batch
                prediction = model(input_state)
                loss = criterion(prediction, label)
                test_running_loss += loss.item()
            print(f"Epoch {epoch} loss: {test_running_loss / len(data_loader)}")

def evaluate(model: nn.Module):
    print("Evaluating...")
    score = 0
    game = Flip7Game()
    num_moves = 100
    for _ in range(num_moves):
        if game.round_over:
            game.reset()

        observation = Tensor(np.array(list(game.card_counts_in_hand.values()))).unsqueeze(0)
        prediction = model(observation)

        if prediction > 0:
            game.in_()
            score += game.get_score_difference()
        else:
            game.reset()

    print(f"Final score per move: {score/num_moves}")



_model = nn.Sequential(
        nn.Linear(Flip7Game.NUM_CARDS, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


evaluate(_model)
train(_model, dataset=create_ds(300_000), test_dataset=create_ds(10_000), num_epochs=30, batch_size=64)
evaluate(_model)
