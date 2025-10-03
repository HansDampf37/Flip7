import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm

from flip7_game import Flip7Game


def create_dataset(dataset_size: int = 5_000) -> TensorDataset:
    """
    Generate a dataset of game states and score differences.
    Each sample = (card_counts_in_hand, score difference after drawing one card).
    """
    game = Flip7Game()
    observations = []
    value_diffs = []

    for _ in tqdm(range(dataset_size), desc="Generating dataset"):
        if game.round_over:
            game.reset()

        # Observation: counts in fixed CARD_TYPES order
        observation = np.array([game.card_counts_in_hand[c] for c in Flip7Game.CARD_TYPES], dtype=np.float32)
        value_diff = np.array([game.mc_sample_expected_score_difference_of_in(200)], dtype=np.float32)
        game.in_()

        observations.append(observation)
        value_diffs.append(value_diff)

    tensor_x = torch.tensor(np.stack(observations))
    tensor_y = torch.tensor(np.stack(value_diffs))
    return TensorDataset(tensor_x, tensor_y)


def train(
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = None,
):
    """
    Train model with Huber loss and Adam optimizer.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for input_state, label in train_loader:
            input_state, label = input_state.to(device), label.to(device)

            optimizer.zero_grad()
            prediction = model(input_state)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Testing
        if epoch % 5 == 0 or epoch == num_epochs:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for input_state, label in test_loader:
                    input_state, label = input_state.to(device), label.to(device)
                    prediction = model(input_state)
                    loss = criterion(prediction, label)
                    test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
        else:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f}")


def evaluate(policy, num_moves: int = 1000):
    """
    Evaluate a policy. The policy should be a function that takes a flip7 game instance and returns true or false.
    """
    game = Flip7Game()
    total_score = 0
    round_scores = []

    for _ in tqdm(range(num_moves)):
        if game.round_over:
            round_scores.append(game.get_score())
            game.reset()

        if policy(game):
            game.in_()
            total_score += game.get_score_difference()
        else:
            round_scores.append(game.get_score())
            game.reset()

    print(f"Evaluation complete: \n\tAvg score per move: {total_score / num_moves:.2f}"
          f"\n\tAvg score per round: {sum(round_scores) / len(round_scores) :.2f}")
    return round_scores
