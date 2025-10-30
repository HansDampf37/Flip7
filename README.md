Flip7 — Optimal play with MC sampling
========================================

![](https://m.media-amazon.com/images/S/aplus-media-library-service-media/a8277586-b146-44f0-adf8-8127971f5ebb.__CR0,23,1920,594_PT0_SX970_V1___.jpg)

Short description
-----------------
Flip7 is a small card-game simulation focused on decision-making under uncertainty. The goal of this repository is to simulate the game, use Monte‑Carlo sampling to estimate move values, and train a neural regression model that approximates the expected score change for the next draw.

Setup
-----------
Requirements: Linux / macOS / Windows with Python (recommended: 3.12, see `environment.yaml`). Two simple ways to install dependencies are shown below.

Conda (recommended)

```bash
conda env create -f environment.yaml
conda activate flip7-env
```

If you need CUDA support, install a CUDA-enabled PyTorch build appropriate for your machine following the instructions at https://pytorch.org.

Pip / Virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: install a CUDA-enabled PyTorch build per the instructions on pytorch.org
```

Game rules (brief)
-------------------
- The deck contains numbered cards (0–12) and special cards (e.g. `+2`, `+4`, `x2`, "Second Chance", "Flip Three", "Freeze").
- A player draws cards into their hand using the `in_()` operation.
- Number cards add to the score (each number counts only once); drawing a duplicate number busts the round (score = 0) unless a "Second Chance" card is present.
- Special cards modify the score (additive or multiplicative modifiers) or trigger actions such as "Flip Three" (draw additional cards immediately), "Freeze" (end the round immediately) or "Second Chance" (ignore a duplicate once).
- If 7 distinct number cards are collected, a bonus of +15 is awarded and the round ends.

Approach
--------
This project uses two related but distinct approaches to estimate the value of taking the `in_()` action from a given game state:

### Monte‑Carlo (ground truth)
- The Monte‑Carlo (MC) procedure is used as a ground-truth estimator for the expected immediate score change when calling `in_()` from the current state.
- It works by repeatedly sampling transitions from the current state: for each sample a deep copy of the game state is drawn (the code shuffles the copy's stack), the `in_()` operation is executed on the copy, and the observed score difference is recorded. Averaging these sampled score differences produces an empirical estimate of the expected value.
- This method requires the full `Flip7Game` object and is computationally expensive (cost scales linearly with the number of samples n) but is unbiased in the limit n → ∞.

Let $ s $ denote the current game state, and let $ p(c \mid s) $ be the probability distribution over cards $ c $ that can be drawn given $ s $.  
Define $ \Delta(c, s) $ as the score change resulting from drawing card $ c $ in state $ s $.  
The true expected immediate score change is:

$$
\mathbb{E}[\Delta \mid s] = \int p(c \mid s)\, \Delta(c, s)\, dc
$$

Since this integral is intractable in practice, we approximate it by Monte Carlo sampling.  
Drawing $ n $ independent samples $ c_1, \ldots, c_n \sim p(c \mid s) $, we obtain the unbiased estimator:

$$
\hat{\mathbb{E}}[\Delta \mid s] = \frac{1}{n} \sum_{i=1}^n \Delta(c_i, s)
$$

where each term  

$$
\Delta(c_i, s) = \text{score}(s \oplus c_i) - \text{score}(s)
$$

represents the simulated score difference when drawing $ c_i $ from the current state.
### Learned model
- The learned model (`Flip7Model` in `src/regressor.py`) is a small MLP trained to predict the same quantity that the MC procedure estimates: the expected immediate score change $\mathbb{E}[\Delta | s]$, but using a compact feature vector (counts of cards in hand ordered by `Flip7Game.CARD_TYPES`).
- Motivation: the learned model enables fast inference and lets you run a policy even when you do not have access to a `Flip7Game` simulator.
- Training: the model is trained using targets produced by the MC estimator (i.e., the dataset generation uses Monte‑Carlo averaging to compute labels), and the training loss in `src/train_regressor.py` uses Huber loss with the Adam optimizer.
- MC is generally more accurate (less biased) if you can afford many samples and have the simulator available; it converges to the true expected value as $n → ∞$.

### Practical recommendation
- Use MC wherever you can. The trained model was just an experiment and is less accurate.
- Use the learned model only when the game simulator is not accessible.

### Evaluation
- The `evaluate()` function runs a policy for many moves and returns per-round scores.
- Policies can be Monte‑Carlo agents, the learned model (used as a decision aid), or simple heuristics.

Project layout
--------------
- `src/flip7_game.py` — game logic and Monte‑Carlo simulator (`mc_sample_expected_score_difference_of_in`).
- `src/regressor.py` — PyTorch model architecture (a small MLP) that predicts the expected score change.
- `src/train_regressor.py` — dataset generation, training and evaluation utilities.
- `main.ipynb` — notebook for training, comparing and visualizing policies.
- `play.py` — small example / interactive script.
- `models/` — pretrained / saved models (e.g. `models/model.pt`).
- `requirements.txt` & `environment.yaml` — dependency manifests / Conda environment.

Quickstart
---------------------
Open and run `main.ipynb`. The notebook contains example workflows for dataset generation, model training, saving (`models/model.pt`) and comparisons between policies with visualizations.

A short script `play.py` is included in order to integrate the Monte Carlo simulator and learned model to live games.

Evaluate a policy (from the notebook or `src.train_regressor.evaluate`): call `evaluate(policy_fn)` with a policy function.

Files & API (quick reference)
-----------------------------
- `Flip7Game` (in `src/flip7_game.py`)
  - Methods: `reset()`, `in_()`, `get_score()`, `get_score_difference()`, `mc_sample_expected_score_difference_of_in(n)`
  - Attributes: `CARD_TYPES` (feature ordering), `card_counts_in_hand` (OrderedDict)

- Training / Data utilities (in `src/train_regressor.py`)
  - `create_dataset(dataset_size)`
  - `train(model, train_dataset, test_dataset, num_epochs, batch_size, lr)`
  - `evaluate(policy, num_moves)`

- Model
  - `Flip7Model` (in `src/regressor.py`): input-dim = `Flip7Game.NUM_CARDS`, output-dim = 1

License
-------
This project is licensed under the MIT License (see `LICENSE`).
