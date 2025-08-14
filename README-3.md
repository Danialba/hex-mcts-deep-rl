# On-Policy MCTS + Deep RL for Hex (with Tic-Tac-Toe baseline)

Train a Hex-playing agent by combining **Monte Carlo Tree Search (MCTS)** with a **PyTorch policy network** (ANET). The MCTS visit counts become supervised targets for the actor network, yielding an **on-policy** training loop in the spirit of AlphaZero. The code is modular (game logic, tree search, network, simulation) and configuration-driven.

> Supports both **Hex** (primary) and **Tic-Tac-Toe** (for quick debugging), plus head-to-head play and a mini **Tournament of Progressive Policies (TOPP)**.

---

## ✨ Features

- **On-policy MCTS**: tree policy for search; the same policy network drives rollouts.
- **PyTorch ANET**: configurable hidden layers, optimizer, learning rate; trained with cross-entropy on MCTS visit distributions.
- **Replay buffer** for supervised updates at the end of each episode.
- **Hex engine** with fast win-detection via disjoint sets / union-find.
- **Multiple modes** via `config.py`: `train`, `topp`, `custom_game`, `oht` (online-tournament client mode stub).
- **Baselines**: random agent and greedy options for quick comparisons.
- **Tic-Tac-Toe** mode for smoke-tests and algorithm sanity checks.

---

## 🧱 Project Structure

```
.
├── anet.py                 # PyTorch actor network (policy)
├── config.py               # All runtime/config params and mode switches
├── game_simulator.py       # Orchestrates episodes, rollouts, training
├── hex.py                  # Hex game rules + win detection (union-find)
├── intelligent_agent.py    # Wraps ANET for acting (greedy/epsilon-greedy)
├── main.py                 # Entry point (dispatches by mode in config)
├── mctsnode.py             # MCTS node/search logic
├── randomagent.py          # Baseline random policy
├── tic_tac_toe.py          # Simple game for debugging/validation
└── (optional) topp.py, rbuf.py, models/, logs/, etc.
```

> Note: The repository intentionally does **not** include any course PDFs; this README stands alone.

---

## ⚙️ Requirements

- Python 3.10+
- PyTorch (2.0+ recommended)
- `disjoint-set` (for union-find in Hex)
- (Common stdlib only elsewhere)

Quick install:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or CUDA build
pip install disjoint-set
```

> If you prefer, add a `requirements.txt` with:
> ```
> torch>=2.0
> disjoint-set>=1.0
> ```

---

## 🚀 Quick Start

1) **Choose your mode** in `config.py`:

```python
run = {
  "train": True,
  "topp":  False,
  "custom_game": False,
  "oht": False,
}

game_config = {
  "game": "hex",          # "hex" | "tic_tac_toe"
  "size": 4,              # board size k (Hex supports 3–10)
  "starting_player": 1,   # 1 or -1
  "show_every_move": True # visualize the real (non-rollout) game
}

# ... training, ANET, TOPP, and custom-game configs live here too
```

2) **Run**:

```bash
python main.py
```

- `train=True` → runs the full self-play + MCTS + supervised update loop for the specified number of episodes, periodically saving policy snapshots (for TOPP).
- `topp=True`  → loads saved ANET snapshots and runs a round-robin mini-tournament.
- `custom_game=True` → pits any two agents (random/model/greedy) for N games.
- `oht=True`   → client mode stub for connecting a trained agent to an online Hex server (left opt-in).

---

## 🧠 Method (high level)

1. **Self-Play Episodes**
   - For each real move: run many MCTS simulations from the current state.
   - Use ANET to drive rollout actions from leaves to terminal states.
   - Backpropagate outcomes to update visit counts.

2. **Targets for ANET**
   - Normalize visit counts at the root to a probability vector over legal moves.
   - Store `(state, target_distribution)` pairs in a **replay buffer**.

3. **Supervised Update**
   - After each episode, sample minibatches and minimize cross-entropy between ANET’s softmax logits and the MCTS targets.
   - Periodically **cache** ANET checkpoints (for TOPP).

4. **Evaluation**
   - **TOPP**: round-robin matches among saved policies (progress over training).
   - **Baselines**: vs. random / greedy agents.

---

## 🛠️ Key Configuration Knobs

All in `config.py`:

- **Game**
  - `game_config.size` — Hex board size (e.g., 4, 5, 7).
  - `starting_player`, `show_every_move`.
- **Training**
  - Episodes, MCTS simulations per move, replay-buffer size, batch size.
  - Epsilon-greedy parameters for rollouts.
- **ANET**
  - Input/Output sizes (derived from board), hidden layers, activations.
  - Optimizer (`Adam`, `RMSprop`, `SGD`, `Adagrad`) and `lr`.
  - Checkpoint cadence & save paths.
- **TOPP**
  - Which checkpoints to load, games per pairing, deterministic vs. stochastic action selection.
- **Custom Game**
  - Agent choices (e.g., `"Random"` or model path `*.pth`), greedy flags, number of games.

> Tip: Start with a small Hex board (4×4 or 5×5) to validate learning quickly.

---

## 📊 Example Workflows

### Train on 5×5 Hex
```python
# config.py
run["train"] = True
game_config["game"] = "hex"
game_config["size"] = 5
# set training[...] and anet_config[...] as desired
```
```bash
python main.py
```
Checkpoints will be saved periodically (e.g., `models/model_<episode>_anet.pth`).

### Play a Head-to-Head (model vs random)
```python
# config.py
run = {"train": False, "topp": False, "custom_game": True, "oht": False}
custom_game = {
  "player1": "Random",
  "player1_greedy": False,
  "player2": "Model_5x5_for_demo/model_200_anet.pth",
  "player2_greedy": True,
  "number_of_games": 50,
  "alt_starting_player": True
}
```
```bash
python main.py
```

### Run a Mini-TOPP
```python
# config.py
run = {"train": False, "topp": True, "custom_game": False, "oht": False}
# point TOPP to a set of saved ANET checkpoints
```
```bash
python main.py
```

---

## 🧩 Notes on Implementation

- **Hex** uses a union-find (disjoint set) structure to maintain connected components and check path completion efficiently.
- **ANET** is a standard feed-forward classifier whose output dimension equals the number of possible board cells. Illegal moves are masked and the distribution renormalized before sampling/argmax during rollouts and play.
- **Loss**: cross-entropy on the visit-count distribution; **optimizer** is configurable, with `Adam` as a sensible default.
- **Rollouts**: use the current ANET, optionally with epsilon-greedy exploration.
- Optional **critic/value head** can be bolted on later (not required); pure rollouts are supported and encouraged for clarity.

---

## 🧪 Testing / Debugging

- Switch to `game="tic_tac_toe"` to validate MCTS/ANET plumbing on a tiny state space.
- Turn on `show_every_move=True` to watch games unfold (real moves only; rollouts are not visualized).
- Start with few MCTS simulations for speed, then scale up.

---

## 📦 Checkpoints & Repro

- Saved policies are plain PyTorch `state_dict`s (`*.pth`).
- You can mix deterministic (greedy) and stochastic action selection at evaluation time via config flags.

---

## 🙏 Acknowledgements

- Inspired by the AlphaZero family of methods and academic coursework on MCTS + Deep RL.
- Hex rules and win-checking leverage a union-find approach for performance.

---

## 📄 License

MIT — see `LICENSE`.
