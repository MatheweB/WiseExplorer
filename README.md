# Wise Explorer

A pattern-based learning model for General Game Playing that explores both promising and unpromising paths to master any N-player game.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Wise Explorer takes a unique approach to Monte Carlo Tree Search (MCTS) by deliberately exploring not just the most promising branches, but also the *least* promising ones. This dual-exploration strategy enables rapid pattern recognition without heuristics, training data, or game-specific knowledge.

**Key Features:**
- **Zero prior knowledge required** — no heuristics, no training data
- **Universal compatibility** — works with any N-player game
- **Dual-phase learning** — systematically finds bad moves and reinforces good ones
- **Statistical clustering** — borrows knowledge from similar positions
- **Bayesian confidence scoring** — conservative estimates that improve with data

[Read the research paper](https://digitalcommons.oberlin.edu/honors/116/) for full technical details.

## A Note

This project is a re-imagining and re-implementation of my Honors thesis. Many strides have been made in efficiency and applicability that I wasn't able to achieve in 2019, and new concepts such as "anchors" and probability distribution sampling are the result of my independent learning and research.

## Installation

```bash
git clone https://github.com/MatheweB/WiseExplorer
cd WiseExplorer
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Play Tic-Tac-Toe against the AI (default)
wise-explorer

# Play Mini Chess with more training
wise-explorer --game minichess --epochs 200

# Watch AI play itself
wise-explorer --ai-only

# Use existing knowledge without training
wise-explorer --no-training

# See all options
wise-explorer --help
```

### CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `--game` | `-g` | Game to play: `tic_tac_toe`, `minichess` |
| `--epochs` | `-e` | Training epochs per move (default: 100) |
| `--turn-depth` | `-t` | Max turns per simulation (default: 40) |
| `--workers` | `-w` | Parallel worker processes (default: CPU count - 1) |
| `--no-training` | | Play using existing memory only |
| `--ai-only` | | AI plays both sides |

### As a Library

```python
from wise_explorer.api import start_simulations
from wise_explorer.memory import GameMemory
from wise_explorer.agent.agent import Agent
from wise_explorer.games import TicTacToe
from wise_explorer.utils.factory import create_agent_swarms

# Initialize game
game = TicTacToe()

# Create agent swarms (one per player)
players = [1, 2]
swarms = create_agent_swarms(players, agents_per_player=20)

# Create memory database
memory = GameMemory.for_game(game)

# Train and play
start_simulations(
    agent_swarms=swarms,
    game=game,
    turn_depth=20,
    simulations=200,
    memory=memory,
)
```

## How It Works

### Transitions Over Positions

Unlike traditional approaches that evaluate board positions, Wise Explorer records **transitions**:

```
state_before → state_after → outcome
```

The outcome (win/tie/loss) is always stored from the perspective of the player who made the move, eliminating the need to track player identities across different game states.

### Statistical Clustering with Anchors

Positions with similar win rates are automatically grouped into "anchors." When encountering a position with limited data, the system intelligently borrows statistics from similar, well-explored positions.

```
Position A: 3 games, 67% wins  ─┐
Position B: 5 games, 60% wins  ─┼─► Anchor: 18 games, 61% wins
Position C: 10 games, 60% wins ─┘
```

This clustering accelerates learning by sharing knowledge across related game states.

### Bayesian Confidence Scoring

Move quality is computed using a lower confidence bound approach:

- **Few samples** → high uncertainty → conservative score near 0.5
- **Many samples** → high confidence → score reflects true win rate
- **Pessimistic default** → assumes worst case within confidence bounds

This prevents the algorithm from overcommitting to moves based on lucky early results.

### Dual-Phase Training

Training alternates between two complementary strategies:

| Phase | Agent Behavior | Purpose |
|-------|----------------|---------|
| **Prune** | Intentionally play worst moves | Identify and confirm losing patterns |
| **Exploit** | Always play best moves | Reinforce winning strategies |

By systematically exploring both extremes, Wise Explorer rapidly builds a comprehensive understanding of the game space.

## Project Structure

```
src/wise_explorer/
├── cli.py                 # Command-line interface
├── api.py                 # Public API (start_simulations)
│
├── agent/
│   └── __init__.py        # Agent class and State enum
│
├── core/
│   ├── types.py           # Stats, scoring constants
│   ├── hashing.py         # Board state hashing
│   └── bayes.py           # Bayes factor clustering
│
├── games/
│   ├── game_base.py       # Abstract game interface
│   ├── game_state.py      # GameState container
│   ├── game_rules.py      # Board utilities
│   ├── tic_tac_toe.py     # Tic-Tac-Toe implementation
│   └── minichess.py       # Mini Chess implementation
│
├── memory/
│   ├── schema.py          # Database schema
│   ├── anchor_manager.py  # Anchor logic class
│   └── game_memory.py     # GameMemory class
│
├── selection/
│   ├── training.py        # Probabilistic selection (exploration)
│   └── inference.py       # Deterministic selection (exploitation)
│
├── simulation/
│   ├── jobs.py            # Job data structures
│   ├── worker.py          # Parallel worker logic
│   ├── runner.py          # SimulationRunner
│   └── training.py        # Training orchestration
│
├── utils/
│   ├── config.py          # Game registry and Config class
│   └── factory.py         # Factory functions
│
├── data/memory/           # SQLite databases (auto-created)
│
└── debug/
    └── viz.py             # Terminal visualization

tests/                     # Test suite (mirrors src structure)

```

## Key Concepts

### GameMemory

The knowledge database that records transitions, manages anchors, and scores moves.

```python
from wise_explorer.memory import GameMemory

# Initialize memory for a game
memory = GameMemory.for_game(game)

# Custom database path
memory = GameMemory("path/to/my_game.db")

# Read-only mode (for parallel workers)
memory = GameMemory("path/to/my_game.db", read_only=True)

# Query statistics
info = memory.get_info()
print(f"Transitions: {info['transitions']}, Anchors: {info['anchors']}")
```

### Agent Strategy

Agents operate in one of two modes:

- **Prune mode** (`is_prune=True`): Deliberately makes poor moves to map out losing strategies
- **Exploit mode** (`is_prune=False`): Selects the highest-scoring moves

The training module implements round-robin pruning, ensuring each player systematically explores bad moves during training.

### Markov vs Non-Markov Modes

Choose how the algorithm treats game states:

```python
# Non-Markov (default): Each transition is distinct
memory = GameMemory.for_game(game, markov=False)

# Markov: Only the resulting position matters
memory = GameMemory.for_game(game, markov=True)
```

**Markov mode** treats positions as equivalent regardless of how they were reached. **Non-Markov mode** considers the path taken, useful for games where move history affects optimal play.

## Implementing a Custom Game

Create a class that implements the `GameBase` interface:

```python
from wise_explorer.games import GameBase, GameState
from wise_explorer.agent.agent import State
import numpy as np

class MyGame(GameBase):
    def game_id(self) -> str:
        return "my_game"
    
    def num_players(self) -> int:
        return 2
    
    def get_state(self) -> GameState:
        """Return current board state and active player."""
        pass
        
    def set_state(self, state: GameState) -> None:
        """Set the board state and active player."""
        pass
        
    def valid_moves(self) -> np.ndarray:
        """Return array of all legal moves."""
        pass
        
    def apply_move(self, move: np.ndarray) -> None:
        """Execute move and update game state."""
        pass
        
    def is_over(self) -> bool:
        """Check if the game has ended."""
        pass
        
    def get_result(self, player: int) -> State:
        """Return WIN/LOSS/TIE/NEUTRAL for the player."""
        pass
        
    def current_player(self) -> int:
        """Return the active player's ID."""
        pass
        
    def deep_clone(self) -> "MyGame":
        """Create an independent copy."""
        pass
    
    def clone(self) -> "MyGame":
        """Shallow copy."""
        pass
    
    def state_string(self) -> str:
        """Pretty-print for debugging."""
        pass
```

Then register it in `utils/config.py`:

```python
from wise_explorer.games.my_game import MyGame

GAMES = {
    "tic_tac_toe": TicTacToe,
    "minichess": MiniChess,
    "my_game": MyGame,  # Add your game
}

INITIAL_STATES = {
    # ... add initial state for your game
}
```

See `games/tic_tac_toe.py` and `games/minichess.py` for complete examples.

## Testing

```bash
pytest                  # Run all tests
pytest tests/core/      # Run core module tests
pytest -v               # Verbose output
```

## Performance

- **Training throughput**: 1,000–5,000 games/second (varies by game complexity)
- **Move selection latency**: <1ms for cached positions
- **Storage efficiency**: ~1KB per 100 unique transitions

Performance scales well with multiprocessing for independent game simulations.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use Wise Explorer in your research, please cite:

```bibtex
@thesis{wise_explorer,
  title={General Game Playing as a Bandit-Arms Problem: A Multiagent Monte-Carlo Solution Exploiting Nash Equilibria},
  author={Mathewe Banda},
  year={2019},
  school={Oberlin College},
  url={https://digitalcommons.oberlin.edu/honors/116/}
}
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

For bug reports and feature requests, please [open an issue](https://github.com/MatheweB/WiseExplorer/issues).

---

Built with curiosity and a willingness to take the path less traveled by