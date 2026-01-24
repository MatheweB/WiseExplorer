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

### Troubleshooting Installation

If you encounter an import error like `ImportError: cannot import name 'main' from 'main'`, this is due to a naming conflict with Anaconda/Conda. To fix:

```bash
pip uninstall wise-explorer
pip install -e .
```

If the issue persists, ensure you don't have conflicting packages installed in your environment.

## Quick Start

### Using as a Library (Recommended)

```python
from omnicron.manager import GameMemory
from simulation.simulation import start_simulations
from agent.agent import Agent
from games.tic_tac_toe import TicTacToe

# Initialize your game (must implement GameBase interface)
game = TicTacToe()

# Create agent swarms (one per player)
swarms = {
    1: [Agent() for _ in range(20)],
    2: [Agent() for _ in range(20)],
}

# Create memory database
memory = GameMemory.for_game(game)

# Train and play
start_simulations(
    agent_swarms=swarms,
    game=game,
    turn_depth=20,       # Maximum turns per training game
    simulations=200,     # Training games per move
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

| Phase   | Agent Behavior            | Purpose                              |
|---------|---------------------------|--------------------------------------|
| **Prune**   | Intentionally play worst moves | Identify and confirm losing patterns |
| **Exploit** | Always play best moves         | Reinforce winning strategies         |

By systematically exploring both extremes, Wise Explorer rapidly builds a comprehensive understanding of the game space.

## Project Structure

```
src/
├── run_wise_explorer.py                    # Main entry point
├── pyproject.toml            # Project configuration
├── pytest.ini                # Test configuration
├── agent/
│   └── agent.py              # Agent implementation
├── data/
│   └── memory/               # SQLite databases
├── games/
│   ├── game_base.py          # Game interface
│   ├── game_rules.py         # Rule definitions
│   ├── game_state.py         # State representation
│   ├── minichess.py          # Example: Chess variant
│   └── tic_tac_toe.py        # Example: Tic-tac-toe
├── omnicron/
│   ├── manager.py            # GameMemory orchestration
│   ├── serializers.py        # State serialization
│   └── transition.py         # Transition records
├── simulation/
│   └── simulation.py         # Training loop
├── utils/
│   └── global_variables.py   # Configuration
└── wise_explorer/
    └── wise_explorer_algorithm.py  # Core algorithm
```

## Key Concepts

### GameMemory

The knowledge database that records transitions, manages anchors, and scores moves.

```python
# Initialize memory for a game
memory = GameMemory.for_game(game)

# Record game outcomes
memory.record_round(GameClass, [
    (player1_moves, State.WIN),
    (player2_moves, State.LOSS),
])

# Query move quality (returns 0.0 to 1.0)
score = memory.get_move_score(from_hash, to_hash)
```

### Agent Strategy

Agents operate in one of two modes:

- **Prune mode** (`is_prune=True`): Deliberately makes poor moves to map out losing strategies
- **Exploit mode** (`is_prune=False`): Selects the highest-scoring moves

The `simulation.py` module implements round-robin pruning, ensuring each player systematically explores bad moves during training.

### Markov vs Non-Markov Modes

Choose how the algorithm treats game states:

```python
# Non-Markov mode (default): Each unique transition is distinct. The acting player is implicitly encoded.
memory = GameMemory.for_game(game, markov=False)

# Markov mode: Only the resulting position matters. The acting player is not encoded.
memory = GameMemory.for_game(game, markov=True)
```

**Markov mode** treats positions as equivalent regardless of how they were reached. **Non-Markov mode** considers the path taken, useful for games where move history affects optimal play.

## Implementing a Custom Game

Create a class that implements the `GameBase` interface:

```python
from games.game_base import GameBase
from games.game_state import GameState, State
import numpy as np

class MyGame(GameBase):
    def get_state(self) -> GameState:
        """Return current board state and active player."""
        pass
        
    def set_state(self, state: GameState) -> None:
        """Set the board state and active player."""
        pass
        
    def valid_moves(self) -> np.ndarray:
        """Return array of all legal moves in current position."""
        pass
        
    def apply_move(self, move: np.ndarray) -> None:
        """Execute the move and update game state."""
        pass
        
    def is_over(self) -> bool:
        """Check if the game has ended."""
        pass
        
    def get_result(self, player: int) -> State:
        """Return WIN/LOSS/TIE/NEUTRAL for the specified player."""
        pass
        
    def current_player(self) -> int:
        """Return the active player's identifier."""
        pass
        
    def deep_clone(self) -> "MyGame":
        """Create an independent copy of the game state."""
        pass
```

See `games/tic_tac_toe.py` and `games/minichess.py` for complete examples.

## Configuration

### Environment Variables

No environment variables are required. Modify `utils/global_variables.py` for custom settings.

### Database Storage

Wise Explorer uses SQLite for persistence, automatically created at first run.

```python
# Default location: data/memory/{game_id}.db
memory = GameMemory.for_game(game)

# Custom database path
memory = GameMemory("path/to/my_game.db")

# Read-only mode (useful for parallel workers)
memory = GameMemory("path/to/my_game.db", read_only=True)
```

## Performance Characteristics

- **Training throughput**: 1,000–5,000 games/second (varies by game complexity)
- **Move selection latency**: <1ms for cached positions
- **Storage efficiency**: ~1KB per 100 unique transitions

Performance scales well with multiprocessing for independent game simulations.

## Testing

Run the test suite to verify your installation:

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests to ensure nothing breaks (`pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

For bug reports and feature requests, please open an issue on GitHub.

## Citation

If you use Wise Explorer in your research, please cite:

```bibtex
@thesis{wise_explorer,
  title={General Game Playing as a Bandit-Arms Problem: A Multiagent Monte-Carlo Solution Exploiting Nash Equilibria},
  author={[Mathewe Banda]},
  year={[2019]},
  school={Oberlin College},
  url={https://digitalcommons.oberlin.edu/honors/116/}
}
```

## Support

- **Documentation**: [Research Paper](https://digitalcommons.oberlin.edu/honors/116/)
- **Issues**: [GitHub Issues](https://github.com/MatheweB/WiseExplorer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MatheweB/WiseExplorer/discussions)

---

Built with curiosity and a willingness to take the path less traveled by