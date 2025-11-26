import pytest
import numpy as np
from unittest.mock import MagicMock

from agent.agent import Agent, State
from games.game_base import GameBase
from omnicron.manager import GameMemory
from simulation.simulation import (
    _apply_move, _simulate_game,
    randomly_assign_agent_pairs, _set_terminal, reset_agents
)


# ------------------------------
# Mock Game for deterministic testing
# ------------------------------
class MockGame(GameBase):
    def __init__(self, result_sequence=None):
        self.result_sequence = result_sequence or []
        self.applied_moves = []
        self.state_index = 0
        self._current_player = 0
        self._game_id = "mock_game"
        self._state = MagicMock()
        self._state.clone = lambda: f"clone_{self.state_index}"
        self._board = np.array([0])
        self._state.board = self._board
        self._state.current_player = self._current_player
        self._current_result = State.NEUTRAL

    def game_id(self) -> str:
        return self._game_id

    def clone(self) -> "GameBase":
        return self

    def deep_clone(self) -> "GameBase":
        return self

    def get_state(self):
        return self._state

    def set_state(self, game_state):
        self._state = game_state

    def current_player(self):
        return self._current_player

    def valid_moves(self):
        return [np.array([1])]

    def apply_move(self, move: np.ndarray):
        self.applied_moves.append(move)
        if self.state_index < len(self.result_sequence):
            self._current_result = self.result_sequence[self.state_index]
            self.state_index += 1
        else:
            self._current_result = State.NEUTRAL

    def is_over(self):
        return self._current_result in [State.WIN, State.LOSS, State.TIE]

    def get_result(self, agent_id: int):
        return self._current_result

    def state_string(self) -> str:
        return f"GameState {self.state_index}"


# ------------------------------
# Mock Memory (minimal, type-safe)
# ------------------------------
class MockMemory(GameMemory):
    def __init__(self):
        self.entries = []

    def write(self, game_id, game_state, outcome, move):
        player = getattr(game_state, "current_player", None)
        self.entries.append((game_state, move, outcome, player))


# ------------------------------
# Terminal Mock for testing end states
# ------------------------------
class TerminalMockGame(MockGame):
    """Single-move terminal game"""
    def __init__(self, final_result: State):
        super().__init__([final_result])
        self._current_result = final_result

    def is_over(self):
        return True


# ------------------------------
# Sequence Game to test neutral moves eventually credited
# ------------------------------
class SequenceGame(MockGame):
    """Game where first moves are neutral but last move is a win"""
    def __init__(self):
        super().__init__([State.NEUTRAL, State.NEUTRAL, State.WIN])
        self._index = 0

    def apply_move(self, move):
        self.applied_moves.append(move)
        if self._index < len(self.result_sequence):
            self._current_result = self.result_sequence[self._index]
            self._index += 1
        else:
            self._current_result = State.NEUTRAL


# ------------------------------
# Helper for monkeypatching _store_outcome_data
# ------------------------------
def mock_store_outcome(memory, agent, anti_agent):
    def _mock_store(game_id, agent_outcome, agent_stack, anti_agent_outcome, anti_agent_stack, omnicron):
        for move_arr, game_state_snapshot, depth in agent_stack:
            memory.entries.append((game_state_snapshot, move_arr, agent_outcome, agent.player_id, depth))
        for move_arr, game_state_snapshot, depth in anti_agent_stack:
            memory.entries.append((game_state_snapshot, move_arr, anti_agent_outcome, anti_agent.player_id, depth))
    return _mock_store


# ------------------------------
# Tests
# ------------------------------
def test_apply_move_terminal_updates():
    game = MockGame(result_sequence=[State.WIN])
    player = Agent(_player_id=0)
    opponent = Agent(_player_id=1)
    player.core_move = np.array([42])
    stack = []

    state = _apply_move(player, opponent, game, depth=1, player_stack=stack)

    assert state == State.WIN
    assert player.game_state == State.WIN
    assert opponent.game_state == State.LOSS
    assert player.change is False
    assert opponent.change is True
    assert np.array_equal(stack[0][0], player.move)
    assert stack[0][2] == 1


def test_simulate_game_terminates_at_terminal():
    game = MockGame(result_sequence=[State.NEUTRAL, State.TIE])
    agent = Agent(_player_id=0)
    anti_agent = Agent(_player_id=1)
    memory = MockMemory()

    _simulate_game(agent, anti_agent, game, turn_depth=10, omnicron=memory, is_prune_stage=True)

    final_result = game.get_result(agent.player_id)
    assert final_result in [State.WIN, State.LOSS, State.TIE]


def test_no_false_losses_logged(monkeypatch):
    game = MockGame(result_sequence=[State.NEUTRAL, State.TIE])
    agent = Agent(_player_id=0)
    anti_agent = Agent(_player_id=1)
    memory = MockMemory()
    monkeypatch.setattr("simulation.simulation._store_outcome_data", mock_store_outcome(memory, agent, anti_agent))
    _simulate_game(agent, anti_agent, game, turn_depth=10, omnicron=memory, is_prune_stage=True)

    agent_entries = [e for e in memory.entries if e[3] == agent.player_id]
    for _, _, outcome, _, _ in agent_entries:
        assert outcome != State.LOSS


def test_set_terminal_mapping():
    p = Agent()
    o = Agent()
    _set_terminal(p, o, State.WIN)
    assert p.game_state == State.WIN
    assert o.game_state == State.LOSS
    assert p.change is False
    assert o.change is True

    _set_terminal(p, o, State.TIE)
    assert p.game_state == State.TIE
    assert o.game_state == State.TIE
    assert p.change is True
    assert o.change is True


def test_randomly_assign_agent_pairs_limits():
    agents = [Agent() for _ in range(5)]
    anti_agents = [Agent() for _ in range(3)]
    pairs = randomly_assign_agent_pairs(agents, anti_agents, n=2)
    assert len(pairs) == 2
    for i, j in pairs:
        assert 0 <= i < len(agents)
        assert 0 <= j < len(anti_agents)


def test_reset_agents_clears_state():
    a = Agent()
    a.core_move = np.array([1])
    a.move = np.array([2])
    a.change = True
    a.game_state = State.WIN
    a.move_depth = 5

    reset_agents([a])
    assert a.core_move.size == 0
    assert a.move.size == 0
    assert a.change is False
    assert a.game_state == State.NEUTRAL
    assert a.move_depth == 0


def test_simulate_game_depth_tracking(monkeypatch):
    game = MockGame(result_sequence=[State.NEUTRAL, State.NEUTRAL, State.WIN])
    agent = Agent(_player_id=0)
    anti_agent = Agent(_player_id=1)
    memory = MockMemory()
    monkeypatch.setattr("simulation.simulation._store_outcome_data", mock_store_outcome(memory, agent, anti_agent))

    _simulate_game(agent, anti_agent, game, turn_depth=5, omnicron=memory, is_prune_stage=True)
    depths = [entry[4] for entry in memory.entries if entry[3] == agent.player_id]
    assert len(depths) > 0
    assert all(d > 0 for d in depths)
    assert max(depths) <= 5


def test_simulate_game_invalid_turn_depth():
    game = MockGame()
    agent = Agent()
    anti_agent = Agent()
    memory = MockMemory()

    with pytest.raises(ValueError):
        _simulate_game(agent, anti_agent, game, turn_depth=0, omnicron=memory, is_prune_stage=True)


def test_early_neutral_moves_eventually_credit_win(monkeypatch):
    agent = Agent(_player_id=0)
    agent.core_move = np.array([1])
    opponent = Agent(_player_id=1)
    memory = MockMemory()
    monkeypatch.setattr("simulation.simulation._store_outcome_data", mock_store_outcome(memory, agent, opponent))

    game = SequenceGame()
    _simulate_game(agent, opponent, game, turn_depth=5, omnicron=memory, is_prune_stage=True)

    agent_entries = [e for e in memory.entries if e[3] == agent.player_id]
    assert len(agent_entries) > 0
    for _, _, outcome, _, _ in agent_entries:
        assert outcome == State.WIN
