import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from agent.agent import Agent, State


def test_default_initialization():
    agent = Agent()

    assert isinstance(agent.core_move, np.ndarray)
    assert agent.core_move.size == 0

    assert isinstance(agent.move, np.ndarray)
    assert agent.move.size == 0

    assert agent.change is False
    assert agent.game_state == State.NEUTRAL
    assert agent.move_depth == 0
    assert agent.player_id == 0


def test_setters_and_getters_for_moves():
    agent = Agent()

    move = np.array([1, 2, 3])
    core_move = np.array([4, 5, 6])

    agent.move = move
    agent.core_move = core_move

    assert np.array_equal(agent.move, move)
    assert np.array_equal(agent.core_move, core_move)


def test_change_flag():
    agent = Agent()

    agent.change = True
    assert agent.change is True

    agent.change = False
    assert agent.change is False


def test_game_state_assignment():
    agent = Agent()

    agent.game_state = State.WIN
    assert agent.game_state == State.WIN

    agent.game_state = State.LOSS
    assert agent.game_state == State.LOSS


def test_move_depth_assignment():
    agent = Agent()

    agent.move_depth = 5
    assert agent.move_depth == 5

    agent.move_depth = 0
    assert agent.move_depth == 0


def test_player_id_assignment():
    agent = Agent()

    agent.player_id = 7
    assert agent.player_id == 7

    agent.player_id = -1
    assert agent.player_id == -1
