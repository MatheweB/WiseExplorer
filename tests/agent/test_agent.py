"""
Tests for wise_explorer.agent.agent

Tests Agent dataclass and State enum.
"""

import numpy as np
import pytest

from wise_explorer.agent.agent import Agent, State


class TestState:
    """State enum tests."""

    def test_all_states_exist(self):
        """All expected states are defined."""
        assert all(hasattr(State, s) for s in ['WIN', 'TIE', 'LOSS', 'NEUTRAL'])

    def test_states_are_distinct(self):
        """Each state has a unique value."""
        values = [State.WIN.value, State.TIE.value, State.LOSS.value, State.NEUTRAL.value]
        assert len(values) == len(set(values))


class TestAgentDefaults:
    """Agent default initialization tests."""

    def test_defaults(self):
        """Default values are correct."""
        agent = Agent()
        assert agent.core_move.size == 0
        assert agent.move.size == 0
        assert agent.change is False
        assert agent.game_state == State.NEUTRAL
        assert agent.move_depth == 0
        assert agent.player_id == 0


class TestAgentProperties:
    """Agent property getter/setter tests."""

    def test_move_property(self):
        """Move property works correctly."""
        agent = Agent()
        move = np.array([1, 2, 3])
        agent.move = move
        np.testing.assert_array_equal(agent.move, move)

    def test_core_move_property(self):
        """Core move property works correctly."""
        agent = Agent()
        core_move = np.array([0, 0])
        agent.core_move = core_move
        np.testing.assert_array_equal(agent.core_move, core_move)

    def test_game_state_property(self):
        """Game state property cycles through all states."""
        agent = Agent()
        for state in State:
            agent.game_state = state
            assert agent.game_state == state

    def test_numeric_properties(self):
        """Numeric properties work correctly."""
        agent = Agent()
        agent.move_depth = 42
        agent.player_id = 2
        assert agent.move_depth == 42
        assert agent.player_id == 2


class TestAgentIndependence:
    """Tests that Agent instances are independent."""

    def test_multiple_agents_independent(self):
        """Multiple Agent instances don't share state."""
        agent1 = Agent()
        agent2 = Agent()
        
        agent1.player_id = 1
        agent1.game_state = State.WIN
        
        agent2.player_id = 2
        agent2.game_state = State.LOSS
        
        assert agent1.player_id != agent2.player_id
        assert agent1.game_state != agent2.game_state