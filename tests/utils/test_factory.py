"""
Tests for wise_explorer.utils.factory

Tests factory functions for creating games and agent swarms.
"""

import pytest

from wise_explorer.agent.agent import Agent
from wise_explorer.games.game_base import GameBase
from wise_explorer.utils.config import GAMES
from wise_explorer.utils.factory import create_agent_swarms, create_game


class TestCreateAgentSwarms:
    """create_agent_swarms function tests."""

    def test_creates_correct_structure(self):
        """Creates dict mapping player IDs to agent lists."""
        swarms = create_agent_swarms([1, 2], agents_per_player=5)
        
        assert isinstance(swarms, dict)
        assert 1 in swarms and 2 in swarms
        assert len(swarms[1]) == 5
        assert len(swarms[2]) == 5

    def test_agents_have_correct_player_id(self):
        """Agents have correct player_id."""
        swarms = create_agent_swarms([1, 2], agents_per_player=3)
        
        for agent in swarms[1]:
            assert agent.player_id == 1
        for agent in swarms[2]:
            assert agent.player_id == 2

    def test_agents_are_agent_instances(self):
        """Created agents are Agent instances."""
        swarms = create_agent_swarms([1], agents_per_player=3)
        
        for agent in swarms[1]:
            assert isinstance(agent, Agent)

    def test_agents_are_independent(self):
        """Each agent is a separate instance."""
        swarms = create_agent_swarms([1], agents_per_player=3)
        agents = swarms[1]
        
        assert agents[0] is not agents[1]
        
        agents[0].move_depth = 100
        assert agents[1].move_depth != 100

    def test_zero_agents(self):
        """Works with zero agents per player."""
        swarms = create_agent_swarms([1, 2], agents_per_player=0)
        assert len(swarms[1]) == 0

    def test_empty_players(self):
        """Works with empty players list."""
        swarms = create_agent_swarms([], agents_per_player=5)
        assert len(swarms) == 0


class TestCreateGame:
    """create_game function tests."""

    def test_creates_game_instance(self):
        """Creates a GameBase instance for any registered game."""
        for game_name in GAMES:
            game = create_game(game_name)
            assert isinstance(game, GameBase)

    def test_game_has_initial_state(self):
        """Created game has initial state set."""
        for game_name in GAMES:
            game = create_game(game_name)
            state = game.get_state()
            
            assert state is not None
            assert state.board is not None
            assert state.current_player == 1

    def test_game_is_playable(self):
        """Created game is in playable state."""
        for game_name in GAMES:
            game = create_game(game_name)
            
            moves = game.valid_moves()
            assert len(moves) > 0

    def test_unknown_game_raises(self):
        """Unknown game name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown game"):
            create_game("unknown_game")

    def test_error_lists_available_games(self):
        """Error message lists available games."""
        try:
            create_game("nonexistent")
        except ValueError as e:
            for game_name in GAMES:
                assert game_name in str(e)


class TestStateIndependence:
    """Game state independence tests."""

    def test_multiple_games_independent(self):
        """Multiple created games have independent state."""
        
        for game_name in GAMES:
            game1 = create_game(game_name)
            game2 = create_game(game_name)
            
            # Modify game1
            move = game1.valid_moves()[0]
            game1.apply_move(move)
            
            # game2 should be unchanged (player still 1)
            assert game2.current_player() == 1