"""
Tests for wise_explorer.utils.config

Tests configuration and game registry.
"""

import pytest

from wise_explorer.utils.config import (
    GAMES, INITIAL_STATES, Config,
    _round_epochs_for_clean_split,
)


class TestGameRegistry:
    """GAMES registry tests."""

    def test_not_empty(self):
        """GAMES registry is not empty."""
        assert len(GAMES) > 0

    def test_game_classes_callable(self):
        """Registered game classes can be instantiated."""
        for name, game_class in GAMES.items():
            game = game_class()
            assert game is not None
            assert hasattr(game, 'game_id')


class TestInitialStates:
    """INITIAL_STATES registry tests."""

    def test_matches_games(self):
        """INITIAL_STATES has entry for each game."""
        for game_name in GAMES:
            assert game_name in INITIAL_STATES

    def test_have_board_and_player(self):
        """Initial states have board and current_player."""
        for name, state in INITIAL_STATES.items():
            assert hasattr(state, 'board')
            assert hasattr(state, 'current_player')
            assert state.current_player == 1


class TestRoundEpochsForCleanSplit:
    """_round_epochs_for_clean_split function tests."""

    def test_already_divisible(self):
        """Returns same value if already divisible."""
        # For 2 players, divisor is 3
        assert _round_epochs_for_clean_split(9, 2) == 9

    def test_rounds_up(self):
        """Rounds up to next divisible value."""
        assert _round_epochs_for_clean_split(10, 2) == 12

    def test_single_player(self):
        """Works with single player (divisor = 2)."""
        assert _round_epochs_for_clean_split(7, 1) == 8

    def test_zero_epochs(self):
        """Works with zero epochs."""
        assert _round_epochs_for_clean_split(0, 2) == 0


class TestConfig:
    """Config class tests."""

    def test_default_config(self):
        """Config can be created with defaults."""
        config = Config()
        assert config.epochs > 0
        assert config.turn_depth > 0
        assert config.num_workers >= 1

    def test_custom_epochs(self):
        """Custom epochs are set (may be rounded)."""
        config = Config(epochs=500)
        assert config.epochs >= 500

    def test_custom_turn_depth(self):
        """Custom turn_depth is set."""
        config = Config(turn_depth=50)
        assert config.turn_depth == 50

    def test_custom_workers(self):
        """Custom num_workers is set."""
        config = Config(num_workers=4)
        assert config.num_workers == 4

    def test_derived_values(self):
        """Derived values are calculated."""
        config = Config(num_workers=4)
        assert config.num_agents == 4
        assert config.simulations == config.epochs * config.num_agents

    def test_unknown_game_raises(self):
        """Unknown game name raises KeyError."""
        with pytest.raises(KeyError):
            Config(game_name="unknown_game")

    def test_epochs_rounded_for_2_player(self):
        """Epochs are rounded for clean split (2-player: divisible by 3)."""
        # Get a 2-player game
        for name, game_class in GAMES.items():
            if game_class().num_players() == 2:
                config = Config(game_name=name, epochs=100)
                assert config.epochs % 3 == 0
                break