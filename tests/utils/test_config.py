"""Tests for wise_explorer.utils.config — game registry, paths, defaults."""

from wise_explorer.utils.config import (
    GAMES, INITIAL_STATES, TURN_DEPTHS,
    default_turn_depth, default_ponder,
)


class TestGameRegistry:
    def test_not_empty(self):
        assert len(GAMES) > 0

    def test_game_classes_callable(self):
        for _name, game_class in GAMES.items():
            game = game_class()
            assert game is not None
            assert hasattr(game, "game_id")


class TestInitialStates:
    def test_matches_games(self):
        for game_name in GAMES:
            assert game_name in INITIAL_STATES

    def test_have_board_and_player(self):
        for _name, state in INITIAL_STATES.items():
            assert hasattr(state, "board")
            assert state.current_player == 1


class TestDefaults:
    def test_turn_depth_known_game(self):
        for name in GAMES:
            assert default_turn_depth(name) == TURN_DEPTHS.get(name, 40)

    def test_turn_depth_unknown_game(self):
        assert default_turn_depth("unknown") == 40

    def test_ponder_positive(self):
        for name in GAMES:
            assert default_ponder(name) > 0

    def test_ponder_unknown_game(self):
        assert default_ponder("unknown") > 0
