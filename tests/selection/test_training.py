"""
Tests for training move selection (Scheme 2: uncertainty-weighted value
distribution).

``move_weight`` is the core logic and is tested as a pure function;
``select_move_for_training`` is covered by an integration smoke + reproducibility
test against a real memory.
"""

import random

import numpy as np

import wise_explorer.memory as M
from wise_explorer.core.types import Stats
from wise_explorer.games.tic_tac_toe import TicTacToe
from wise_explorer.selection import select_move_for_training
from wise_explorer.selection.training import move_weight


class TestMoveWeight:
    """The Scheme 2 weight: drive = std_error, spent by value lean."""

    def test_exploit_favors_high_score(self):
        """With equal uncertainty, exploit weights the promising move higher."""
        good = Stats(8, 1, 1)
        bad = Stats(1, 1, 8)  # mirror of good -> identical std_error, lower score
        assert abs(good.std_error - bad.std_error) < 1e-9  # symmetric counts
        assert move_weight(good, is_prune=False) > move_weight(bad, is_prune=False)

    def test_prune_favors_low_score(self):
        """With equal uncertainty, prune weights the unpromising move higher."""
        good = Stats(8, 1, 1)
        bad = Stats(1, 1, 8)
        assert move_weight(bad, is_prune=True) > move_weight(good, is_prune=True)

    def test_weights_are_mirror_images_summing_to_drive(self):
        """exploit + prune weight == std_error (the two split the uncertainty)."""
        for s in (Stats(3, 2, 5), Stats(10, 0, 0), Stats(0, 0, 0), Stats(1, 7, 2)):
            total = move_weight(s, is_prune=False) + move_weight(s, is_prune=True)
            assert abs(total - s.std_error) < 1e-12

    def test_weight_scales_with_uncertainty(self):
        """Self-correction: an under-sampled move outweighs a well-sampled one
        of the same value, so sampling (which shrinks std_error) lowers weight."""
        unsure = Stats(1, 0, 1)     # score 0.5, high std_error
        resolved = Stats(50, 0, 50)  # score 0.5, low std_error
        assert unsure.mean_score == resolved.mean_score
        assert move_weight(unsure, is_prune=False) > move_weight(resolved, is_prune=False)
        assert move_weight(unsure, is_prune=True) > move_weight(resolved, is_prune=True)


def _random_game_stacks(rng, n_games):
    """Play n random Tic-Tac-Toe games; return record_round stacks."""
    stacks = []
    for _ in range(n_games):
        g = TicTacToe()
        moves = {1: [], 2: []}
        while not g.is_over():
            p = g.current_player()
            board = g.get_state().board.copy()
            valid = list(g.valid_moves())
            mv = valid[rng.randrange(len(valid))]
            g.apply_move(mv, validated=True)
            moves[p].append((mv, board, p))
        for p in (1, 2):
            stacks.append((moves[p], g.get_result(p)))
    return stacks


def _legal(game):
    return {tuple(int(x) for x in np.asarray(m).ravel()) for m in game.valid_moves()}


def _as_key(move):
    return tuple(int(x) for x in np.asarray(move).ravel())


class TestSelectMoveForTraining:
    """Integration: sampling returns legal moves and is reproducible."""

    def test_fresh_memory_returns_legal_move(self, tmp_path):
        """Unexplored frontier: all weights tie -> uniform fallback, still legal."""
        mem = M.for_game(TicTacToe(), base_dir=str(tmp_path))
        g = TicTacToe()
        for is_prune in (False, True):
            random.seed(0)
            sel = select_move_for_training(g, mem, is_prune=is_prune)
            assert _as_key(sel) in _legal(g)
        mem.close()

    def test_populated_memory_returns_legal_move(self, tmp_path):
        """Weighted path (real stats): both phases return a legal move."""
        mem = M.for_game(TicTacToe(), base_dir=str(tmp_path))
        mem.record_round(TicTacToe, _random_game_stacks(random.Random(0), 60))
        g = TicTacToe()
        for is_prune in (False, True):
            random.seed(7)
            sel = select_move_for_training(g, mem, is_prune=is_prune)
            assert _as_key(sel) in _legal(g)
        mem.close()

    def test_reproducible_under_seed(self, tmp_path):
        """Same seed -> same sampled sequence."""
        mem = M.for_game(TicTacToe(), base_dir=str(tmp_path))
        mem.record_round(TicTacToe, _random_game_stacks(random.Random(1), 60))
        g = TicTacToe()

        def draws():
            return [_as_key(select_move_for_training(g, mem, is_prune=False)) for _ in range(20)]

        random.seed(123); first = draws()
        random.seed(123); second = draws()
        assert first == second
        mem.close()
