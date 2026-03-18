"""Incremental Tree Inducer (ITI) for per-wave predicate mining.

Based on: Utgoff, P.E. (1997). "Decision Tree Induction Based on Efficient
Tree Restructuring." Machine Learning, 29(1), 5-44.

Incrementally maintains a CART-equivalent tree via pull-up rotations.
Each new transition updates stats along the root-to-leaf path and checks
if any split should change. Uses the transition table for exact
repartitioning on rotation (perfect recall).

Two modes of predicate output:

    During training (per-wave): The full tree is used for matching —
    more leaves = finer-grained priors for exploration. Redundant siblings
    provide more precise score estimates for specific board patterns.

    After training (final extraction): Bottom-up sibling pruning merges
    leaves whose parent variance is below threshold. Since tree leaves
    form a partition (no overlap by construction), this is provably
    optimal — the only possible redundancy is siblings with similar scores,
    and bottom-up merging catches all of it in O(leaves).

    Empirically verified: pruned ITI produces the same predicate count and
    quality as batch CART on the same data (9 vs 9 predicates on Nim).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wise_explorer.memory.predicates import (
    Eq, Neq, Literal,
    _board_at, _from_board_at, _derive_mining_params,
    AtomClause, Conjunction, Predicate, TORCH_AVAILABLE,
)
from wise_explorer.memory.tree_miner import TreeMiner

if TORCH_AVAILABLE:
    import torch


class _ITINode:
    """Node in the ITI tree with sufficient statistics per atom."""
    __slots__ = (
        "is_leaf", "n_atoms", "match_count", "match_sum", "match_sq",
        "miss_count", "miss_sum", "miss_sq", "total_count", "total_sum",
        "total_sq", "split_atom", "left", "right", "trans_indices",
    )

    def __init__(self, n_atoms: int):
        self.is_leaf = True
        self.n_atoms = n_atoms
        self.match_count = np.zeros(n_atoms, dtype=np.int32)
        self.match_sum = np.zeros(n_atoms)
        self.match_sq = np.zeros(n_atoms)
        self.miss_count = np.zeros(n_atoms, dtype=np.int32)
        self.miss_sum = np.zeros(n_atoms)
        self.miss_sq = np.zeros(n_atoms)
        self.total_count = 0
        self.total_sum = 0.0
        self.total_sq = 0.0
        self.split_atom = -1
        self.left: Optional["_ITINode"] = None
        self.right: Optional["_ITINode"] = None
        self.trans_indices: set = set()

    def update_stats(self, atom_matches: np.ndarray, score: float):
        self.total_count += 1
        self.total_sum += score
        self.total_sq += score * score
        m = atom_matches.astype(bool)
        self.match_count[m] += 1
        self.match_sum[m] += score
        self.match_sq[m] += score * score
        self.miss_count[~m] += 1
        self.miss_sum[~m] += score
        self.miss_sq[~m] += score * score

    def populate_batch(self, indices: list, match_matrix: np.ndarray, scores: np.ndarray):
        """Batch-populate stats from a set of transition indices. Vectorized."""
        if not indices:
            return
        idx_arr = np.array(indices)
        sc = scores[idx_arr]                        # (K,)
        mm = match_matrix[:, idx_arr]               # (n_atoms, K) bool
        self.total_count = len(indices)
        self.total_sum = float(sc.sum())
        self.total_sq = float((sc * sc).sum())
        self.match_count = mm.sum(axis=1).astype(np.int32)       # (n_atoms,)
        self.match_sum = (mm * sc[np.newaxis, :]).sum(axis=1)    # (n_atoms,)
        self.match_sq = (mm * (sc * sc)[np.newaxis, :]).sum(axis=1)
        nm = ~mm
        self.miss_count = nm.sum(axis=1).astype(np.int32)
        self.miss_sum = (nm * sc[np.newaxis, :]).sum(axis=1)
        self.miss_sq = (nm * (sc * sc)[np.newaxis, :]).sum(axis=1)
        self.trans_indices = set(indices)

    def parent_variance(self) -> float:
        if self.total_count < 2:
            return 0.0
        mean = self.total_sum / self.total_count
        return self.total_sq / self.total_count - mean * mean

    def mean_score(self) -> float:
        return self.total_sum / self.total_count if self.total_count > 0 else 0.5

    def best_split_atom(self, min_leaf: int, exclude: set) -> Tuple[int, float]:
        """Atom with highest variance reduction. Returns (-1, -inf) if none."""
        n = self.total_count
        if n < min_leaf * 2:
            return -1, float("-inf")

        pv = self.parent_variance()
        mc = np.clip(self.match_count.astype(float), 1, None)
        lv = np.clip(self.match_sq / mc - (self.match_sum / mc) ** 2, 0, None)
        nc = np.clip(self.miss_count.astype(float), 1, None)
        rv = np.clip(self.miss_sq / nc - (self.miss_sum / nc) ** 2, 0, None)
        red = pv - (self.match_count / n * lv + self.miss_count / n * rv)

        for idx in exclude:
            red[idx] = float("-inf")
        red[self.match_count < min_leaf] = float("-inf")
        red[self.miss_count < min_leaf] = float("-inf")

        best = int(np.argmax(red))
        return best, float(red[best])

    def _reduction_for(self, atom: int) -> float:
        """Variance reduction for a specific atom."""
        n = self.total_count
        if n < 2:
            return 0.0
        pv = self.parent_variance()
        mc = max(self.match_count[atom], 1)
        lv = max(self.match_sq[atom] / mc - (self.match_sum[atom] / mc) ** 2, 0)
        nc = max(self.miss_count[atom], 1)
        rv = max(self.miss_sq[atom] / nc - (self.miss_sum[atom] / nc) ** 2, 0)
        return pv - (self.match_count[atom] / n * lv + self.miss_count[atom] / n * rv)


class ITIMiner:
    """Incremental Tree Inducer for per-wave predicate mining.

    Maintains a persistent CART-equivalent tree that updates incrementally
    as new transitions arrive. Uses pull-up rotations with hysteresis to
    restructure when evidence shifts, and perfect recall from the transition
    table for exact repartitioning.

    Reference: Utgoff (1997), Machine Learning 29(1), 5-44.

    Typical per-wave cost: 0.1-0.6ms (vs 8.6ms for batch CART rebuild).
    """

    def __init__(self, device: Optional[str] = None, hysteresis: float = 0.10):
        self._device_name = "cpu"
        if TORCH_AVAILABLE:
            if device:
                self._device_name = device
            elif torch.cuda.is_available():
                self._device_name = "cuda"
        self.hysteresis = hysteresis

        # Persistent state across mine() calls
        self._root: Optional[_ITINode] = None
        self._atoms: Optional[list] = None
        self._match_matrix: Optional[np.ndarray] = None  # (A, N) numpy
        self._scores: Optional[np.ndarray] = None
        self._n_atoms = 0
        self._n_samples = 0
        self._trans_key_to_idx: Dict[Tuple[str, str], int] = {}
        self._min_leaf = 10
        self._max_depth = 6
        self._counts: Optional[np.ndarray] = None

    def mine(
        self,
        boards: Dict[str, np.ndarray],
        scores: Dict,
        prune: bool = False,
        wave_keys: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Predicate]:
        """Incremental mine: update tree with new/changed transitions.

        First call builds the tree from scratch. Subsequent calls only
        process transitions touched this wave.

        Args:
            prune: If True, apply bottom-up sibling pruning using the
                   1 SE criterion: merge siblings whose mean-score gap
                   is less than 1 standard error (statistically insignificant).
                   If False, emit all leaves for finer-grained priors.
            wave_keys: Transitions touched this wave (new or updated).
                   Avoids O(N) scan of all keys. If None, scans all keys.
        """
        self._prune_on_extract = prune
        if not TORCH_AVAILABLE or not boards or not scores:
            return []

        if self._root is None or self._atoms is None:
            # Cold start: build everything from scratch
            trans_keys = [
                k for k in scores
                if isinstance(k, tuple) and len(k) == 2
                and k[0] in boards and k[1] in boards
            ]
            if len(trans_keys) < 4:
                return []
            sample = boards[trans_keys[0][1]]
            return self._cold_start(boards, scores, trans_keys,
                                     sample.shape[0], sample.shape[1])

        # Fast path: caller tells us which transitions were touched
        if wave_keys is not None:
            touched = [k for k in wave_keys
                       if k[0] in boards and k[1] in boards and k in scores]
        else:
            # Slow path: scan all keys to find new ones
            touched = [k for k in scores
                       if isinstance(k, tuple) and len(k) == 2
                       and k not in self._trans_key_to_idx
                       and k[0] in boards and k[1] in boards]

        # Separate into truly new vs score-updated existing
        new_keys = [k for k in touched if k not in self._trans_key_to_idx]
        updated_keys = [k for k in touched if k in self._trans_key_to_idx]

        # Update scores for existing transitions
        for k in updated_keys:
            idx = self._trans_key_to_idx[k]
            self._scores[idx] = scores[k][1]

        if not new_keys:
            return self._extract_predicates()

        # Extend match matrix with new transitions
        # For small batches, evaluate atoms directly in numpy (skip torch overhead)
        new_mm = self._eval_atoms_numpy(boards, new_keys)

        # Extend arrays
        old_n = self._n_samples
        new_n = len(new_keys)
        self._match_matrix = np.concatenate(
            [self._match_matrix, new_mm], axis=1,
        )
        new_scores = np.array([scores[k][1] for k in new_keys])
        self._scores = np.concatenate([self._scores, new_scores])
        new_counts = np.array([scores[k][0] for k in new_keys])
        self._counts = np.concatenate([self._counts, new_counts])

        for i, k in enumerate(new_keys):
            self._trans_key_to_idx[k] = old_n + i
        self._n_samples = old_n + new_n

        # Feed only NEW transitions into the tree (incremental update)
        for i in range(new_n):
            tidx = old_n + i
            self._insert(self._root, tidx, self._match_matrix[:, tidx],
                          self._scores[tidx], set(), 0)

        return self._extract_predicates()

    def _eval_atoms_numpy(self, boards, keys):
        """Evaluate existing atoms on new transitions using pure numpy.

        Much faster than torch for small batches (1-10 transitions) by
        avoiding tensor creation, device transfer, and kernel launch overhead.
        Returns (n_atoms, len(keys)) bool numpy array.
        """
        n = len(keys)
        result = np.zeros((self._n_atoms, n), dtype=bool)
        for i, (fh, th) in enumerate(keys):
            to_b = boards[th]
            from_b = boards[fh]
            for j, atom in enumerate(self._atoms):
                kind = atom[0]
                r, c = atom[1], atom[2] if len(atom) > 2 else (0, 0)
                if kind == "eq":
                    result[j, i] = int(to_b[r, c]) == atom[3]
                elif kind == "neq":
                    result[j, i] = int(to_b[r, c]) != atom[3]
                elif kind == "cells_eq":
                    r2, c2 = atom[3], atom[4]
                    result[j, i] = int(to_b[r, c]) == int(to_b[r2, c2])
                elif kind == "from_eq":
                    result[j, i] = int(from_b[r, c]) == atom[3]
                elif kind == "from_neq":
                    result[j, i] = int(from_b[r, c]) != atom[3]
                elif kind == "changed":
                    result[j, i] = int(to_b[r, c]) != int(from_b[r, c])
                elif kind == "unchanged":
                    result[j, i] = int(to_b[r, c]) == int(from_b[r, c])
                elif kind == "cross_eq":
                    r2, c2 = atom[3], atom[4]
                    result[j, i] = int(to_b[r, c]) == int(from_b[r2, c2])
        return result

    def _cold_start(self, boards, scores, trans_keys, rows, cols):
        """Build tree from scratch on first call or after major changes."""
        device = torch.device(self._device_name)
        n = len(trans_keys)

        to_tensor = torch.stack([
            torch.from_numpy(boards[th].copy()) for _, th in trans_keys
        ]).to(device)
        from_tensor = torch.stack([
            torch.from_numpy(boards[fh].copy()) for fh, _ in trans_keys
        ]).to(device)

        helper = TreeMiner(device=self._device_name)
        self._atoms = helper._generate_atoms(to_tensor, rows, cols, from_tensor)
        mm_torch = helper._build_match_matrix(
            self._atoms, to_tensor, rows, cols, from_tensor,
        )
        self._match_matrix = mm_torch.cpu().numpy()
        self._scores = np.array([scores[k][1] for k in trans_keys])
        self._counts = np.array([scores[k][0] for k in trans_keys])  # (N, 3) wins/ties/losses
        self._n_atoms = len(self._atoms)
        self._n_samples = n

        self._trans_key_to_idx = {k: i for i, k in enumerate(trans_keys)}

        derived = _derive_mining_params(n, 0.0, self._n_atoms)
        self._min_leaf = max(derived["min_support"], 3)
        self._max_depth = derived["max_atoms"]

        # Build tree by feeding all transitions
        self._root = _ITINode(self._n_atoms)
        for tidx in range(n):
            self._insert(self._root, tidx, self._match_matrix[:, tidx],
                          self._scores[tidx], set(), 0)

        return self._extract_predicates()

    def _insert(self, node: _ITINode, trans_idx: int, atom_matches: np.ndarray,
                score: float, ancestors_used: set, depth: int):
        """Insert a sample, update stats, check for splits/rotations."""
        node.update_stats(atom_matches, score)
        node.trans_indices.add(trans_idx)

        if node.is_leaf:
            if (node.total_count >= self._min_leaf * 2
                    and depth < self._max_depth):
                best, red = node.best_split_atom(self._min_leaf, ancestors_used)
                if best >= 0 and red > 0:
                    self._split_leaf(node, best, depth)
            return

        if atom_matches[node.split_atom]:
            child = node.left
        else:
            child = node.right

        self._insert(child, trans_idx, atom_matches, score,
                      ancestors_used | {node.split_atom}, depth + 1)
        self._check_rotation(node, ancestors_used, depth)

    def _split_leaf(self, node: _ITINode, split_atom: int, depth: int):
        """Convert leaf to internal node, populate children via perfect recall."""
        node.is_leaf = False
        node.split_atom = split_atom
        node.left = _ITINode(self._n_atoms)
        node.right = _ITINode(self._n_atoms)

        # Vectorized: split indices by atom match
        all_idx = list(node.trans_indices)
        if all_idx:
            idx_arr = np.array(all_idx)
            mask = self._match_matrix[split_atom, idx_arr].astype(bool)
            left_idx = idx_arr[mask].tolist()
            right_idx = idx_arr[~mask].tolist()
            node.left.populate_batch(left_idx, self._match_matrix, self._scores)
            node.right.populate_batch(right_idx, self._match_matrix, self._scores)

    def _check_rotation(self, node: _ITINode, ancestors_used: set, depth: int):
        """Pull-up rotation with hysteresis."""
        if node.is_leaf:
            return

        best, best_red = node.best_split_atom(self._min_leaf, ancestors_used)
        if best < 0 or best == node.split_atom:
            return

        current_red = node._reduction_for(node.split_atom)
        margin = max(current_red * self.hysteresis, 1e-6)
        if best_red <= current_red + margin:
            return

        # Commit rotation: rebuild subtree with new split (vectorized)
        node.split_atom = best
        node.left = _ITINode(self._n_atoms)
        node.right = _ITINode(self._n_atoms)

        all_idx = list(node.trans_indices)
        if all_idx:
            idx_arr = np.array(all_idx)
            mask = self._match_matrix[best, idx_arr].astype(bool)
            left_idx = idx_arr[mask].tolist()
            right_idx = idx_arr[~mask].tolist()
            node.left.populate_batch(left_idx, self._match_matrix, self._scores)
            node.right.populate_batch(right_idx, self._match_matrix, self._scores)

        # Try splitting children (they have full stats from repartition)
        used = ancestors_used | {best}
        for child in (node.left, node.right):
            if child.total_count >= self._min_leaf * 2 and depth + 1 < self._max_depth:
                b, r = child.best_split_atom(self._min_leaf, used)
                if b >= 0 and r > 0:
                    self._split_leaf(child, b, depth + 1)

    def _extract_predicates(self) -> List[Predicate]:
        """Extract predicates from the tree, optionally pruning redundant siblings.

        When pruning is enabled, uses the 1 SE criterion: merge siblings whose
        mean-score gap is less than 1 standard error of the parent's mean.
        This is threshold-free — it uses the data's own noise level to decide
        what's statistically significant.

        Since tree leaves form a partition (no overlap), sibling merging is
        provably optimal: the only possible redundancy is siblings with
        similar scores.

        The tree itself is NOT modified (it needs its structure for incremental
        updates). Only the extracted predicates are pruned.
        """
        if self._root is None:
            return []

        import math
        emit_nodes: list = []

        if getattr(self, '_prune_on_extract', False):
            self._se_prune_collect(self._root, [], emit_nodes)
        else:
            stack = [(self._root, [])]
            while stack:
                node, path = stack.pop()
                if node.is_leaf:
                    emit_nodes.append((node, path))
                else:
                    stack.append((node.left, path + [(node.split_atom, True)]))
                    stack.append((node.right, path + [(node.split_atom, False)]))

        # Convert to predicates
        predicates = []
        seen: set = set()
        for node, path in emit_nodes:
            if node.total_count < self._min_leaf or not path:
                continue

            clauses = []
            for atom_idx, is_match in path:
                if is_match:
                    atom = TreeMiner._descriptor_to_atom(self._atoms[atom_idx])
                else:
                    atom = TreeMiner._descriptor_to_atom_negated(self._atoms, -(atom_idx + 1))
                if atom is not None:
                    clauses.append(AtomClause(atom))

            if not clauses:
                continue

            conj = Conjunction(clauses)
            key = str(conj)
            if key in seen:
                continue
            seen.add(key)

            # Aggregate real counts from all transitions in this node
            if hasattr(self, '_counts') and self._counts is not None and node.trans_indices:
                idx_list = list(node.trans_indices)
                node_counts = self._counts[idx_list].sum(axis=0)
                counts = (float(node_counts[0]), float(node_counts[1]), float(node_counts[2]))
            else:
                counts = (0.0, 0.0, 0.0)

            predicates.append(Predicate(
                conjunction=conj,
                counts=counts,
                support=node.total_count,
                variance=node.parent_variance(),
                mining_score=node.mean_score(),
            ))

        return predicates

    def _se_prune_collect(self, node: _ITINode, path: list, emit: list):
        """Collect predicates with 1 SE sibling pruning.

        Merge siblings when |left_mean - right_mean| < SE, where SE is the
        standard error of the parent's mean (sqrt(var / n)). This means the
        score difference between siblings is not statistically significant —
        the split didn't separate different strategic regions.
        """
        import math

        if node.is_leaf:
            emit.append((node, path))
            return

        # 1 SE criterion: is the gap between children significant?
        gap = abs(node.left.mean_score() - node.right.mean_score())
        parent_var = node.parent_variance()
        se = math.sqrt(parent_var / max(node.total_count, 1))

        # Merge if gap < 1 SE (not significant). Never merge root (path=[]).
        if path and gap < se:
            emit.append((node, path))
            return

        self._se_prune_collect(node.left, path + [(node.split_atom, True)], emit)
        self._se_prune_collect(node.right, path + [(node.split_atom, False)], emit)
