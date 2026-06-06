"""Batch CART decision-tree predicate miner (GPU/CPU).

Builds a regression tree where atom match values are features and
transition scores are targets. Each leaf yields a predicate
(conjunction of atom decisions along the root-to-leaf path).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wise_explorer.memory.predicates import (
    Eq, Neq, Gt, Le, Literal, BoardAt, MakeSq, FromBoardAt,
    _board_at, _from_board_at, _derive_mining_params, _AggAtom, _NegAggAtom,
    AtomClause, Conjunction, Predicate, TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch


class TreeMiner:
    """Decision-tree predicate miner using CART splitting.

    Builds a regression tree where:
    - Features = atom match values (boolean)
    - Target = score (bell or mean)
    - Each leaf = a predicate (conjunction of atom decisions along the path)

    Advantages:
    - Finds both WIN and LOSS patterns (tree explores both branches)
    - Parameter-free (depth from max_atoms, leaf size from min_support)
    - Fast: O(atoms x boards x depth) — typically <10ms
    - Tensor-accelerated atom generation and match matrix building
    """

    def __init__(self, device: Optional[str] = None):
        if TORCH_AVAILABLE:
            if device:
                self._device_name = device
            elif torch.cuda.is_available():
                self._device_name = "cuda"
            else:
                self._device_name = "cpu"
        else:
            self._device_name = "cpu"

    def _generate_atoms(self, board_tensor, rows, cols, from_tensor=None):
        """Generate atom descriptors as (type, params) tuples.

        Returns list of tuples describing each atom, used to build both
        the match matrix and the expression-tree Atom objects.
        """
        atoms = []
        flat = board_tensor.reshape(board_tensor.shape[0], -1)  # (N, R*C)
        n_cells = rows * cols

        # === Board atoms (destination board) ===

        # CellEquals: for each cell, for each observed value
        for cell_idx in range(n_cells):
            r, c = divmod(cell_idx, cols)
            unique_vals = flat[:, cell_idx].unique().tolist()
            for v in unique_vals:
                atoms.append(("eq", r, c, int(v)))

        # CellGt: for each cell, for each observed value (threshold)
        for cell_idx in range(n_cells):
            r, c = divmod(cell_idx, cols)
            unique_vals = flat[:, cell_idx].unique().tolist()
            for v in unique_vals:
                atoms.append(("gt", r, c, int(v)))

        # CellNonEmpty: for each cell
        for cell_idx in range(n_cells):
            r, c = divmod(cell_idx, cols)
            atoms.append(("neq", r, c, 0))

        # CellsEqual: for each cell pair
        for i in range(n_cells):
            r1, c1 = divmod(i, cols)
            for j in range(i + 1, n_cells):
                r2, c2 = divmod(j, cols)
                atoms.append(("cells_eq", r1, c1, r2, c2))

        # === Cross-board atoms (from → to transformation) ===
        if from_tensor is not None:
            from_flat = from_tensor.reshape(from_tensor.shape[0], -1)  # (N, R*C)

            # FromBoardAt equals value
            for cell_idx in range(n_cells):
                r, c = divmod(cell_idx, cols)
                unique_vals = from_flat[:, cell_idx].unique().tolist()
                for v in unique_vals:
                    atoms.append(("from_eq", r, c, int(v)))

            # FromBoardAt non-empty
            for cell_idx in range(n_cells):
                r, c = divmod(cell_idx, cols)
                atoms.append(("from_neq", r, c, 0))

            # Cell changed: to[r,c] != from[r,c]
            for cell_idx in range(n_cells):
                r, c = divmod(cell_idx, cols)
                atoms.append(("changed", r, c))

            # Cell unchanged: to[r,c] == from[r,c]
            for cell_idx in range(n_cells):
                r, c = divmod(cell_idx, cols)
                atoms.append(("unchanged", r, c))

        # === Aggregate atoms (multi-cell properties) ===
        # These capture group-level relationships that no single-cell atom
        # can express: counting, summing, extremes. Groups: whole board +
        # rows + columns for 2D games.

        all_cells = list(range(n_cells))

        groups = [("all", all_cells)]
        if rows > 1:
            for r in range(rows):
                groups.append((f"row{r}", [r * cols + c for c in range(cols)]))
            for c in range(cols):
                groups.append((f"col{c}", [r * cols + c for r in range(rows)]))

        for group_name, cell_indices in groups:
            group_vals = flat[:, cell_indices]  # (N, group_size)

            # Sum: for each unique observed sum value, == and >
            sums = group_vals.sum(dim=1)
            for v in sums.unique().tolist():
                v = int(v)
                atoms.append(("agg_sum_eq", group_name, tuple(cell_indices), v))
                atoms.append(("agg_sum_gt", group_name, tuple(cell_indices), v))

            # Max: for each unique observed max value, == and >
            maxes = group_vals.max(dim=1).values
            for v in maxes.unique().tolist():
                v = int(v)
                atoms.append(("agg_max_eq", group_name, tuple(cell_indices), v))
                atoms.append(("agg_max_gt", group_name, tuple(cell_indices), v))

            # Count nonzero: for each unique count, == and >
            counts_nz = (group_vals != 0).sum(dim=1)
            for v in counts_nz.unique().tolist():
                v = int(v)
                atoms.append(("agg_count_nz_eq", group_name, tuple(cell_indices), v))
                atoms.append(("agg_count_nz_gt", group_name, tuple(cell_indices), v))

            # Count equal to specific value: for each observed cell value
            all_vals = group_vals.unique().tolist()
            for val in all_vals:
                val = int(val)
                counts_eq = (group_vals == val).sum(dim=1)
                for v in counts_eq.unique().tolist():
                    v = int(v)
                    atoms.append(("agg_count_eq", group_name, tuple(cell_indices), val, v))
                    atoms.append(("agg_count_eq_gt", group_name, tuple(cell_indices), val, v))

            # Min: for each unique observed min value, == and >
            mins = group_vals.min(dim=1).values
            for v in mins.unique().tolist():
                v = int(v)
                atoms.append(("agg_min_eq", group_name, tuple(cell_indices), v))
                atoms.append(("agg_min_gt", group_name, tuple(cell_indices), v))

            # Count distinct non-zero values
            cd_list = []
            for row_idx in range(group_vals.shape[0]):
                row_vals = group_vals[row_idx]
                nz = row_vals[row_vals != 0]
                cd_list.append(len(nz.unique()) if nz.numel() > 0 else 0)
            cd_t = torch.tensor(cd_list, device=group_vals.device)
            for v in cd_t.unique().tolist():
                v = int(v)
                atoms.append(("agg_count_distinct_eq", group_name, tuple(cell_indices), v))
                atoms.append(("agg_count_distinct_gt", group_name, tuple(cell_indices), v))

            # XOR: bitwise XOR of all values in group
            xors = group_vals[:, 0].clone()
            for ci in range(1, group_vals.shape[1]):
                xors = xors ^ group_vals[:, ci]
            for v in xors.unique().tolist():
                v = int(v)
                atoms.append(("agg_xor_eq", group_name, tuple(cell_indices), v))
                atoms.append(("agg_xor_gt", group_name, tuple(cell_indices), v))

        return atoms

    def _build_match_matrix(self, atoms, board_tensor, rows, cols, from_tensor=None):
        """Build (num_atoms, N) boolean match matrix with vectorized ops."""
        n_boards = board_tensor.shape[0]
        flat = board_tensor.reshape(n_boards, -1)
        from_flat = from_tensor.reshape(n_boards, -1) if from_tensor is not None else None

        match_rows = []
        for atom in atoms:
            if atom[0] == "eq":
                _, r, c, v = atom
                cell_idx = r * cols + c
                match_rows.append(flat[:, cell_idx] == v)
            elif atom[0] == "gt":
                _, r, c, v = atom
                cell_idx = r * cols + c
                match_rows.append(flat[:, cell_idx] > v)
            elif atom[0] == "neq":
                _, r, c, v = atom
                cell_idx = r * cols + c
                match_rows.append(flat[:, cell_idx] != v)
            elif atom[0] == "cells_eq":
                _, r1, c1, r2, c2 = atom
                idx1 = r1 * cols + c1
                idx2 = r2 * cols + c2
                match_rows.append(flat[:, idx1] == flat[:, idx2])
            elif atom[0] == "from_eq":
                _, r, c, v = atom
                cell_idx = r * cols + c
                match_rows.append(from_flat[:, cell_idx] == v)
            elif atom[0] == "from_neq":
                _, r, c, v = atom
                cell_idx = r * cols + c
                match_rows.append(from_flat[:, cell_idx] != v)
            elif atom[0] == "changed":
                _, r, c = atom
                cell_idx = r * cols + c
                match_rows.append(flat[:, cell_idx] != from_flat[:, cell_idx])
            elif atom[0] == "unchanged":
                _, r, c = atom
                cell_idx = r * cols + c
                match_rows.append(flat[:, cell_idx] == from_flat[:, cell_idx])
            # --- Aggregate atoms ---
            elif atom[0] == "agg_sum_eq":
                _, _name, indices, v = atom
                match_rows.append(flat[:, list(indices)].sum(dim=1) == v)
            elif atom[0] == "agg_sum_gt":
                _, _name, indices, v = atom
                match_rows.append(flat[:, list(indices)].sum(dim=1) > v)
            elif atom[0] == "agg_max_eq":
                _, _name, indices, v = atom
                match_rows.append(flat[:, list(indices)].max(dim=1).values == v)
            elif atom[0] == "agg_max_gt":
                _, _name, indices, v = atom
                match_rows.append(flat[:, list(indices)].max(dim=1).values > v)
            elif atom[0] == "agg_count_nz_eq":
                _, _name, indices, v = atom
                match_rows.append((flat[:, list(indices)] != 0).sum(dim=1) == v)
            elif atom[0] == "agg_count_nz_gt":
                _, _name, indices, v = atom
                match_rows.append((flat[:, list(indices)] != 0).sum(dim=1) > v)
            elif atom[0] == "agg_count_eq":
                _, _name, indices, val, v = atom
                match_rows.append((flat[:, list(indices)] == val).sum(dim=1) == v)
            elif atom[0] == "agg_count_eq_gt":
                _, _name, indices, val, v = atom
                match_rows.append((flat[:, list(indices)] == val).sum(dim=1) > v)
            elif atom[0] == "agg_min_eq":
                _, _name, indices, v = atom
                match_rows.append(flat[:, list(indices)].min(dim=1).values == v)
            elif atom[0] == "agg_min_gt":
                _, _name, indices, v = atom
                match_rows.append(flat[:, list(indices)].min(dim=1).values > v)
            elif atom[0] == "agg_count_distinct_eq":
                _, _name, indices, v = atom
                cols_g = flat[:, list(indices)]
                cd = torch.tensor([len(cols_g[i][cols_g[i] != 0].unique()) if (cols_g[i] != 0).any() else 0 for i in range(cols_g.shape[0])], device=flat.device)
                match_rows.append(cd == v)
            elif atom[0] == "agg_count_distinct_gt":
                _, _name, indices, v = atom
                cols_g = flat[:, list(indices)]
                cd = torch.tensor([len(cols_g[i][cols_g[i] != 0].unique()) if (cols_g[i] != 0).any() else 0 for i in range(cols_g.shape[0])], device=flat.device)
                match_rows.append(cd > v)
            elif atom[0] == "agg_xor_eq":
                _, _name, indices, v = atom
                idx_list = list(indices)
                xor_vals = flat[:, idx_list[0]].clone()
                for ci in idx_list[1:]:
                    xor_vals = xor_vals ^ flat[:, ci]
                match_rows.append(xor_vals == v)
            elif atom[0] == "agg_xor_gt":
                _, _name, indices, v = atom
                idx_list = list(indices)
                xor_vals = flat[:, idx_list[0]].clone()
                for ci in idx_list[1:]:
                    xor_vals = xor_vals ^ flat[:, ci]
                match_rows.append(xor_vals > v)

        return torch.stack(match_rows)  # (num_atoms, N)

    @staticmethod
    def _descriptor_to_atom(desc) -> "Atom":
        """Convert an atom descriptor tuple back to an expression-tree Atom."""
        if desc[0] == "eq":
            _, r, c, v = desc
            return Eq(_board_at(r, c), Literal(v))
        elif desc[0] == "gt":
            _, r, c, v = desc
            return Gt(_board_at(r, c), Literal(v))
        elif desc[0] == "neq":
            _, r, c, v = desc
            return Neq(_board_at(r, c), Literal(v))
        elif desc[0] == "cells_eq":
            _, r1, c1, r2, c2 = desc
            return Eq(_board_at(r1, c1), _board_at(r2, c2))
        elif desc[0] == "from_eq":
            _, r, c, v = desc
            return Eq(_from_board_at(r, c), Literal(v))
        elif desc[0] == "from_neq":
            _, r, c, v = desc
            return Neq(_from_board_at(r, c), Literal(v))
        elif desc[0] == "changed":
            _, r, c = desc
            return Neq(_board_at(r, c), _from_board_at(r, c))
        elif desc[0] == "unchanged":
            _, r, c = desc
            return Eq(_board_at(r, c), _from_board_at(r, c))
        elif desc[0].startswith("agg_"):
            # Aggregate atoms: store descriptor, evaluate directly on board
            return _AggAtom(desc)
        raise ValueError(f"Unknown atom descriptor: {desc}")

    def _masked_variance(self, scores, mask):
        """Compute variance of scores where mask is True."""
        device = torch.device(self._device_name)
        masked = scores[mask]
        if masked.numel() < 2:
            return torch.tensor(0.0, device=device)
        return masked.var(correction=0)

    def mine(
        self,
        boards: Dict[str, np.ndarray],
        scores: Dict,
    ) -> List[Predicate]:
        """Discover predicates using CART decision tree.

        Args:
            boards: {hash: board_array}
            scores: transition-keyed {(from_hash, to_hash): (counts, score)}
                    Each sample is a transition, preserving player identity
                    and cross-board context.
        """
        if not TORCH_AVAILABLE or not boards or not scores:
            return []

        # Build per-transition tensors: each sample is (from_board, to_board, score)
        trans_keys = []  # ordered list of (from_hash, to_hash)
        for key in scores:
            if isinstance(key, tuple) and len(key) == 2:
                fh, th = key
                if fh in boards and th in boards:
                    trans_keys.append((fh, th))

        if len(trans_keys) < 4:
            return []

        sample = boards[trans_keys[0][1]]
        rows, cols = sample.shape
        device = torch.device(self._device_name)
        n_samples = len(trans_keys)

        # To-board tensor (destination boards, one per transition)
        to_tensor = torch.stack([
            torch.from_numpy(boards[th].copy()) for _, th in trans_keys
        ]).to(device)

        # From-board tensor (source boards, one per transition)
        from_tensor = torch.stack([
            torch.from_numpy(boards[fh].copy()) for fh, _ in trans_keys
        ]).to(device)

        score_tensor = torch.tensor(
            [scores[key][1] for key in trans_keys],
            dtype=torch.float32, device=device,
        )
        counts_tensor = torch.tensor(
            [scores[key][0] for key in trans_keys],
            dtype=torch.float32, device=device,
        )

        # Generate cell + cross-board atoms
        all_atoms = self._generate_atoms(to_tensor, rows, cols, from_tensor)

        if not all_atoms:
            return []

        # Build match matrix
        match_matrix = self._build_match_matrix(
            all_atoms, to_tensor, rows, cols, from_tensor,
        )
        atoms = all_atoms

        # Derive parameters from transition count
        derived = _derive_mining_params(n_samples, 0.0, len(atoms))
        min_leaf = max(derived["min_support"], 3)
        max_depth = derived["max_atoms"]

        # Two per-sample quantities keep mining robust and parameter-free:
        #   support — games seen (evidence weight). Down-weights rarely-seen
        #             transitions so single-observation noise can't dominate.
        #   se2     — Bayesian std_error² of the value. Lets the tree stop once a
        #             node's spread is within its own samples' noise.
        support = counts_tensor.sum(dim=1).clamp(min=1.0)
        w_, t_, l_ = counts_tensor[:, 0], counts_tensor[:, 1], counts_tensor[:, 2]
        w1, t1, l1 = w_ + 1.0, t_ + 1.0, l_ + 1.0
        n_pseudo = w1 + t1 + l1
        mean_c = (w1 + 0.5 * t1) / n_pseudo
        meansq_c = (w1 + 0.25 * t1) / n_pseudo
        se2 = (meansq_c - mean_c ** 2) / (w_ + t_ + l_ + 3.0)

        # Build support-weighted CART tree — splits on structural features only
        predicates = self._build_tree(
            atoms, match_matrix, score_tensor, counts_tensor, support, se2,
            min_leaf, max_depth, n_samples, device,
        )

        return predicates

    def _add_graph_atoms(
        self,
        graph: Dict[str, List[str]],
        hash_to_idx: Dict[str, int],
        score_tensor,
        n_boards: int,
        device,
    ) -> Tuple[list, list]:
        """Generate graph-aware atoms from transition adjacency.

        NOTE: Currently unused — reserved for future validation.

        Adds atoms for:
        - n_successors (out-degree) bucketed
        - n_predecessors (in-degree) bucketed
        - mean_successor_score bucketed
        - all_successors_low (all successors have score < 0.3)
        - exists_successor_high (some successor has score > 0.7)
        """
        # Build adjacency
        successors: Dict[int, List[int]] = {i: [] for i in range(n_boards)}
        predecessors: Dict[int, List[int]] = {i: [] for i in range(n_boards)}
        for to_hash, from_hashes in graph.items():
            if to_hash not in hash_to_idx:
                continue
            to_idx = hash_to_idx[to_hash]
            for fh in from_hashes:
                if fh in hash_to_idx:
                    from_idx = hash_to_idx[fh]
                    successors[from_idx].append(to_idx)
                    predecessors[to_idx].append(from_idx)

        atoms = []
        match_rows = []

        # Out-degree atoms
        out_degrees = torch.tensor([len(successors[i]) for i in range(n_boards)],
                                    dtype=torch.float32, device=device)
        for threshold in [1, 2, 3, 5]:
            row = out_degrees >= threshold
            if row.any() and not row.all():
                atoms.append(("graph_out_gte", threshold))
                match_rows.append(row)

        # In-degree atoms
        in_degrees = torch.tensor([len(predecessors[i]) for i in range(n_boards)],
                                   dtype=torch.float32, device=device)
        for threshold in [1, 2, 3, 5]:
            row = in_degrees >= threshold
            if row.any() and not row.all():
                atoms.append(("graph_in_gte", threshold))
                match_rows.append(row)

        # Mean successor score atoms
        mean_succ = torch.full((n_boards,), 0.5, device=device)
        for i in range(n_boards):
            if successors[i]:
                idxs = torch.tensor(successors[i], dtype=torch.long, device=device)
                mean_succ[i] = score_tensor[idxs].mean()

        for threshold in [0.2, 0.4, 0.6, 0.8]:
            row = mean_succ >= threshold
            if row.any() and not row.all():
                atoms.append(("graph_succ_score_gte", threshold))
                match_rows.append(row)
            row_lt = mean_succ < threshold
            if row_lt.any() and not row_lt.all():
                atoms.append(("graph_succ_score_lt", threshold))
                match_rows.append(row_lt)

        # All successors low (opponent has no good moves = we're winning)
        all_succ_low = torch.zeros(n_boards, dtype=torch.bool, device=device)
        for i in range(n_boards):
            if successors[i]:
                idxs = torch.tensor(successors[i], dtype=torch.long, device=device)
                all_succ_low[i] = (score_tensor[idxs] < 0.3).all()
        if all_succ_low.any() and not all_succ_low.all():
            atoms.append(("graph_all_succ_low",))
            match_rows.append(all_succ_low)

        # Exists successor high (there's a winning move available)
        exists_succ_high = torch.zeros(n_boards, dtype=torch.bool, device=device)
        for i in range(n_boards):
            if successors[i]:
                idxs = torch.tensor(successors[i], dtype=torch.long, device=device)
                exists_succ_high[i] = (score_tensor[idxs] > 0.7).any()
        if exists_succ_high.any() and not exists_succ_high.all():
            atoms.append(("graph_exists_succ_high",))
            match_rows.append(exists_succ_high)

        return atoms, match_rows

    def _build_tree(
        self,
        atoms: list,
        match_matrix,       # (A, N) bool
        score_tensor,        # (N,) float  — the value the tree fits
        counts_tensor,       # (N, 3) float — raw W/T/L (for leaf aggregation)
        weight_tensor,       # (N,) float  — support (games seen)
        se2_tensor,          # (N,) float  — per-sample measurement variance
        min_leaf: int,
        max_depth: int,
        n_boards: int,
        device,
    ) -> List[Predicate]:
        """Build a CART regression tree and extract predicates from its leaves.

        Structure detection is UNWEIGHTED — every transition contributes equally
        to variance and to split gain, so a single sharp move is never buried
        under the well-trodden lines. The only place support enters is the STOP:
        a node becomes a leaf once its (unweighted) value-variance falls within
        the support-weighted measurement-noise floor of its samples — i.e. the
        remaining spread is explained by sampling noise, with no real structure
        left to split on. Both halves are parameter-free.
        """
        predicates: List[Predicate] = []

        def emit_leaf(mask, used_atoms, n_match, variance, mean):
            if not used_atoms or n_match < min_leaf:
                return
            matched_counts = counts_tensor[mask].sum(dim=0)
            clauses = [
                AtomClause(a) for a in (
                    TreeMiner._descriptor_to_atom_negated(atoms, idx)
                    for idx in used_atoms
                ) if a is not None
            ]
            if clauses:
                predicates.append(Predicate(
                    conjunction=Conjunction(clauses),
                    counts=(matched_counts[0].item(), matched_counts[1].item(),
                            matched_counts[2].item()),
                    support=int(n_match),
                    variance=variance,
                    mining_score=mean,
                ))

        # Recursive split via stack (avoid deep recursion).
        # Each entry: (board_mask, used_atom_indices, depth)
        stack = [(torch.ones(n_boards, dtype=torch.bool, device=device), [], 0)]

        while stack:
            mask, used_atoms, depth = stack.pop()
            n_match = mask.sum().item()
            if n_match < min_leaf:
                continue

            matched_scores = score_tensor[mask]
            current_var = matched_scores.var(correction=0).item() if n_match >= 2 else 0.0
            current_mean = matched_scores.mean().item()

            # Support-weighted measurement-noise floor: the spread we'd expect
            # from sampling noise alone, with rarely-seen transitions kept from
            # inflating it. Stop once the variance is within that floor.
            w_node = weight_tensor[mask]
            node_floor = (w_node * se2_tensor[mask]).sum().item() / max(w_node.sum().item(), 1e-9)

            if (depth >= max_depth or n_match < min_leaf * 2
                    or current_var <= node_floor):
                emit_leaf(mask, used_atoms, n_match, current_var, current_mean)
                continue

            # Vectorized (unweighted) variance reduction over all atoms.
            left_masks = match_matrix & mask.unsqueeze(0)            # (A, N)
            right_masks = (~match_matrix) & mask.unsqueeze(0)
            left_n = left_masks.sum(dim=1).float()
            right_n = right_masks.sum(dim=1).float()

            # Both children need >= min_leaf distinct samples; don't reuse atoms.
            valid = (left_n >= min_leaf) & (right_n >= min_leaf)
            for idx in used_atoms:
                valid[abs(idx) - 1 if idx < 0 else idx] = False
            if not valid.any():
                emit_leaf(mask, used_atoms, n_match, current_var, current_mean)
                continue

            vi = valid.nonzero(as_tuple=True)[0]
            lm = left_masks[vi].float()
            rm = right_masks[vi].float()
            l_scores = score_tensor.unsqueeze(0) * lm
            l_means = l_scores.sum(dim=1) / left_n[vi].clamp(min=1)
            l_vars = ((l_scores - l_means.unsqueeze(1)) * lm).pow(2).sum(dim=1) / left_n[vi].clamp(min=1)
            r_scores = score_tensor.unsqueeze(0) * rm
            r_means = r_scores.sum(dim=1) / right_n[vi].clamp(min=1)
            r_vars = ((r_scores - r_means.unsqueeze(1)) * rm).pow(2).sum(dim=1) / right_n[vi].clamp(min=1)
            n_total = float(n_match)
            reductions = current_var - (
                left_n[vi] / n_total * l_vars + right_n[vi] / n_total * r_vars
            )

            best_local = reductions.argmax()
            best_atom = vi[best_local].item()
            if reductions[best_local].item() <= 0:
                emit_leaf(mask, used_atoms, n_match, current_var, current_mean)
                continue

            # Split: left child takes the atom, right child its negation
            # (tracked via the negative-index convention).
            stack.append((mask & match_matrix[best_atom],
                          used_atoms + [best_atom], depth + 1))
            stack.append((mask & ~match_matrix[best_atom],
                          used_atoms + [-(best_atom + 1)], depth + 1))

        # Deduplicate by conjunction string.
        seen: set = set()
        unique: List[Predicate] = []
        for p in predicates:
            key = str(p.conjunction)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    @staticmethod
    def _descriptor_to_atom_negated(atoms, idx):
        """Convert atom index to Atom, handling negation for right-branch splits.

        Returns None for graph atoms (not representable as board predicates).
        """
        real_idx = abs(idx) - 1 if idx < 0 else idx
        desc = atoms[real_idx]

        # Graph atoms can't be represented as expression-tree atoms
        if desc[0].startswith("graph_"):
            return None

        if idx >= 0:
            return TreeMiner._descriptor_to_atom(desc)

        # Negated atom: flip Eq↔Neq, Gt→Le
        if desc[0] == "eq":
            _, r, c, v = desc
            return Neq(_board_at(r, c), Literal(v))
        elif desc[0] == "gt":
            _, r, c, v = desc
            return Le(_board_at(r, c), Literal(v))
        elif desc[0] == "neq":
            _, r, c, v = desc
            return Eq(_board_at(r, c), Literal(v))
        elif desc[0] == "cells_eq":
            _, r1, c1, r2, c2 = desc
            return Neq(_board_at(r1, c1), _board_at(r2, c2))
        elif desc[0] == "from_eq":
            _, r, c, v = desc
            return Neq(_from_board_at(r, c), Literal(v))
        elif desc[0] == "from_neq":
            _, r, c, v = desc
            return Eq(_from_board_at(r, c), Literal(v))
        elif desc[0] == "changed":
            _, r, c = desc
            return Eq(_board_at(r, c), _from_board_at(r, c))
        elif desc[0] == "unchanged":
            _, r, c = desc
            return Neq(_board_at(r, c), _from_board_at(r, c))
        elif desc[0].startswith("agg_"):
            # Negate aggregate: flip == to != and > to <=
            # Represented as a NegAggAtom that inverts the test
            return _NegAggAtom(desc)
        raise ValueError(f"Unknown atom descriptor: {desc}")
