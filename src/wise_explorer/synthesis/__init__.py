"""Concept invention via MDL-guided program synthesis, with reuse.

The discovery engine searches for board features to build out of generic
primitives (cell reads, arithmetic/bitwise ops, one fold combinator), scores
them by how much they compress the values (MDL), reuses its own discoveries to
reach richer concepts, and stops a round once it no longer pays. Split into
exprs (the algebra) -> engine (the search) -> reader (the renderer).

Public entry point: ``invent(memory, game_id)`` -> ``InventionResult``.
See ``docs/concept-invention.md`` and the ``invent`` CLI verb.
"""
from wise_explorer.synthesis.exprs import (
    Expr, Cell, Lit, BinOp, UnaryOp, Named, Elem, Fold,
    CellDomain, BoardDomain, GroupDomain, Concept,
    expr_to_dict, expr_from_dict, _OPS, _FOLD, _UNARY, _WORD,
)
from wise_explorer.synthesis.engine import (
    Rule, RoundInfo, InventionResult, invent, invent_from_boards,
    _boards_values, _supports, _cell_group, _group_fold_candidates, _residual,
    CAP, MIN_BOARDS,
)
from wise_explorer.synthesis.reader import (
    render, meaning, _handles, _pretty, _tree_lines, _key_lines, _region_geometry,
)
