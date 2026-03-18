"""
Feature-based predicate system for board state generalization.

Discovers interpretable structural patterns (conjunctions of typed
relational predicates) that predict score outcomes with low variance.
Provides score priors for unseen board states via pattern matching.

Architecture:
    Layer 1 — Expression tree (typed value producers)
    Layer 2 — Atoms (boolean comparisons between expressions)
    Layer 3 — Clauses & Conjunctions (∃, ∧, ¬ logic)
    Layer 4 — Predicates (conjunction + score statistics)
    Layer 5 — Mining & Library (discovery + persistence)

The language primitives (∃, ∧, ¬) plus typed functions/arithmetic/comparisons
derive all higher-order constructs: ∀ ≡ ¬∃¬, ∉ ≡ ∧ of !=, ∨ ≡ ¬(¬A ∧ ¬B).
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wise_explorer.core.types import Stats, Counts


# =============================================================================
# Layer 1 — Expression Tree
# =============================================================================

class Expr(ABC):
    """Base class for typed expressions that produce values."""

    @abstractmethod
    def evaluate(self, board: np.ndarray, bindings: Dict[str, Any]) -> Any:
        """Evaluate this expression given a board and variable bindings."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""

    @staticmethod
    def from_dict(d: dict) -> "Expr":
        """Deserialize from dict."""
        t = d["type"]
        if t == "literal":
            return Literal(d["value"])
        if t == "var":
            return Var(d["name"])
        if t == "board_at":
            return BoardAt(Expr.from_dict(d["square"]))
        if t == "from_board_at":
            return FromBoardAt(Expr.from_dict(d["square"]))
        if t == "rank_of":
            return RankOf(Expr.from_dict(d["square"]))
        if t == "file_of":
            return FileOf(Expr.from_dict(d["square"]))
        if t == "make_sq":
            return MakeSq(Expr.from_dict(d["rank"]), Expr.from_dict(d["file"]))
        if t == "add":
            return Add(Expr.from_dict(d["expr"]), d["offset"])
        raise ValueError(f"Unknown expression type: {t}")

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(json.dumps(self.to_dict(), sort_keys=True))


@dataclass(frozen=True)
class Literal(Expr):
    """Constant integer value."""
    value: int

    def evaluate(self, board, bindings):
        return self.value

    def to_dict(self):
        return {"type": "literal", "value": self.value}

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class Var(Expr):
    """Reference to a bound variable."""
    name: str

    def evaluate(self, board, bindings):
        return bindings[self.name]

    def to_dict(self):
        return {"type": "var", "name": self.name}

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class BoardAt(Expr):
    """Read board cell value at a square. board[sq] → value."""
    square: Expr

    def evaluate(self, board, bindings):
        sq = self.square.evaluate(board, bindings)
        return int(board[sq[0], sq[1]])

    def to_dict(self):
        return {"type": "board_at", "square": self.square.to_dict()}

    def __repr__(self):
        return f"board[{self.square}]"


@dataclass(frozen=True)
class RankOf(Expr):
    """Extract row index from a square."""
    square: Expr

    def evaluate(self, board, bindings):
        sq = self.square.evaluate(board, bindings)
        return sq[0]

    def to_dict(self):
        return {"type": "rank_of", "square": self.square.to_dict()}

    def __repr__(self):
        return f"rank({self.square})"


@dataclass(frozen=True)
class FileOf(Expr):
    """Extract column index from a square."""
    square: Expr

    def evaluate(self, board, bindings):
        sq = self.square.evaluate(board, bindings)
        return sq[1]

    def to_dict(self):
        return {"type": "file_of", "square": self.square.to_dict()}

    def __repr__(self):
        return f"file({self.square})"


@dataclass(frozen=True)
class MakeSq(Expr):
    """Construct a square from rank and file expressions."""
    rank: Expr
    file: Expr

    def evaluate(self, board, bindings):
        r = self.rank.evaluate(board, bindings)
        f = self.file.evaluate(board, bindings)
        return (r, f)

    def to_dict(self):
        return {"type": "make_sq", "rank": self.rank.to_dict(), "file": self.file.to_dict()}

    def __repr__(self):
        return f"sq({self.rank},{self.file})"


@dataclass(frozen=True)
class FromBoardAt(Expr):
    """Read pre-move board cell value. from_board[sq] → value.

    Requires bindings["_from"] to contain the pre-move board.
    Used for cross-board (transformation) predicates.
    """
    square: Expr

    def evaluate(self, board, bindings):
        from_board = bindings.get("_from")
        if from_board is None:
            return -999  # sentinel — comparison will fail gracefully
        sq = self.square.evaluate(board, bindings)
        return int(from_board[sq[0], sq[1]])

    def to_dict(self):
        return {"type": "from_board_at", "square": self.square.to_dict()}

    def __repr__(self):
        return f"from[{self.square}]"


@dataclass(frozen=True)
class Add(Expr):
    """Offset arithmetic: expr + constant."""
    expr: Expr
    offset: int

    def evaluate(self, board, bindings):
        return self.expr.evaluate(board, bindings) + self.offset

    def to_dict(self):
        return {"type": "add", "expr": self.expr.to_dict(), "offset": self.offset}

    def __repr__(self):
        if self.offset >= 0:
            return f"({self.expr}+{self.offset})"
        return f"({self.expr}{self.offset})"


# =============================================================================
# Layer 2 — Atoms (boolean tests)
# =============================================================================

class Atom(ABC):
    """Base class for boolean predicates on board state."""

    @abstractmethod
    def evaluate(self, board: np.ndarray, bindings: Dict[str, Any]) -> bool:
        """Evaluate this atom given a board and variable bindings."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""

    @staticmethod
    def from_dict(d: dict) -> "Atom":
        """Deserialize from dict."""
        t = d["type"]
        if t == "eq":
            return Eq(Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "neq":
            return Neq(Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "lt":
            return Lt(Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "gt":
            return Gt(Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "le":
            return Le(Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "ge":
            return Ge(Expr.from_dict(d["left"]), Expr.from_dict(d["right"]))
        if t == "in_bounds":
            return InBounds(Expr.from_dict(d["rank"]), Expr.from_dict(d["file"]))
        raise ValueError(f"Unknown atom type: {t}")

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(json.dumps(self.to_dict(), sort_keys=True))


@dataclass(frozen=True)
class Eq(Atom):
    """Equality comparison: left == right."""
    left: Expr
    right: Expr

    def evaluate(self, board, bindings):
        return self.left.evaluate(board, bindings) == self.right.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "eq", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self):
        return f"{self.left}=={self.right}"


@dataclass(frozen=True)
class Neq(Atom):
    """Inequality: left != right."""
    left: Expr
    right: Expr

    def evaluate(self, board, bindings):
        return self.left.evaluate(board, bindings) != self.right.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "neq", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self):
        return f"{self.left}!={self.right}"


@dataclass(frozen=True)
class Lt(Atom):
    """Less than: left < right."""
    left: Expr
    right: Expr

    def evaluate(self, board, bindings):
        return self.left.evaluate(board, bindings) < self.right.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "lt", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self):
        return f"{self.left}<{self.right}"


@dataclass(frozen=True)
class Gt(Atom):
    """Greater than: left > right."""
    left: Expr
    right: Expr

    def evaluate(self, board, bindings):
        return self.left.evaluate(board, bindings) > self.right.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "gt", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self):
        return f"{self.left}>{self.right}"


@dataclass(frozen=True)
class Le(Atom):
    """Less than or equal: left <= right."""
    left: Expr
    right: Expr

    def evaluate(self, board, bindings):
        return self.left.evaluate(board, bindings) <= self.right.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "le", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self):
        return f"{self.left}<={self.right}"


@dataclass(frozen=True)
class Ge(Atom):
    """Greater than or equal: left >= right."""
    left: Expr
    right: Expr

    def evaluate(self, board, bindings):
        return self.left.evaluate(board, bindings) >= self.right.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "ge", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self):
        return f"{self.left}>={self.right}"


@dataclass(frozen=True)
class InBounds(Atom):
    """Check if (rank, file) is within board dimensions."""
    rank: Expr
    file: Expr

    def evaluate(self, board, bindings):
        r = self.rank.evaluate(board, bindings)
        f = self.file.evaluate(board, bindings)
        return 0 <= r < board.shape[0] and 0 <= f < board.shape[1]

    def to_dict(self):
        return {"type": "in_bounds", "rank": self.rank.to_dict(), "file": self.file.to_dict()}

    def __repr__(self):
        return f"in_bounds({self.rank},{self.file})"


# =============================================================================
# Layer 3 — Clauses & Conjunctions
# =============================================================================

class Clause(ABC):
    """A single element in a conjunction: atom, ∃, or ¬."""

    @abstractmethod
    def matches(self, board: np.ndarray, bindings: Dict[str, Any]) -> bool:
        """Evaluate this clause."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""

    @staticmethod
    def from_dict(d: dict) -> "Clause":
        """Deserialize from dict."""
        t = d["type"]
        if t == "atom":
            return AtomClause(Atom.from_dict(d["atom"]))
        if t == "exists":
            body = [Clause.from_dict(c) for c in d["body"]]
            return ExistsClause(d["var"], d["domain"], body)
        if t == "not":
            body = [Clause.from_dict(c) for c in d["body"]]
            return NotClause(body)
        if t == "or":
            branches = [[Clause.from_dict(c) for c in branch] for branch in d["branches"]]
            return OrClause(branches)
        raise ValueError(f"Unknown clause type: {t}")


@dataclass
class AtomClause(Clause):
    """Wraps a single atom as a clause."""
    atom: Atom

    def matches(self, board, bindings):
        return self.atom.evaluate(board, bindings)

    def to_dict(self):
        return {"type": "atom", "atom": self.atom.to_dict()}

    def __repr__(self):
        return repr(self.atom)


@dataclass
class ExistsClause(Clause):
    """Existential quantifier: ∃var ∈ domain: body.

    Domain is one of "squares", "ranks", "files".
    Body is a list of clauses that must all hold for some binding.
    """
    var: str
    domain: str  # "squares", "ranks", "files"
    body: List[Clause]

    def matches(self, board, bindings):
        for val in self._iter_domain(board):
            new_bindings = {**bindings, self.var: val}
            if all(c.matches(board, new_bindings) for c in self.body):
                return True
        return False

    def _iter_domain(self, board: np.ndarray):
        rows, cols = board.shape
        if self.domain == "squares":
            for r in range(rows):
                for c in range(cols):
                    yield (r, c)
        elif self.domain == "ranks":
            yield from range(rows)
        elif self.domain == "files":
            yield from range(cols)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def to_dict(self):
        return {
            "type": "exists",
            "var": self.var,
            "domain": self.domain,
            "body": [c.to_dict() for c in self.body],
        }

    def __repr__(self):
        body_str = " ∧ ".join(repr(c) for c in self.body)
        return f"∃{self.var}∈{self.domain}: ({body_str})"


@dataclass
class NotClause(Clause):
    """Negation: ¬(body). Combined with ∃ gives ∀ via ¬∃¬."""
    body: List[Clause]

    def matches(self, board, bindings):
        return not all(c.matches(board, bindings) for c in self.body)

    def to_dict(self):
        return {"type": "not", "body": [c.to_dict() for c in self.body]}

    def __repr__(self):
        body_str = " ∧ ".join(repr(c) for c in self.body)
        return f"¬({body_str})"


@dataclass
class OrClause(Clause):
    """Disjunction: matches if ANY branch matches.

    Each branch is a list of clauses (a conjunction).
    The overall OrClause is a DNF: OR of ANDs.
    """
    branches: List[List[Clause]]

    def matches(self, board, bindings):
        return any(
            all(c.matches(board, bindings) for c in branch)
            for branch in self.branches
        )

    def to_dict(self):
        return {
            "type": "or",
            "branches": [[c.to_dict() for c in branch] for branch in self.branches],
        }

    def __repr__(self):
        parts = []
        for branch in self.branches:
            parts.append("(" + " ∧ ".join(repr(c) for c in branch) + ")")
        return " ∨ ".join(parts)


@dataclass
class Conjunction:
    """Ordered list of clauses, all must match (logical AND)."""
    clauses: List[Clause]

    def matches(self, board: np.ndarray, bindings: Optional[Dict[str, Any]] = None) -> bool:
        b = bindings or {}
        return all(c.matches(board, b) for c in self.clauses)

    @property
    def num_atoms(self) -> int:
        """Count total atoms (depth-1 count for specificity)."""
        count = 0
        for c in self.clauses:
            if isinstance(c, AtomClause):
                count += 1
            elif isinstance(c, (ExistsClause, NotClause)):
                count += len(c.body)
            elif isinstance(c, OrClause):
                for branch in c.branches:
                    count += len(branch)
        return count

    def to_dict(self) -> dict:
        return {"clauses": [c.to_dict() for c in self.clauses]}

    @staticmethod
    def from_dict(d: dict) -> "Conjunction":
        return Conjunction([Clause.from_dict(c) for c in d["clauses"]])

    def __repr__(self):
        return " ∧ ".join(repr(c) for c in self.clauses)


# =============================================================================
# Layer 4 — Predicate (conjunction + score stats)
# =============================================================================

@dataclass
class Predicate:
    """A conjunction with associated score statistics from matching boards."""
    conjunction: Conjunction
    counts: Counts = (0.0, 0.0, 0.0)
    support: int = 0
    variance: float = 1.0
    mining_score: Optional[float] = None  # avg of the signal used during mining

    def matches(self, board: np.ndarray) -> bool:
        return self.conjunction.matches(board)

    @property
    def mean_score(self) -> float:
        """Best available score: mining signal if available, else raw counts."""
        if self.mining_score is not None:
            return self.mining_score
        return Stats(*self.counts).mean_score

    @property
    def raw_mean_score(self) -> float:
        """Raw mean from win/tie/loss counts (ignores mining signal)."""
        return Stats(*self.counts).mean_score

    @property
    def specificity(self) -> int:
        return self.conjunction.num_atoms

    def __repr__(self):
        return (f"({self.conjunction}) → score={self.mean_score:.3f}, "
                f"var={self.variance:.4f}, n={self.support}")


# =============================================================================
# Layer 5 — Mining (Phase 1: absolute atoms only)
# =============================================================================

def _sq(r: int, c: int) -> MakeSq:
    """Helper: create a literal square expression."""
    return MakeSq(Literal(r), Literal(c))


def _board_at(r: int, c: int) -> BoardAt:
    """Helper: create BoardAt with literal square."""
    return BoardAt(_sq(r, c))


def _from_board_at(r: int, c: int) -> FromBoardAt:
    """Helper: create FromBoardAt with literal square."""
    return FromBoardAt(_sq(r, c))


def _derive_mining_params(
    n_boards: int,
    base_variance: float,
    n_atoms: int,
) -> Dict[str, Any]:
    """Derive mining parameters from dataset statistics.

    - min_support: 2 (minimum for variance computation). Overfitting
      protection comes from variance reduction (refuses to split when
      improvement is unreliable) and 1 SE pruning (merges statistically
      insignificant siblings). Empirically verified: min_support=1
      produces identical results to min_support=sqrt(N) because the
      tree self-regulates — small nodes can't demonstrate reliable
      variance reduction so they become leaves naturally.

    - max_atoms: Hard cap at 6 to bound compute. In practice the
      variance reduction stopping criterion terminates growth at
      2-4 atoms for most patterns.
    """
    return {
        "min_support": 2,
        "max_atoms": 6,
    }


def _score_variance(matching: list, hash_scores: Dict[str, float]) -> float:
    """Compute raw score variance across matching board hashes."""
    if len(matching) < 2:
        return 0.0
    scores = [hash_scores[h] for h in matching]
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


# =============================================================================
# Layer 5 — Tensor-Accelerated Mining (GPU/CPU)
# =============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Layer 5 — Predicate Library (persistence + matching)
# =============================================================================

class PredicateLibrary:
    """Stores discovered predicates in DB and matches boards against them."""

    def __init__(self, conn: sqlite3.Connection, read_only: bool = False):
        self._conn = conn
        self._read_only = read_only
        self._predicates: List[Predicate] = []
        self._load()

    def _load(self) -> None:
        """Load predicates from database."""
        try:
            rows = self._conn.execute(
                "SELECT atoms_json, wins, ties, losses, support, score_variance, mining_score "
                "FROM predicates ORDER BY score_variance ASC"
            ).fetchall()
        except sqlite3.OperationalError:
            return  # Table doesn't exist yet

        self._predicates = []
        for atoms_json, w, t, l, support, var, ms in rows:
            data = json.loads(atoms_json)
            conjunction = Conjunction.from_dict(data)
            self._predicates.append(Predicate(
                conjunction=conjunction,
                counts=(w, t, l),
                support=support,
                variance=var,
                mining_score=ms,
            ))

    def match(self, board: np.ndarray, from_board: Optional[np.ndarray] = None) -> Optional[Stats]:
        """Find tightest matching predicate for a board transition.

        Returns Stats from the most specific predicate (most atoms),
        breaking ties by lowest variance.

        Args:
            board: destination (post-move) board
            from_board: source (pre-move) board for cross-board predicates
        """
        bindings = {"_from": from_board} if from_board is not None else {}

        best: Optional[Predicate] = None
        for pred in self._predicates:
            if not pred.conjunction.matches(board, bindings):
                continue
            if best is None:
                best = pred
            elif pred.specificity > best.specificity:
                best = pred
            elif pred.specificity == best.specificity and pred.variance < best.variance:
                best = pred

        if best is not None:
            return Stats(*best.counts)
        return None

    def match_all(self, board: np.ndarray, from_board: Optional[np.ndarray] = None) -> List[Predicate]:
        """Return all matching predicates, sorted by specificity then variance."""
        bindings = {"_from": from_board} if from_board is not None else {}
        matches = [p for p in self._predicates if p.conjunction.matches(board, bindings)]
        matches.sort(key=lambda p: (-p.specificity, p.variance))
        return matches

    def save(self, predicates: List[Predicate]) -> None:
        """Replace all stored predicates with new ones."""
        if self._read_only:
            return

        cur = self._conn.cursor()
        cur.execute("DELETE FROM predicates")
        for pred in predicates:
            atoms_json = json.dumps(pred.conjunction.to_dict())
            cur.execute(
                "INSERT INTO predicates (atoms_json, wins, ties, losses, support, score_variance, mining_score) "
                "VALUES (?,?,?,?,?,?,?)",
                (atoms_json, pred.counts[0], pred.counts[1], pred.counts[2],
                 pred.support, pred.variance, pred.mining_score),
            )
        self._conn.commit()
        self._predicates = list(predicates)

    @property
    def count(self) -> int:
        return len(self._predicates)

    @property
    def predicates(self) -> List[Predicate]:
        return list(self._predicates)


# Re-exports for convenience
from wise_explorer.memory.tree_miner import TreeMiner  # noqa: E402
from wise_explorer.memory.iti_miner import ITIMiner  # noqa: E402
