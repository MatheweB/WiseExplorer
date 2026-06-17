# Certified forgetting: rules replace transitions

The transition table is a **cache, not the ground truth.** The ground truth is the game —
a pure function you can call any time; every stored transition is a memoized observation of
it. A rule, in turn, is an **executable claim** about that function ("boards with nim-sum 0
are winning"), so the game itself can confirm or refute it on demand.

This licenses a strong form of forgetting: once a rule's value for a region is *proven
against the game*, the stored rows in that region carry no information beyond the rule and
can be deleted. In MDL terms the database is exactly the residual of the two-part code:

$$\text{DB size} \;\approx\; |\,\text{data} \mid \text{theory}\,| \qquad\Rightarrow\qquad \text{a true theory drives it to zero.}$$

Statistics flow *into* rules ([concept-invention.md](concept-invention.md)); proven rules
*replace* their statistics. What remains stored is precisely what the theory cannot yet
account for.

Entry points: `TransitionMemory.frontier_certify` (the proof pass) and
`TransitionMemory.collapse_proven` (deletion).

## Where it runs

Prove + forget is the **cheap tier** of the value loop ([value-loop.md](value-loop.md)): it runs
**every wave**, so the certified frontier creeps forward continuously. It needs no refit — only
the reply graph — which is why it can run so often. (The expensive rebuild that *discovers* the
rules runs only on the games-doubling clock; certify + collapse just ride on top.)

```mermaid
flowchart LR
    classDef ev fill:#0e7490,stroke:#155e75,color:#ecfeff
    classDef di fill:#065f46,stroke:#047857,color:#d1fae5
    classDef pr fill:#713f12,stroke:#a16207,color:#fef9c3
    classDef cut fill:#9a3412,stroke:#7c2d12,color:#ffedd5
    P(["self-play<br/>every wave"]):::ev --> F["frontier_certify<br/>prove by induction from terminals"]:::pr
    F --> C["collapse_proven<br/>delete the rows the proofs reproduce"]:::cut
    C --> M[("what's left =<br/>the unproven residue")]:::di
    M -.->|"steers the next wave<br/>of exploration"| P
```

## Proof by induction — the frontier

A claim like "the creator of board `b` wins" is recursive: it means *every* opponent reply
from `b` can be answered by a move into a position that is *also* a creator-win. So instead
of proving the whole game tree at once, the frontier certifies **one ply at a time, anchored
at the terminals**:

> A board is **proven** once every legal reply is proven (or terminal), and its value is then
> the exact backup `1 − max(reply values)`.

Terminal boards are proven directly from the game's verdict. Depth-1 boards certify against
the terminals; depth-2 against the now-proven depth-1 layer; the certified frontier creeps
backward from the end of the game, one exact step at a time. A board proven this way carries
a **proof**, not an estimate. On Nim the frontier amounts to proving Bouton's theorem by
induction over the certificate set.

**Nothing the theory believes enters the check.** The theory only *nominates* boards to
test; the game supplies the legal moves and the terminal verdicts; prior certificates supply
the inductive hypothesis. There is no play policy anywhere — nothing is sampled, no seat
"plays" — so there is nothing for a wrong theory to bias, and *every* reply is enumerated,
including the refutations a theory-guided opponent would never play. This is why an inductive
certificate is a fact, not a probability. (`frontier_certify` reuses the value loop's
`reply_graph` enumeration and runs as one vectorized fixpoint sweep — zero playouts.)

Earlier prototypes certified interior boards with **adversarial rollouts** (k bilateral
rule-guided playouts per board). That works but is probabilistic — two seats sharing one
wrong theory can blunder symmetrically and "confirm" a false claim (~20% leak measured on
an inverted library) — and it costs thousands of playouts per cycle. The inductive frontier
replaced it: proofs instead of probabilities, and the verification bill dropped to zero.

## The deletion invariant

> A row may be deleted iff a **proof** reproduces its value.

`collapse_proven` deletes a transition only where its completed value (`propagated_score`)
matches the certificate of the board it lands on, within `ε = 0.25`:

```sql
DELETE FROM transitions WHERE propagated_score IS NOT NULL
  AND EXISTS (SELECT 1 FROM certificates c WHERE c.board_hash = transitions.to_hash
              AND ABS(transitions.propagated_score - c.value) <= ε)
```

Two properties make this safe:

- **Sound under a wrong theory.** The comparison is against the *certified* value — the
  game's, never the library's. A corrupted theory cannot cause a wrongful deletion, because
  the theory's prices are not consulted here at all.
- **Surgical, not blanket.** Rows whose value the proof *cannot* reproduce are kept — they
  mark stale beliefs or genuine exceptions. Deletion removes redundancy, not evidence.

`WISE_COLLAPSE=0` disables deletion; the cycle still completes and values stay sound, just
with redundant rows retained.

## Proofs need no expiry

A certificate here is a game-theoretic **fact**, which dissolves machinery earlier designs
needed. A fact cannot churn, so it requires no aging before it earns deletion rights and no
revocation pass when the theory is refit — both of which existed only to manage *probabilistic*
rollout certificates. And because completion pins proven boards to their certified values, a
proof *improves* the targets discovery fits, rather than merely sitting beside them.

## Steering: the residue is the itinerary

The stored residue is a map of where the theory is incomplete, and exploration reads it.
Training-time selection drives each move by its **total remaining uncertainty** — statistical
noise and theory–evidence disagreement in quadrature, and **zero once the board is proven**:

$$\text{drive} = \begin{cases} 0 & \text{proven — nothing left to learn} \\ \sqrt{\,\mathrm{se}^2 + (\text{concept} - \text{stat})^2\,} & \text{the theory makes a claim} \\ \mathrm{se} & \text{the theory is silent} \end{cases}$$

This is parameter-free and **direction-blind**: the theory enters only through the
*magnitude* of its disagreement with the evidence, so it pulls games toward boards where it
is informative and untested — never toward boards it merely favors. A confidently wrong claim
attracts exactly the games that refute it; a confirmed claim fades to plain `se`, then to
zero at proof. Measured on Tic-Tac-Toe, steering discovers ~1.5× more new boards per game than
uniform exploration with no loss of play strength.

**The bright line.** The theory may choose *where* recorded games go; it must never choose
*which move is good* inside a recorded game. Steering toward a wrong claim multiplies the
traffic that can disprove it; letting the theory *play* the recorded games would instead
suppress the refuting replies and feed its own errors back as evidence (measured: recorded
rule-guided play echoes — the clean-room collapse). So steering tilts the sampler; the moves
themselves stay uncertainty-driven, and the raw counts stay independent of the theory.

## Self-healing

Forgetting does not require certainty — only two properties the system already has:
**error detectability** (a wrong rule cannot suppress the observations that convict it, since
training is theory-blind) and **evidence recoverability** (the game is replayable, so a
wrongful deletion costs re-exploration, never knowledge).

The hard test: corrupt the theory *after* its supporting rows are gone. Play craters — on
Nim-4, where the table is empty, it falls to 8/96. One training cycle refits the rules from
fresh evidence (the corrupted rule does not survive contact with the observations that
disagree with it) and play returns to 96/96. The proofs never wavered through any of it —
they are facts, and deletion stayed sound against them the whole time.

## Measured behavior

| | complete theory (Nim-4) | partial theory (Tic-Tac-Toe) |
|---|---|---|
| frontier proves | all 120 stored boards, one sweep | 2,192 at setup → 5,234 over four cycles |
| memory | **594 rows → 0**, every cycle | 7,108 → ~3,000 (empties only where proven) |
| play | 96/96 optimal throughout | 234/300, steady |
| verification cost | **zero playouts** | **zero playouts** |
| corruption gate | 8/96 → 96/96 in one cycle | 51/300 → 223/300 in one cycle |

On a solved game the table empties; on a partial theory it empties *exactly where the theory
holds* and stays dense elsewhere — the database becomes a map of the theory's blind spots,
and a free, exact audit of it (`|library(b) − proven(b)|` over proven boards measured the
TTT library mispricing ~30% of them — matching the ~31% the rollout prototype found by
playing games).

**The evidence ladder beats the old arbitration.** Ablating the deleted machinery (hiding
the four-signal stack's bell + anchors from competitive selection, leaving only
proven > concept > statistics) was an A/B from identical Tic-Tac-Toe snapshots, same
seeds. The ladder *improved* play at every checkpoint (237 vs 231 at setup, 238 vs 226
and 234 vs 226 over two cycles) and degraded *less* under corruption (80/300 vs 51/300) —
because the old bell signal is filled by the library where coverage runs out, so a
corrupted theory flows through it, while raw counts are theory-blind and cannot be
poisoned. Anchors earned nothing measurable. This is why competitive play ranks the
ladder and `bell` survives only as discovery's internal fit target.

**Scale (1,000-game probes).** Nim-6 (5,040 positions): theory found, 2,567 boards proven,
40% collapsed, corruption recovered in one cycle. Nim-8 (362,880 positions, single run,
replication pending) taught two things: steered discovery found the nim-sum in ~1,600 games
and reached 200/200 play, where the unsteered baseline finds nothing in 3,000; and the
audit's authority is proportional to frontier coverage — with proofs over only a 0.7%
near-terminal shell, a shell-fitting theory can pass the audit while playing poorly. The
safety floor held throughout: a weak theory is never granted deletion authority, because
collapse answers to proofs, not to the library.

## Honest limits

- **Frontier-bounded.** A board certifies only once its whole reply layer is certified, so
  deep claims wait. On a coverage-limited game (8-pile Nim) the frontier reaches only the
  endgame shell; that is also where statistics are weakest, so the two cover different
  regions rather than the same one.
- **Cyclic games.** The induction assumes replies chain to terminals. Positions with
  repetition (minichess) never satisfy the all-replies-proven condition and are simply left
  unproven — `solve_graph`'s value iteration still handles their values; a fixpoint variant
  of the frontier is the open path.
- **Zero-sum proofs.** The proof backup uses the pure `1 − max` form; the cross-player
  alignment factor α ([value-loop.md](value-loop.md)) would need threading through the
  induction for non-zero-sum proofs.
- **Seat legality.** The ∀-over-replies side tolerates a superset (conservative); the
  ∃-our-response side needs genuinely legal moves — exact for Nim and Tic-Tac-Toe.

## Where it lives

| piece | place |
|---|---|
| inductive proof pass | `TransitionMemory.frontier_certify` |
| proof-licensed deletion | `TransitionMemory.collapse_proven` |
| proven-board pinning in completion | `TransitionMemory.complete_values` |
| certificate store / cache | `GameMemory.certified_values` · `certificates` table |
| the steering drive | `selection.select_move_for_training` |
| cycle ordering (prove + forget last) | `GameMemory.grow_concepts` |
| deletion off-switch | `WISE_COLLAPSE=0` |
