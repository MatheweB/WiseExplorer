# Getting the mover, and extending to N-player / arbitrary-turn games

There are **two independent questions** hiding here. Keeping them apart is the
whole trick:

1. **Who moved?** — the *mover* of each transition. Needed to enumerate the right
   replies (a board's replies are the side-to-move's legal moves) and to score a
   terminal for whoever reached it.
2. **How do a board's child values combine into its own value?** — the *backup*.

We solved **(1) generally** (it works for any number of players and any turn
order). **(2)** is currently solved only for 2-player zero-sum; the general form
is *maxn*, and it's the deferred piece.

---

## Part 1 — Getting the mover: just record it

The key realization: **at play time we already know whose turn it is**
(`current_player`). We don't have to *infer* it later from the board — we write
it down once.

A self-play game is a chain of moves. Here's a deliberately ugly 4-player order —
`p1, p4, p2, p4` — irregular, repeats p4, skips p3 entirely (a "non-monotone"
turn order):

```
   board:   B0 ──move──▶ B1 ──move──▶ B2 ──move──▶ B3 ──move──▶ B4   (terminal)
           (start)
 who moved:      p1           p4           p2           p4
                  │            │            │            │
                  └─ store ────┴─ store ────┴─ store ────┘
                     boards.to_move:  B0→p1   B1→p4   B2→p2   B3→p4
```

We store, per board, **the player who moves *out* of it** (`to_move`). That's it.
No turn-order rule, no alternation assumption, no reading the piece encoding.

The value loop needs two facts per board — both read back in O(1):

```
   to_move(B)    = boards.to_move[B]             "who acts at B"
   just_moved(B) = boards.to_move[ parent(B) ]   "who moved INTO B"
                   (parent = any recorded  parent ──▶ B  edge)
```

Applied to the chain above:

```
   board   to_move(stored)   parent   just_moved = to_move(parent)
   ─────   ───────────────   ──────   ───────────────────────────
   B0      p1                 —        —          (root: nobody moved in)
   B1      p4                 B0       p1
   B2      p2                 B1       p4
   B3      p4                 B2       p2
   B4      — (terminal)       B3       p4    ← terminal scores for p4
```

**Why this is fully general.** Nothing here assumes players alternate, or that
there are two of them, or how pieces are encoded. `p1→p4→p2→p4` is fine. A player
moving twice in a row is fine. 7 players is fine. We recorded *exactly* who moved,
so there's nothing to get wrong.

> This is why the old approaches broke: *guessing* the mover from the board —
> "the placed piece's value is the seat" — only held for Tic-Tac-Toe. Minichess
> encodes pieces by sign, so the guess returned a piece *type*, not a player.
> Recording sidesteps all of that.

---

## Part 2 — The backup: where N-player actually bites

Knowing the mover is enough to *enumerate the right replies*. But to turn child
values into a parent value, we run a backup — and **that's** where the current
code assumes 2 players.

### Negamax (what we have now) — one number per board

```
   Each board stores ONE value:  V = "value to whoever just moved in".

   V(board) = 1 − max over (next player's replies) V(reply)
                   └──────────────── "good for the next mover = bad for me"
```

That `1 − (…)` is a **2-player zero-sum** assumption: my value and the next
mover's value are exact complements (`x` and `1−x`). It's exact for TTT, Nim,
chess — anything strictly two-sided.

### Maxn (the general form) — a *vector* per board

```
   Each board stores a VECTOR:  V = (V_p1, V_p2, …, V_pN)   one entry per player.

   Backup:  the player to move M picks the child that maximizes THEIR OWN V_M;
            the board copies that child's WHOLE vector.
            "value to just-moved" = V[ the player who moved in ]
```

For 2 players this *collapses back* to negamax (because `V_p2 ≡ 1 − V_p1`, the
vector is redundant and `1−max` falls right out). So negamax isn't wrong — it's
the 2-player special case of maxn.

### Why one number isn't enough — a 3-player counterexample

```
                 R        (p1 to move)
                ╱ ╲
          A   ╱     ╲   B
       [1,0,0]        X        (p2 to move)   ← p1 just moved in
       p1 wins        │
                      │  (p2's only move)
                      Y        (p3 to move)   ← p2 just moved in
                     ╱ ╲
                C  ╱     ╲  D
            [0,1,0]       [0,0,1]
            p2 wins        p3 wins

   maxn (correct), vectors are (p1,p2,p3):
     Y: p3 maximizes V_p3  → picks D → Y = [0,0,1]
     X: only child is Y                → X = [0,0,1]
     value to p1 (who moved into X) = X[p1] = 0     ✓  p1 LOSES (p3 took it)

   negamax 1−max (wrong here):
     V(X) = 1 − max(V(Y)) = 1 − 0 = 1               ✗  claims p1 WINS
     it reasoned "p2 does badly ⇒ p1 does well" — but a THIRD player took the
     prize, so p1 AND p2 both lost. One scalar literally cannot express
     "both of us lost"; a per-player vector can.
```

The same single-scalar flaw shows up even with **2 players** if the turn order
isn't strict alternation — e.g. if `p1` moves twice in a row, `1−max` wrongly
"flips" against p1 on the second move (treating p1 as its own opponent). Maxn has
no flip: each mover just maximizes their own component, whoever they are.

---

## Where we are

| piece | status | general over… |
|---|---|---|
| **getting the mover** (store `current_player` per board) | **shipped** | any player count, any turn order, any encoding |
| **backup = `1 − max`** (negamax) | shipped | **2-player, zero-sum, alternating only** |
| **backup = maxn vectors** | deferred | any player count / turn order / payoffs |

The mover machinery is already general — it would feed an N-player game correctly
today. The only thing that would need to change for real N-player / non-monotone
*values* is swapping the scalar `1 − max` backup for the vector `maxn` backup
(and α, the cooperative/adversarial blend, retires into it). We hold off until a
genuine multiparty game is on the table, because for every game we actually run,
the scalar is exact and the vector is dead weight.
