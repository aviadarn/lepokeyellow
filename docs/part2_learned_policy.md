# Part 2: a learned policy under a play-time contract

The second project: replace Part 1's planner with a policy that sees only
pixels at play time, and measure what it reaches. Back to the
[README](../README.md).

## The harness

Part 1 ([part1_world_model_planner.md](part1_world_model_planner.md)) steers
with demonstration waypoints and arrival read from RAM.
That is a fair description of a planner, not of a policy, and the honest
version of the next step is a contract rather than a claim. The design is in
[docs/superpowers/specs/2026-10-21-part2-world-model-rl-design.md](superpowers/specs/2026-10-21-part2-world-model-rl-design.md);
this section is what has been built of it.

**At play time** the policy sees pixels and which objective it is on, and
nothing else. No coordinate, map-id or wall reads, and no save states.
`poke/contract.py` makes that a rule the code enforces: the policy is handed a
`PlayTimeEnv` whose surface is `screen`, `read_ram`, `game` and `act_frames`,
and whose `read_ram` refuses the position addresses and the bytes beside them.
Anything else raises. **At training time** anything goes -- RAM rewards,
save-state resets, oracle routes.

```bash
# Which WRAM byte or screen feature says the player has control? Measured
# against a snapshot-and-compare probe, fitted on five states, held out on one.
uv run python -m scripts.calibrate_control --samples 250

# The BFS oracle: milestone save states and the ceiling. Training only.
uv run python -m scripts.oracle --state states/bedroom.state \
    --replay results/play_milestones_s1.json

# Play Part 2 under the contract. One continuous run, no rollback.
uv run python -m scripts.play2 --policy random --budget 1461

# How fast does this machine emulate in parallel?
uv run python -m scripts.bench_workers --steps 300 --workers 1 2 4 8 12
```

### The learned share is an upper bound, not a measurement

Every action is logged with the controller that chose it, and `learned_share`
is the policy's share of the actions somebody owned. Two measured facts bound
how much that number is worth.

The control probe is an approximation. No WRAM byte reproduces the
snapshot-and-compare answer: scoring every byte in WRAM, "zero means control"
topped out at 0.797 balanced accuracy, a best-value-set rule reached 0.878 with
winners that scored 0.08 on a held-out state, and excluding battle frames
lifted the best byte to 0.966 -- but every candidate sat in `0xDFEE-0xDFFB`,
which is the stack. It was reading the return address of whatever routine
happened to be running. The detector shipped first was the white fraction of
the screen's bottom six rows, cut at 0.7725: 0.9503 fit, 0.9835 held out.

**Those numbers were true and the detector was wrong.** Whiteness was never
the signal. Pallet Town's entrance is 82% white -- whiter than plenty of real
boxes -- so 36 of 967 walkable samples read as a text box, and a trained
policy later spent 1,411 of its 1,461 actions on one of them. A Gen-1 box is
drawn in two shades, white paper and black frame-and-glyphs; scenery uses all
four the DMG has. The detector is now the shade count with a white floor:
**0.9683 fit against the old rule's 0.9497 on the same samples, and 0 of 967
walkable samples called a box, against 36.** `scripts/calibrate_box.py`,
`results/control_calibration_v2.json`.

The zero is the number that matters, not the percentage point. A walkable tile
read as a box is *absorbing*: the handler owns every turn it claims and only
the policy can step off a tile, so one bad tile ends the run.
`poke.handlers.BOX_PATIENCE` bounds that at one policy turn in thirty-three,
which is what kept the earlier failure survivable -- a Part 1 run spent 2,714
of 4,000 actions on a single Pallet Town square -- but one turn in
thirty-three is not control, and bounding the cost was mistaken for fixing it.

There is no honest held-out number for the new rule, and the report says so.
The ground truth is the rollback probe: press two buttons, compare screens. In
a menu the d-pad moves a cursor, so two buttons give two screens and the probe
reports control with a box plainly on screen. 87.5% of the holdout state's
"walkable" samples are menus, and both rules score 0.5106 there -- that
measures the probe, not the detector.

And a walking cutscene draws no box, so the probe says the player has control,
the turn goes to the policy, and it is counted as learned work. That is the
direction of the error: the share is inflated, not deflated.

### The ceiling is a number, not a run

The BFS oracle reaches Viridian Forest from Route 1 in **487 actions**
(196 search edges; the rest is the A-mashing each edge spends waiting for
control), in 13 minutes of search. The policy arms get
3x that, 1461 actions.

It is not a baseline run, and `scripts/play2.py` offers no oracle policy. The
oracle's own action log does not reproduce its own route: replaying the parcel
phase's 87 actions from the Route 1 state ends at (12, 28, 9), still on Route
1, where the search ended at (42, 5, 2) in the Viridian Mart. The search probes
control by snapshotting and restoring between actions, and PyBoy save/load is
not bit-exact for whatever that depends on -- the same break measured earlier
in this repo, when `has_control()` inside `play.py`'s `step()` stopped the run
reproducing its own log at action 368.

### The floor

1461 actions each from Route 1 -- the same budget the policy arms get -- under
the contract: no constant button and no random policy reaches `get_parcel`,
`deliver_parcel` or `enter_forest`. Random spends 94% of its turns on the
policy and the rest in the handlers.
Objectives already true when a run starts are not credited -- Route 1 begins
with `trigger_oak` and `get_starter` already flagged, and the first version of
this table had all eight constant buttons "clearing" two objectives apiece.

### Parallel emulation is not the bottleneck

668 actions/s with one worker, 2,388 at four (efficiency 0.89), 3,490 at eight
(0.65), 4,003 at twelve (0.50). An action is 60 frames, so four workers emulate
about 143k frames/s while the encoder does 1,758. The RL budget was going to
assume the emulator was the expensive part. It is the forward pass.

## Under the contract: the policy does not finish it

One arm was trained, the design's arm 3: the frozen JEPA encoder's features as
the observation, plus a Plan2Explore-style intrinsic bonus from the same world
model's latent prediction error. Arms 1 and 2 were not run, by scope ruling, so
**nothing here shows the world model helped.** The comparison that would show it
is a separate plan.

### What was trained

Nine wall-clock hours on an M-series Mac: **8,211 PPO updates**, **8.41M
environment steps**, 20,548 episodes, 259 steps/s across 8 emulator workers with
an fp16 encoder. The run is a resume -- the weights continue an earlier 0.85 h
leg of 730 updates and 747,520 steps, and the archive started empty because it
was memory-only until after that pause. The report said "8,941 updates" for
nine hours by printing an inherited counter beside a fresh one; `run_report`
now reports both separately. The policy is a 597k-parameter actor-critic -- a linear body
into a GRU cell, 7 actions -- reading a 4-frame window of 192-dim embeddings,
the objective one-hot and the previous action. Episodes are 512 actions, started
from a Go-Explore archive keyed by `(objective, map, y//4, x//4)` and seeded
from `states/oracle/route1.state`.

In training the policy did reach `get_parcel`, and the archive grew to 168 cells
across 11 maps, the Viridian Mart among them. It never reached `enter_forest`;
the archive only ever held cells under objectives 2 and 3.

### What it scored

Eleven contract runs from Route 1 -- ten sampling seeds and one greedy -- each a
single continuous episode with the 1,461-action budget, no rollback, nothing read
but pixels:

| | `get_parcel` | `deliver_parcel` | `enter_forest` |
|---|---|---|---|
| oracle ceiling (487 actions, uses rollback) | reached | reached | reached |
| trained policy, 10 seeds + greedy | **0/11** | **0/11** | **0/11** |
| floor: random, and each of 8 constant buttons | 0 | 0 | 0 |

The policy scores the floor. Median learned share is 0.969, and the control
detector that produces it was the white-fraction rule at **0.9503 fit, 0.9835
held out** -- which is the rule the section above shows was trapping this very
run; the share is an
upper bound, as above, so "the policy chose 97% of these actions" is itself
generous.

### What the GIFs show

Two failure modes, and the per-run controller counts separate them cleanly.

Seven seeds spend almost the whole budget under policy control (1,385-1,451 of
1,461 actions) and walk **north** -- up Route 1, into Viridian City, around the
Pokémon Center and the Mart until the budget ends.

North is the right way. An earlier version of this section said `get_parcel`
was south in Oak's lab and that the north walk was the tile bonus leading the
policy astray. Both are wrong, and the objective's own description says so:
"cross Route 1 to Viridian and pick up Oak's Parcel". The parcel is handed over
in the Viridian Mart, map 42, which is where the oracle's first phase ends.
`deliver_parcel` is the leg that goes back south to Oak's lab. So those seven
seeds reached the right city and never completed the pickup -- the parcel needs
the player inside the Mart, at the counter, pressing A -- which is a different
and smaller failure than walking the wrong way.

Three seeds and the greedy run go south into Pallet instead, which is the wrong
direction for the first objective, and are then taken away from the policy
entirely. Greedy
spends 49 of 1,461 actions on the policy and 1,411 in the text and decline
handlers; its GIF holds two distinct frames.

That is not the policy pressing A at a sign. **The control detector
false-positives on the tiles where Route 1 enters Pallet Town.** Walking south
puts the player at (0, 4, 11); the bottom six rows there are Pallet's pale
ground and a roof, 0.818 white against the 0.7725 cut, and no text box is on
screen at all. The detector says a box is open, so `choose()` claims the turn,
and `BOX_PATIENCE = 32` hands the policy one turn in thirty-three -- which is
the measured 49/1,461. The policy's one action is `down`, into the house wall
below it. The state is absorbing.

It is a false positive, not a real box: from that state `up` moves the player to
y=3 and `left` moves them to x=9. A real box freezes both. Over a 900-action
handler-free random walk, 5 of 147 visited Pallet tiles read as a box -- (3,8),
(3,10), (4,3), (4,8), (4,9), plus (4,11) above -- against 0 of 80 on Route 1 and
the two houses. The band sits at y=3-4, exactly where you arrive from Route 1.

`BOX_PATIENCE` was added for this and it is why the run ends at 1,461 actions
instead of never, but 1-in-33 control is not control. (0, 4, 3) is in that list:
it is the same tile the harness section above names as the case that cost one
Part 1 run 2,714 of 4,000 actions. The cut was recalibrated; the tile still fails.

### The clone: knowing the route is not walking it

208 demonstrations cloned into the same network, fitted to a training accuracy
of 1.0, then run under the contract: **0 of 11**, same as the floor.

The eleven runs are not eleven runs. Cross-entropy to a loss of 0.0005 leaves
the logits nearly one-hot, so sampling and argmax agree and every seed replays
the same trajectory -- identical controller counts, identical learned share,
0.980 in all eleven. That is worth knowing before reading a clone's seed
spread as evidence of anything.

What it does: five correct decisions. From the Route 1 start it walks north,
the right way, from (12, 35, 10) to (12, 28, 9) -- and then oscillates between
(12, 28, 9) and (12, 28, 11) for the remaining four hundred actions.

This is the compounding-error failure that pure behaviour cloning always has,
arriving earlier than expected. The dataset is one optimal trajectory per
objective; it contains no frame showing how to recover, because the oracle
never made a mistake to recover from. Six decisions in, the clone is on a
screen its training set does not contain, and nothing it learned applies.

The learned share is the part worth keeping: **0.980**, against 0.034 for the
runs the old detector trapped. The policy now owns 1,432 of 1,461 actions.
Whatever is wrong is the policy's, not the harness taking its turns.

So two different failures have now been separated. The RL policy could not
find the route. The clone knows the route and cannot stay on it. Only the
second is what DAgger fixes -- clone, roll out, re-solve from the states the
clone actually reaches, and add those -- and the oracle can be re-run from any
state, which is what that needs.

### DAgger: the first objective a learned policy reaches, and how unstable that is

Roll the clone out under the contract, keep a save state behind each new
decision, restore the states it got stuck in and run the oracle from them,
retrain on everything. `scripts/dagger.py`, eight rounds, six searches each.

**It works, and the per-run number is not what it first looked like.** The
first eval of the five-round checkpoint scored `get_parcel` on 5 of 11 runs,
and that is what this section originally reported. Retraining the same dataset
under two more training seeds gives 0 and 5. The spread that matters here is
not the ten sampling seeds `scripts/eval_part2.py` already varies -- it is the
training seed, and one eval of one checkpoint does not measure the dataset
that produced it. Training is deterministic, so the first number reproduces
exactly; it was simply the best of three draws, reported as the result.

Three training seeds per condition, each scored over ten sampling seeds and a
greedy run, pooled by `scripts/seed_spread.py`:

| | per seed | `get_parcel` | vs. the clone |
|---|---|---|---|
| behaviour clone | 0, 0, 0 | 0 / 33 | -- |
| DAgger rounds 1-5 | 5, 0, 5 | **10 / 33** | p = 0.0009 |
| DAgger rounds 1-8 | 1, 3, 0 | 4 / 33 | p = 0.11 |

Fisher's exact, two-sided, which is the smallest honest test at these counts.
Five rounds of DAgger is the first thing in this repo that moves an objective
off zero by more than one draw's luck: the clone reaches `get_parcel` on none
of its thirty-three runs, and nine hours of PPO and every scripted baseline
reach it on none of theirs. Rounds 6 to 8 did not help, and may have hurt --
10/33 against 4/33 is p = 0.13, which three seeds cannot settle either way.

Distance on a seed that works, greedy from `states/oracle/route1.state`:

| | distinct tiles | best y on Route 1 | maps reached |
|---|---|---|---|
| behaviour clone | 6 | 28 | 12 |
| 2 rounds | 17 | 20 | 12 |
| 5 rounds | 69 | 0 | 12, 1, 42 |

Route 1 runs y=35 at the start down to y=0 at Viridian. The clone covered
seven of those thirty-five tiles and stopped; after five rounds it walks the
whole route, enters Viridian, enters the Mart and takes the parcel in 99
actions against the oracle's 87.

Three things the rounds taught on the way.

**The frozen encoder was not the bottleneck**, which was the obvious suspect:
it was measured recovering which room a frame is in and not where in the room,
and Route 1 is one room. `scripts/probe_observability.py` asks whether the
expert's button is a function of what the policy sees -- leave-one-out 1-NN
over the demonstrations. On the original set the encoder's window scores
**0.847** against a 0.549 majority baseline, *better* than the true (map, y,
x) position at 0.829, with 9 exactly-conflicting steps against position's 88.
A four-frame window carries the direction of travel, which a coordinate does
not, and a tile crossed by two different optimal routes genuinely wants two
different buttons.

**The search budget was going to the states that already worked.** Five of
round 2's six relabelled states came from turns 0-14, because almost every
distinct state of a stuck rollout happens before it gets stuck, and the first
selection rule spread evenly over distinct states. `by_time_spent` replaced
it: the states the rollout returned to most, which are the absorbing ones.
Rounds 3 and 4 then relabelled turns 13-21, and by round 5 the rollout reached
49 distinct states and the searches launched from turns 50-57 needed only
13-16 actions to finish -- the clone had walked almost the whole leg itself.

**Round 6 picked up the next objective on its own, and that did not help
either.** The loop reads the objective from event flags, so once a rollout
carried the parcel its searches became `deliver_parcel` searches: five of
round 6's six, ending in Oak's Lab, 227 new steps for a return leg that until
then had only its sixty original demonstrations. The suspicion was that the
new data contradicted the old, since both legs walk the same corridor in
opposite directions and 59 of the 289 demonstrated tiles do carry opposite
majority actions under the two objectives. The probe says no: leave-one-out
1-NN over the aggregated set goes *up* when rounds 6 to 8 are added, 0.882 to
0.898 against a 0.500 majority baseline, with 21 exactly-conflicting steps in
1,841. The sprite faces north on the way out and south on the way back, so the
frames separate the legs and the labels stay a function. Whatever those rounds
cost, it was not label noise.

A detail that mattered more than it looks. A relabelled episode starts in the
middle of a run, and replayed from a fresh GRU and a fresh frame window it is
a different input than that same game state is at play time, where the window
holds the last four frames and the hidden state carries hundreds of oscillating
steps -- which is exactly the state a recovery has to be conditioned on. So
each visit keeps the learner's own preceding sixteen turns and the episode
replays them before the expert's, moving the GRU and the window and scored on
nothing. Sixteen rather than the whole prefix is a memory and compute cap, and
a GRU warmed on sixteen oscillating steps approximates one warmed on four
hundred rather than equalling it.

What the loop cannot reach from here is structural rather than mysterious.
DAgger relabels only the objective the rollout is currently on, so the
dataset's shape follows how far the policy gets: after eight rounds it holds
44 episodes of `get_parcel`, 6 of `deliver_parcel`, and still exactly 1 of
`enter_forest` -- the original demonstration. `train_clone` takes one optimizer
step per episode, so the objective the design actually grades gets one
gradient step in fifty-one. Reaching it needs rollouts that begin past the
parcel, which means seeding the loop from oracle states at the later
objectives instead of only from Route 1. That is a training-time use of save
states, like every demonstration here; the eval stays one continuous episode
from `states/oracle/route1.state`.

### Why, as far as the logs say

The archive stopped growing at hour 6 -- 168 cells, unchanged for the last three
hours. Policy entropy is the weaker claim: over the whole leg it starts at
0.383, runs a median of 0.736 and ends at 0.312, against ln(7) = 1.95, and it
is not a decline but a wobble -- in the final twelve minutes alone it spans
0.220 to 1.214. An earlier version of this section said it "fell from 0.96 to
0.48", which is neither endpoint of anything; the committed report kept only
the last 200 of 8,941 rows, so the window it was read from was too short to
show a trend at all. The report keeps every row now. From an archive state near Oak's Lab it can collect the
parcel; from Route 1 it has no reason to go there, because the only signal
pointing south is a +1 that 8.41M steps of exploration never found from that
start.

**That is not a statement about this reward and this budget alone, and an
earlier version of this section said it was.** Three other suspects are live,
and none was ruled out:

- *The optimiser was broken.* A review found that GAE discarded every episode's
  terminal transition -- the step that pays `enter_forest` its +1 -- so the goal
  produced no gradient at all, and that PPO replayed each rollout from a zero
  GRU state while the driver carried the real one, making the importance ratio
  something other than 1 before any weight changed. Both are fixed; neither was
  fixed when this run happened.
- *The representation is untested.* The encoder ran **1 of its 6 epochs**,
  stopped to free wall-clock. Part 1's encoder reached a validation prediction
  loss of 0.0102; this one stopped at 0.0317.
- *The harness was trapping the route.* The control detector false-positived on
  the Pallet Town entrance, which is on the only path from the start state to
  `get_parcel`, so an episode heading for the objective lost 32 of every 33
  turns on arrival.

So this run does not show that the reward is the problem, and it certainly does
not show anything about the world model. Separating any of it needs the fixes
above and arms 1 and 2.

And it is confounded by the detector. The false-positive band sits at Pallet
Town's entrance, which is on the path to `deliver_parcel` -- the return leg to
Oak's lab -- so any episode that got as far as carrying the parcel home lost 32
of every 33 turns on arrival. It does not explain the first objective, which
lies north. Training episodes seeded from archive states already inside Pallet
or the lab are not taxed, which is part of how `get_parcel` was reached in
training and never in a contract run from Route 1. Fixing the detector comes
before reading anything more into these numbers.
