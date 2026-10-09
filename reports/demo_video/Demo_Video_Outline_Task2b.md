# Demo Video Outline – Task 2B (Peak-Day Allocation)

The demo video is a single 3–5 minute unlisted YouTube video covering the whole project. This outline is the Task 2B segment, **about 75–90 seconds**. Tasks 1 and 2A have their own segments.

Figures below come from `Task2B_policy.md`. Re-check them against the latest solver run before recording.

## Timing

| Time | Section | On screen |
|---|---|---|
| 0:00–0:10 | The problem | Booklet scenario text, or one slide: "S1, Peliyagoda, festival week, demand exceeds fleet" |
| 0:10–0:25 | Data preparation | `Task2B_Data_Preprocessing.md`; fleet file showing 28 available and 10 in workshop |
| 0:25–0:40 | What limits service | Policy table of demand against capacity |
| 0:40–1:05 | Architecture and method | `architecture/task2B_architecture.png`, walked left to right |
| 1:05–1:20 | Results and validation | `check_allocation.py` passing; summary numbers |
| 1:20–1:30 | Challenges and trade-offs | Deferral table (`Task2B_policy.md` §4) |

## Talking points

### 1. The problem (10 s)
- One dispatch day, 85 orders, Peliyagoda depot, festival one week away.
- Fresh demand (dairy, meat, produce) is rising, and 10 vehicles are in the workshop.
- No model is trained. This is an allocation and prioritization problem judged on feasibility and reasoning.

### 2. Data preparation (15 s)
- Only vehicles marked `available` are used, which is 28 of 38.
- Orders join to vehicle specs. Trip time is outbound + inter-stop × (n − 1) + handling allowance, with no return leg.
- Fresh trips share a 270-minute budget. Style and Tech trips share 480 minutes. Each vehicle gets at most two trips.

### 3. What limits service (15 s)
- Chilled volume is the binding constraint. Demand is 181.6 m³, and four available reefers carry 86.2 m³ in one wave.
- Pre-dawn minutes are the second constraint. Far districts use 103–173 minutes outbound, so a far trip leaves room for at most one short second trip.
- Ambient capacity is ample, so ambient orders are not the bottleneck.
- Even with every reefer doing two full loads, chilled demand could not all move. So some chilled volume had to stay behind; which orders stayed was our choice.

### 4. Method (25 s)
- **Priority value:** 10 per stop, plus volume, plus a chilled bonus, plus a large bonus if the outlet was deferred yesterday, plus a bonus for days since last served.
- **Stage 1, reefers first:** enumerate every feasible day plan per reefer, then branch-and-bound searches for the best combination (3 million node cap, so it is the best plan found, not a proven optimum).
- **Stage 2, ambient packing:** pack remaining orders per brand and district, Fresh first.
- **Stage 3, repair:** insert deferred orders wherever they fit, then try 1-for-1 swaps for higher-value orders.
- **Windows:** stops are sequenced by earliest window close, and late arrivals are simulated and penalized.

### 5. Results and validation (15 s)
- 79 of 85 orders served (92.9%), 328.3 of 409.9 m³ (80.1%).
- 23 trips on 13 of the 28 available vehicles.
- Chilled service: 21 of 26 orders (140.7 m³).
- Show `check_allocation.py` passing. State that this confirms feasibility, not optimality.

### 6. Challenges and trade-offs (10 s)
- Balancing fairness, so that large orders don't starve small outlets (hence the per-stop term), against product volume.
- Reefers are scarce and time-limited, so the choice of which chilled orders to defer matters most.
- 8 stops are simulated to arrive late. Windows are soft in this plan, and that is a deliberate trade-off.
- Show the deferral table in `Task2B_policy.md` §4 (or the final notebook) and say which deferrals were unavoidable and which were a choice.

## Recording tips
- Keep the Task 2B segment to one continuous screen share: documentation, diagram, then validator output.
- Say every number out loud once, and leave it on screen for two seconds.
- Don't show raw dataset files in full. The data terms prohibit sharing the datasets, so show only aggregates, headers or a few rows.
- The booklet asks the video to cover architecture, preprocessing, label construction and challenges. For Task 2B, say plainly that there is no label construction because no model is used.
