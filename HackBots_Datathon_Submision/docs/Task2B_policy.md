# Task 2B - Peak-day allocation policy (S1, Peliyagoda)

**Result:** 79/85 orders served (92.9%), 328.3 of 409.9 m3 (80.1%). 23 trips on 13 of 28 available vehicles. Passes check_allocation.py.

## 1. What limits service today

| Resource | Demand | Available (one wave) | Verdict |
|---|---|---|---|
| Reefer volume (chilled) | 181.6 m3 / 32.8 t, 26 orders | 4 reefers = 86.2 m3 / 17.0 t (VEH001, VEH002, VEH004, VEH005, VEH035 in workshop) | **Binding** |
| Pre-dawn minutes on reefers | far districts use 103-173 min outbound | 270 min per vehicle | **Binding** - a far trip leaves room for at most one short trip |
| Van-only chilled (OUT001-003) | 1096 kg | 1 reefer van, 1040 kg per trip | needs both van trips |
| Ambient trucks/vans | 228.2 m3 | 673 m3 | ample |

Even if every reefer did two full loads (172.4 m3), chilled demand of 181.6 m3 could not all move, and the 270-min window prevents two trips to the far districts. We served **140.7 m3 chilled** (21/26 orders).

## 2. Priority rule

Value of serving an order = 10 per stop + 1 x m3 + 5 if chilled + 20 if deferred yesterday + 4 x (days since last served - 1); -3 per stop simulated to arrive after its window closes.

Rationale: chilled dairy/meat cannot be held and the festival is a week away; an outlet already skipped yesterday or unserved for 5 days is at real stock-out risk, so a second miss costs more than one large fresh delivery to a recently served outlet. Volume rewards product on shelf; the per-stop term keeps small outlets from being starved by big ones.

## 3. How the allocation was built

1. **Scarcest resource first** - every feasible day plan (<=2 trips, capacity, 270-min budget) was enumerated for each reefer, and branch-and-bound chose the plan combination with maximum priority value (proven optimal for this objective).
2. **Ambient fleet** - remaining orders packed per brand+district, Fresh first, preferring a fresh vehicle's first trip (on-time before 08:00) and keeping Style/Tech daytime trips on vehicles already used pre-dawn.
3. **Repair** - any deferred order is inserted wherever it fits; 1-for-1 swaps replace a lower-value order.
4. **Windows (soft)** - stops sequenced by earliest window close; departures 03:30 (Fresh) / 08:00 (daytime), return = outbound time. 8 of 79 stops simulate late.

**Reefer trips (the decisive part of the plan):**

| Vehicle | Trip | District | Stops | m3 / cap | kg / cap | Minutes (out + inter + handling) |
|---|---|---|---|---|---|---|
| VEH003 | 1 | Kurunegala | 3 | 25.0 / 26.4 | 4593 / 5510 | 127 + 38 + 45 = **210** (day 250/270) |
| VEH003 | 2 | Colombo | 1 | 11.7 / 26.4 | 1991 / 5510 | 24 + 0 + 16 = **40** (day 250/270) |
| VEH006 | 1 | Gampaha | 3 | 28.7 / 33.4 | 5250 / 6840 | 37 + 18 + 46 = **101** (day 210/270) |
| VEH006 | 2 | Colombo | 4 | 29.9 / 33.4 | 5513 / 6840 | 24 + 24 + 61 = **109** (day 210/270) |
| VEH007 | 1 | Gampaha | 3 | 19.3 / 19.4 | 3485 / 3610 | 37 + 18 + 45 = **100** (day 234/270) |
| VEH007 | 2 | Kalutara | 3 | 16.0 / 19.4 | 2930 / 3610 | 64 + 24 + 46 = **134** (day 234/270) |
| VEH036 | 1 | Colombo | 2 | 5.7 / 7 | 1032 / 1040 | 24 + 8 + 31 = **63** (day 127/270) |
| VEH036 | 2 | Colombo | 2 | 4.3 / 7 | 778 / 1040 | 24 + 8 + 32 = **64** (day 127/270) |

## 4. Deferrals - unavoidable vs. chosen, and their cost

| Order | Outlet | District | Type | m3 | Def. yday | Days | Why |
|---|---|---|---|---|---|---|---|
| S1-056 | OUT053 | Galle | chilled Fresh | 3.75 | 0 | 2 | **capacity (reefer)**: reefers full - lower priority than every served chilled stop it could replace |
| S1-058 | OUT054 | Galle | chilled Fresh | 16.52 | 0 | 1 | **capacity (reefer)**: reefers full - lower priority than every served chilled stop it could replace |
| S1-064 | OUT060 | Matara | chilled Fresh | 6.78 | 0 | 1 | **capacity (reefer)**: reefers full - lower priority than every served chilled stop it could replace |
| S1-067 | OUT062 | Matara | chilled Fresh | 5.19 | 0 | 1 | **capacity (reefer)**: reefers full - lower priority than every served chilled stop it could replace |
| S1-078 | OUT070 | Kurunegala | ambient Style | 40.66 | 0 | 2 | **unavoidable**: order is 40.66 m3 / 2562 kg; largest compatible vehicle VEH011 holds 38 m3 / 7200 kg |
| S1-083 | OUT074 | Puttalam | chilled Fresh | 8.66 | 1 | 5 | **capacity (reefer)**: reefers full - lower priority than every served chilled stop it could replace |

- **Unavoidable (1)**: physically larger than any available vehicle; must be split by the outlet/merchandising team or sent with a hired truck.
- **Capacity-driven, our choice of which (5)**: 40.9 m3 had to stay behind because reefer capacity is short; *which* orders stayed is our choice. We kept back the lowest-value chilled orders (none deferred yesterday) in the closest districts, which can be recovered first tomorrow.
- **Cost**: 81.6 m3 not delivered (40.9 m3 chilled), 6 outlets short today; these orders carry into tomorrow's peak.

## 5. Trade-off check (same solver, different priorities)

| Policy | Orders served | m3 served | Chilled m3 | Deferred-yesterday orders served |
|---|---|---|---|---|
| Chosen (priority-weighted) | 79 | 328.3 | 140.7 | 9 |
| Max volume only | 77 | 331.3 | 143.8 | 9 |
| Max order count only | 79 | 328.3 | 140.7 | 9 |

## 6. Recommendations

- Release a reefer from the workshop (VEH001, VEH002, VEH004, VEH005, VEH035): every extra reefer recovers roughly one deferred chilled load.
- Serve tomorrow's carried-over chilled orders first (they will be `deferred_yesterday = 1`).
- Ask OUT070 to split its 40.7 m3 Style order, or book a larger vehicle.
