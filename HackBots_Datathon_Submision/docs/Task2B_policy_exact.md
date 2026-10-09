# Task 2B - Peak-day allocation policy, scenario S1 (Peliyagoda) - exact solver

**Result (lexicographic policy):** 77/85 orders served, 313.83 of 409.9 m3; 19/26 chilled orders; all 10 outlets skipped yesterday served; 25 trips on 15 vehicles; no ambient stop is planned after its window closes; 1 stop(s) on reefers' second Fresh trips arrive late only by a literal clock (the booklet's 270-minute budget already allows for the return leg). Passes the official `check_allocation.py` and an independent re-check of every rule. Solved exactly (reefer MILP, proven optimal; ambient subset DP): about a minute for the clock-aware model, about 4 seconds for the fast model.

## 1. What limited service
- **Reefer volume:** chilled demand 181.6 m3 vs 86.2 m3 for the 4 available reefers per wave (172.4 m3 even with two full trips each; five reefers are in the workshop).
- **Pre-dawn minutes:** each vehicle has 270 min; serving every chilled order would need more reefer minutes and trips than exist. Far districts (Puttalam 173 min, Matara 137, Galle 103 outbound) consume a large share of one vehicle's window for one or two stops.
- Ambient trucks and vans are not binding (24 vehicles for about 17 trips).

## 2. Priority policy (strict order, each level optimised and then fixed)
1. Orders **deferred yesterday** - never miss an outlet twice in a row.
2. **Chilled** orders - the most chilled orders served (outlets kept in stock for the festival build-up).
3. Highest **days since last served**.
4. Volume delivered, as a tie-break.
Ambient orders are all served unless physically impossible. Trips are planned so that no stop arrives after its window closes; constraints are as in the booklet (brand+district per trip, reefer for chilled, vans for van_only, capacity, 270/480-minute budgets, whole orders, at most 2 trips).

## 3. Deferrals
| Order | Outlet | District | Type | m3 | Def. yday | Days | Why |
|---|---|---|---|---|---|---|---|
| S1-003 | OUT002 | Colombo | chilled | 1.8 | 0 | 1 | reefer capacity |
| S1-058 | OUT054 | Galle | chilled | 16.5 | 0 | 1 | reefer capacity |
| S1-064 | OUT060 | Matara | chilled | 6.8 | 0 | 1 | reefer capacity |
| S1-067 | OUT062 | Matara | chilled | 5.2 | 0 | 1 | reefer capacity |
| S1-071 | OUT065 | Kurunegala | chilled | 12.0 | 0 | 1 | reefer capacity |
| S1-073 | OUT066 | Kurunegala | chilled | 5.8 | 0 | 1 | reefer capacity |
| S1-075 | OUT067 | Kurunegala | chilled | 7.2 | 0 | 2 | reefer capacity |
| S1-078 | OUT070 | Kurunegala | ambient | 40.7 | 0 | 2 | unavoidable: larger than any compatible vehicle |

- **Unavoidable (1):** S1-078 is larger than any compatible vehicle (cannot be split); ask the outlet to split it or hire a larger truck.
- **Chosen (reefer capacity):** which chilled orders wait is our policy decision. Serving the Puttalam outlet skipped yesterday costs a 188-minute trip, so 7 other orders wait instead: they become first-time misses.

## 4. Cost of the choice (exact comparison)
Weighted score instead of the written order: 79 orders, 328.3 m3, 21 chilled, but 1 outlet skipped for a second day (['S1-003', 'S1-071', 'S1-073', 'S1-075']). The score's deferred-yesterday weight (20) is below the value (about 28) at which it would choose the same plan.

## 5. Recommendations
- Release a reefer from the workshop: the what-if table above gives the exact orders recovered for each of VEH001, VEH002, VEH004, VEH005, VEH035.
- Serve tomorrow's carried-over chilled orders first (they will be `deferred_yesterday = 1`).
- Ask OUT070 to split its 40.7 m3 Style order.
