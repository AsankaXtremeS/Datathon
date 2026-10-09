# Task 2B – Data Preparation

Task 2B (peak-day allocation, scenario S1, Peliyagoda) uses no trained model, so there is no label construction or feature engineering. Preparation consists of loading five files, removing unavailable vehicles, joining orders to vehicle and travel data, and computing trip time. This is implemented in `load()` and `trip_minutes()` in `solve_task2b.py`.

## 1. Inputs

| File | Used for |
|---|---|
| `task2b_peak_day_scenarios.csv` | 85 orders: outlet, brand, district, depot, dock type, parking constraint, delivery window, temperature requirement, weight, volume, `deferred_yesterday`, `days_since_last_served` |
| `task2b_peak_day_fleet.csv` | Vehicle status for S1 (`available` / `in_workshop`) |
| `vehicles.csv` | Vehicle type, temperature capability, weight and volume capacity, home depot |
| `district_travel.csv` | `depot_to_district_freeflow_min`, `inter_stop_freeflow_min` per district |
| `service_allowance.csv` | Handling minutes per brand and dock type |

## 2. In-workshop exclusion

The fleet file lists 38 vehicles: **28 `available` and 10 `in_workshop`**. Only vehicles with status `available` are kept; all `in_workshop` vehicles (for example VEH001, VEH002, VEH004, VEH005, VEH035) are dropped before any allocation, so they cannot appear in any trip. The exclusion is applied at load time, so every later stage sees only usable vehicles.

## 3. Joins

- **Fleet to vehicles:** the available `vehicle_id`s are joined to `vehicles.csv` to attach type (truck/van), temp (reefer/ambient), `weight_cap_kg`, `volume_cap_m3` and home depot.
- **Orders to outlet attributes:** no join is needed. The scenario file already carries brand, district, depot, dock type, parking constraint and window for each order. `order_ref` is the allocation key, because `outlet_id` can repeat.
- **Orders to travel and handling:** looked up by key at trip-building time. The district gives the travel times, and (brand, dock type) gives the handling allowance.

## 4. Derived fields

- Window times (`HH:MM`) are converted to minutes after midnight.
- `temp_requirement == chilled` becomes a boolean. Chilled orders need a reefer; reefers can also carry ambient goods.
- Trips are grouped as **pre-dawn** (Fresh, 270-minute budget) or **daytime** (Style and Tech combined, 480-minute budget).
- Each order gets a priority value: 10 per stop, plus 1 per m³, plus 5 if chilled, plus 20 if deferred yesterday, plus 4 per day since last served beyond the first (see the policy document for the rationale).

## 5. Trip-time calculation

Following the booklet, with no return leg:

```
trip_minutes = depot_to_district_freeflow_min
             + inter_stop_freeflow_min x (orders - 1)
             + sum(service_allowance_min[brand, dock_type] for each order)
```

Worked example, a Fresh trip to Gampaha with three orders (two rear dock, one street):
37 + 9 x 2 + 15 + 15 + 16 = **101 min**.

The result is checked against the vehicle's remaining budget: Fresh trips share 270 min and Style/Tech trips share 480 min.

## 6. Feasibility filters applied to prepared data

An order can be placed on a vehicle only if all of these hold:

1. The vehicle's home depot equals the order's depot.
2. A chilled order goes only on a reefer.
3. A `van_only` outlet goes only on a van.
4. Orders on one trip share one brand and one district.
5. Trip weight and volume stay within the vehicle's caps.
6. At most two trips per vehicle, within the time budgets above.

Orders are never split across trips or vehicles. The outputs are validated with the supplied `check_allocation.py`.
