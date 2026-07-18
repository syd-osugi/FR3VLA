# RealHandApi Function Reference (In-Depth)

This is an in-depth summary of all functions exposed by `RealHandApi` in `RealHand/real_hand_api.py`, plus the behaviors documented in `doc/API-Reference.md`.

**Initialization**
1. `RealHandApi.__init__(hand_type="left", hand_joint="L10", modbus="None", can="can0")`:
   - Loads config from `RealHand/config/setting.yaml` via `LoadWriteYaml`.
   - Prints SDK version and selects the correct hardware driver class based on `hand_joint` and `modbus`.
   - Hand ID: left is `0x28`, right is `0x27`.
   - If on Linux and `modbus == "None"`, opens CAN via `OpenCan` and exits if the interface isn’t up.
   - Queries embedded hardware version and warns if it’s missing.

**Motion Control**
1. `finger_move(pose=[])`:
   - Sets target joint positions.
   - Validates numeric range `0–255`.
   - Enforces model‑specific pose length:
     - O6/L6: 6
     - L7: 7
     - L10: 10
     - L20/G20: 20
     - L21/L25: 25
   - Writes to `self.hand.set_joint_positions()` when length matches.
   - Stores `self.last_position`.

2. `set_speed(speed=[100]*5)`:
   - Sets motor speed.
   - Validates numeric range `0–255`.
   - Requires at least 5 values, or 7 for L7.
   - Calls `self.hand.set_speed(speed=...)`.

3. `set_joint_speed(speed=[100]*5)`:
   - “Topic” speed setter (alternate path).
   - Validates numeric range `10–255` (note lower bound 10 here).
   - No length checks beyond non‑empty.
   - Calls `self.hand.set_speed(speed=...)`.

4. `set_torque(torque=[180]*5)`:
   - Sets maximum torque (grip force).
   - Validates numeric range `0–255`.
   - Requires at least 5 values, or 7 for L7, or exactly 6 for L6/O6.
   - Calls `self.hand.set_torque(torque=...)`.

5. `set_current(current=[250]*5)`:
   - Sets motor current **only for L20**.
   - Validates numeric range `0–255`.
   - Calls `self.hand.set_current(...)` on L20; otherwise no‑op.

**State and Feedback**
1. `get_state()`:
   - Returns current joint state list via `self.hand.get_current_status()`.

2. `get_state_for_pub()`:
   - Returns `self.hand.get_current_pub_status()` (publishing‑format state).

3. `get_speed()`:
   - Returns current speed list from `self.hand.get_speed()`.

4. `get_joint_speed()`:
   - Returns speed, but with special formatting for L20:
     - L20 uses a 20‑element list mapping partial speed values into a larger vector.
   - For most models, returns `self.hand.get_speed()` directly.

5. `get_torque()`:
   - Returns current torque limits from `self.hand.get_torque()`.

6. `get_temperature()`:
   - Returns motor temperature list via `self.hand.get_temperature()`.

7. `get_fault()`:
   - Returns fault codes via `self.hand.get_fault()`.
   - Fault categories are defined in `doc/API-Reference.md` (overload, over‑temp, encoder error, over/under voltage).

8. `clear_faults()`:
   - Only supported on L20.
   - Calls `self.hand.clear_faults()`; otherwise returns `[0]*5`.

9. `get_embedded_version()`:
   - Returns hardware version array from `self.hand.get_version()`.

10. `get_current()`:
    - Returns motor current via `self.hand.get_current()`.

**Touch / Force Sensing**
Not supported:
- O6 Base Model.
- L25 `get_touch()` returns `[-1] * 6` (explicit in code).

1. `_get_normal_force()`:
   - Internal helper that calls `self.hand.get_normal_force()`.

2. `_get_tangential_force()`:
   - Internal helper that calls `self.hand.get_tangential_force()`.

3. `_get_tangential_force_dir()`:
   - Internal helper that calls `self.hand.get_tangential_force_dir()`.

4. `_get_approach_inc()`:
   - Internal helper that calls `self.hand.get_approach_inc()`.

5. `get_force()`:
   - Calls all four helpers above, then returns `self.hand.get_force()`.
   - This combines normal pressure, tangential pressure, tangential direction, and proximity/approach signals (per `doc/API-Reference.md`).

6. `get_touch_type()`:
   - Returns tactile sensor type if present; returns `-1` or `None` when unsupported (model‑dependent).

7. `get_touch()`:
   - Returns tactile data per finger (and palm if supported).

8. `get_matrix_touch()`:
   - Returns matrix touch data via `self.hand.get_matrix_touch()`.

9. `get_matrix_touch_v2()`:
   - Returns matrix touch data (alternate API) via `self.hand.get_matrix_touch_v2()`.

10. `get_thumb_matrix_touch(sleep_time=0)`:
    - Returns thumb matrix touch data.
    - If `sleep_time > 0`, passes it through to the driver.

11. `get_index_matrix_touch(sleep_time=0)`:
    - Same as above, for index finger.

12. `get_middle_matrix_touch(sleep_time=0)`:
    - Same as above, for middle finger.

13. `get_ring_matrix_touch(sleep_time=0)`:
    - Same as above, for ring finger.

14. `get_little_matrix_touch(sleep_time=0)`:
    - Same as above, for little finger.

**Motor Enable/Disable (L25 Only)**
1. `set_enable()`:
   - Enables motors via `self.hand.set_enable_mode()` for L25.

2. `set_disable()`:
   - Disables motors via `self.hand.set_disability_mode()` for L25.

**Model Metadata**
1. `get_finger_order()`:
   - Returns motor/finger order via `self.hand.get_finger_order()` for L21/L25/G20.
   - Returns empty list for other models.

**Unit Conversion Helpers**
1. `range_to_arc_left(state, hand_joint)`:
   - Converts left-hand joint “range” to arc using `utils.mapping`.

2. `range_to_arc_right(state, hand_joint)`:
   - Converts right-hand joint “range” to arc using `utils.mapping`.

3. `arc_to_range_left(state, hand_joint)`:
   - Converts left-hand arc to range.

4. `arc_to_range_right(state, hand_joint)`:
   - Converts right-hand arc to range.

**Utility**
1. `show_fun_table()`:
   - Calls `self.hand.show_fun_table()` (driver-provided function table).

2. `close_can()`:
   - Calls `self.open_can.close_can0()` to shut down CAN.
