# Policy Training Outline

This repo currently supports camera capture, localization, fusion, and
trajectory planning. A trained policy will need a separate data and runtime
path. The files in `policy/` and the new robot interface files are placeholders
for that path.

## Runtime Shape

1. Build a fresh `PolicyObservation`.
2. Run `policy.predict(observation)`.
3. Pass the action through `SafetySupervisor`.
4. Execute the approved action with `RobotInterface`.
5. Log observation, raw action, safety result, execution result, and task label.

## Files Added

- `policy/observation.py`: stable observation schema for camera, robot, target,
  and task data.
- `policy/actions.py`: stable action schema for learned or scripted policies.
- `policy/base.py`: policy interface plus a safe no-op placeholder policy.
- `policy/scripted.py`: executable adapter that uses the current trajectory
  planner as a replaceable policy-like module.
- `policy/dataset.py`: JSONL episode logger for demos and rollouts.
- `policy/inference.py`: one-step inference wrapper connecting policy, safety,
  robot execution, and logging.
- `policy/train.py`: placeholder training CLI and roadmap.
- `robot/robot_interface.py`: hardware boundary for real or no-op robot drivers.
- `robot/safety.py`: action validation and bounding before hardware execution.
- `robot/learning_controller.py`: future closed-loop policy controller shell.

## Main Work Still Needed

- Implement `FrankaRobotInterface` with a persistent robot connection.
- Implement synchronized observation providers for D435, D405, robot state, and
  automatic object/target tracking.
- Choose the policy action space and control frequency.
- Collect demonstration episodes with calibrated camera and robot metadata.
- Implement dataset loading, normalization, training, checkpointing, and eval.
- Add tests for safety limits, dataset schema validation, and controller stop
  behavior before enabling hardware motion.

## Executable Proof Of Concept

Run the current planner through the future policy/controller path:

```bash
python3 examples/policy_poc.py --target 0.4 0.1 0.2 --steps 3
```

This uses `ScriptedTrajectoryPolicy` plus `NoOpRobotInterface`, so it produces
policy actions, safety decisions, and execution records without moving hardware.

The intended module replacement path is:

1. Keep `ObservationBuilder`, `PolicyAction`, `SafetySupervisor`,
   `RobotInterface`, and `LearningFeedbackController` stable.
2. Replace `ScriptedTrajectoryPolicy` with a trained policy implementation.
3. Replace `NoOpRobotInterface` with `FrankaRobotInterface`.
4. Keep the controller loop and logging structure largely unchanged.
