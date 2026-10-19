# Project rules
- Never claim something works without running it and showing real output.
- Never loosen a test tolerance to make a test pass. Investigate the cause first.
- Preserve the CV/IR separation: vision/ outputs pixel-space strokes (480x480, top-left origin, Y down); mapping stays in coordinate_transform.py.
- Disturbances act only on the simulated plant. The vision corrector may use ONLY the TopDownVisionSensor image, target strokes and the nominal model (enforced by an AST test; keep it passing).
- Settle tolerance stays 0.25 deg. Nominal (no disturbance) baseline must reproduce RMSE 0.318 mm, Chamfer 0.228 mm.
- CoppeliaSim must be open with the scene loaded. Do not edit the .ttt directly; scene changes go through scripts/setup_scene.py (idempotent). Do not run two simulation jobs at once.
- Report honest results, including cases where a method does not help.
- After each task: what changed, what was verified, what is next.