# Nav-command sweep results (P-13, 2026-09-26)

Cold single-frame inference, CUDA, 10 mini scenes. L2 vs GT per cmd head.

| scene | gt_y3 | intent   | L2(cmd0) | L2(cmd1) | L2(cmd2) | winner |
|-------|-------|----------|----------|----------|----------|--------|
| 0 | +3.67 | left     |  8.50 | 10.69 |  7.94 | cmd2 |
| 1 | -0.10 | straight |  7.89 |  9.35 |  8.25 | cmd0 |
| 2 | -0.00 | straight |  4.53 |  3.12 |  0.02 | cmd2 |
| 3 | +0.31 | straight | 12.91 | 15.04 | 13.93 | cmd0 |
| 4 | -0.00 | straight |  4.19 |  2.02 |  0.01 | cmd2 |
| 5 | +0.04 | straight | 21.47 | 23.75 | 22.75 | cmd0 |
| 6 | -0.47 | straight | 12.04 | 12.11 | 11.92 | cmd2 |
| 7 | +0.16 | straight | 26.27 | 28.21 | 27.34 | cmd0 |
| 8 | +0.35 | straight | 14.76 | 16.84 | 15.83 | cmd0 |
| 9 | -0.06 | straight |  2.89 |  1.27 |  0.54 | cmd2 |

Means: cmd0 11.55 / cmd1 ~12.7 / cmd2 10.85. cmd1 never wins.

Probe (`cmd_probe.py`): scene 0 (left turn, GT end [16.3, 7.3]) ->
cmd0 end [3.7, 13.6], cmd1 end [-4.4, 12.9], cmd2 end [-0.8, 6.8].
Scene 2 (stopped, GT~0) -> cmd0 [4.2, 7.6], cmd1 [-2.2, 5.7],
cmd2 [-0.0, 0.0] (near-perfect static).

Conclusion: no consistent straight/left/right mapping exists in this
checkpoint's behavior -- all heads curve +y on the left turn, and the
cmd effect (+-1-3 m) is second-order to scene difficulty (0.01-28 m).
Intent conditioning is weak under cold single-frame inference.
Keep default [1,0,0]; revisit under N-2 warm inference. Do NOT wire
GT-derived intent into calibration on an unverified mapping.
