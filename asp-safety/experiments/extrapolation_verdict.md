# N-3(b) extrapolation verdict (2026-09-26): DROP

Dual sweep, 10 mini scenes, CUDA (`calibration_baseline.json` strict
vs `calibration_extrapolated.json` CV-extended to T=4,5):

| metric | strict 3 s | extrapolated 6 s |
|---|---|---|
| pred rule hits (measured T<=3) | lane_keeping 27, wrong_way 2 | same 27 + 2 |
| pred hits at invented T>=4 | n/a | lane_keeping +20 |
| pred fixed rate | 1.00 | 0.90 (corrector gives up) |
| corrector mean ms | ~95 | 562 |

The +20 extrapolated hits are constant-velocity-extension artifacts
(straight lines leaving curved lanes), not planner behavior. The
extended horizon also costs 6x corrector time and drops the fix rate.
Noise > signal: keep the honest 3 s shield (N-3a), report blind past
3 s. Revisit only with a learned (not CV) extension. N-3(c) model
head-extension spike remains Week-4 work.
