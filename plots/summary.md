# Benchmark summary

run_id `20261016T122113Z`, device `none`, compiler `gcc 15.2.0`. Times are medians in ms.

| impl | N | status | correct | time_ms_median | h2d_ms_median | d2h_ms_median | e2e_ms_median | gflops | speedup_vs_cpu | speedup_e2e_vs_cpu | max_abs_err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | 128 | ok | True | 0.948499 |  |  |  | 4.42204 | 1.0 |  | 6.23204e-06 |
| cuda_naive | 128 | not_run |  |  |  |  |  |  |  |  |  |
| cuda_tiled | 128 | not_run |  |  |  |  |  |  |  |  |  |
| cpu | 256 | ok | True | 7.96052 |  |  |  | 4.2151 | 1.0 |  | 1.31609e-05 |
| cuda_naive | 256 | not_run |  |  |  |  |  |  |  |  |  |
| cuda_tiled | 256 | not_run |  |  |  |  |  |  |  |  |  |
| cpu | 512 | ok | True | 102.738 |  |  |  | 2.61282 | 1.0 |  | 3.22277e-05 |
| cuda_naive | 512 | not_run |  |  |  |  |  |  |  |  |  |
| cuda_tiled | 512 | not_run |  |  |  |  |  |  |  |  |  |
| cpu | 1024 | ok | True | 2511.44 |  |  |  | 0.855081 | 1.0 |  | 5.88516e-05 |
| cuda_naive | 1024 | not_run |  |  |  |  |  |  |  |  |  |
| cuda_tiled | 1024 | not_run |  |  |  |  |  |  |  |  |  |
| cpu | 2048 | ok | True | 52923.2 |  |  |  | 0.324619 | 1.0 |  | 0.00011324 |
| cuda_naive | 2048 | not_run |  |  |  |  |  |  |  |  |  |
| cuda_tiled | 2048 | not_run |  |  |  |  |  |  |  |  |  |
