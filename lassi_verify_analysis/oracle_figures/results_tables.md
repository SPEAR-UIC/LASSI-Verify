## Headline — soundness & completeness (four axes, never pooled)

| Axis | n | TA | FA | OS | TR | Soundness | Completeness |
|---|---:|---:|---:|---:|---:|---:|---:|
| Verify x LASSI | 350 | 318 | **0** | 19 | 13 | 1.000 | 0.944 |
| LLM Judge x LASSI | 350 | 309 | **3** | 28 | 10 | 0.769 | 0.917 |
| Verify x zero-shot | 37 | 25 | **1** | 8 | 3 | 0.750 | 0.758 |
| LLM Judge x zero-shot | 37 | 33 | **1** | 0 | 3 | 0.750 | 1.000 |

Soundness = 1 - FA/(FA+TR); Completeness = TA/(TA+OS).

## Per-kernel — framework & judge on the LASSI population

| Kernel | n | FW TA | FW FA | FW OS | FW TR | JD TA | JD FA | JD OS | JD TR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entropy | 22 | 22 | 0 | 0 | 0 | 22 | 0 | 0 | 0 |
| extrema | 12 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0 |
| all-pairs-distance | 22 | 22 | 0 | 0 | 0 | 22 | 0 | 0 | 0 |
| atomicCost | 20 | 11 | 0 | 9 | 0 | 3 | 0 | 17 | 0 |
| threadfence | 19 | 19 | 0 | 0 | 0 | 19 | 0 | 0 | 0 |
| layout | 12 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0 |
| bsearch | 24 | 23 | 0 | 1 | 0 | 24 | 0 | 0 | 0 |
| randomaccess | 26 | 17 | 0 | 7 | 2 | 23 | 0 | 1 | 2 |
| colorwheel | 27 | 26 | 0 | 0 | 1 | 26 | 0 | 0 | 1 |
| chacha20 | 27 | 22 | 0 | 0 | 5 | 13 | 0 | 9 | 5 |
| floydwarshall | 27 | 27 | 0 | 0 | 0 | 27 | 0 | 0 | 0 |
| murmurhash3 | 23 | 22 | 0 | 0 | 1 | 22 | 0 | 0 | 1 |
| jaccard | 23 | 23 | 0 | 0 | 0 | 23 | 0 | 0 | 0 |
| lid-driven-cavity | 24 | 19 | 0 | 2 | 3 | 21 | 2 | 0 | 1 |
| segment-reduce | 21 | 21 | 0 | 0 | 0 | 20 | 0 | 1 | 0 |
| pathfinder | 21 | 20 | 0 | 0 | 1 | 20 | 1 | 0 | 0 |
