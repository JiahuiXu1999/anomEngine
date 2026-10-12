# PatchCore variants through one C ABI object

`anom_patchcore_t` exposes `variant` at the outermost level:

| C enum | Manifest `algorithm.variant` | Fitted model |
| --- | --- | --- |
| `ANOM_PATCHCORE_ORIGINAL = 0` | `original` (default) | Original FAISS memory bank |
| `ANOM_PATCHCORE_MAHALANOBIS = 1` | `mh` | PCA/whitening statistics and a transformed FAISS bank |
| `ANOM_PATCHCORE_SUBSPACE_AD = 2` | `subad` | PCA mean and orthonormal basis; no index artifact |

All variants use the same `create`, `add_batch`, `get_progress`, `save_checkpoint`,
`load_checkpoint`, `cancel`, `finalize`, `load`, and `predict` function table.
The selected mode is immutable while either fitting or inference state exists.
Call `release` before changing it. `release` preserves the selected variant.

```c
#include <anomEngine/patchcore.h>

anom_patchcore_t detector = {0};
detector.struct_size = sizeof(detector);
detector.variant = ANOM_PATCHCORE_MAHALANOBIS; /* ORIGINAL or SUBSPACE_AD are also valid */
anom_status_t status = anom_patchcore_init(&detector);

anom_patchcore_fitter_options_t fit = {0};
fit.struct_size = sizeof(fit);
fit.template_package_utf8 = "models/mh-template";
if (status == ANOM_STATUS_OK)
    status = detector.create(&detector, &fit);
/* Feed normal images with detector.add_batch(&detector, images, count).
   Check each status, then detector.finalize(&detector, "models/mh-fitted"). */
detector.release(&detector);
```

For inference, select the corresponding enum and call `load` with the fitted
package. A package with a different variant returns `ANOM_STATUS_ALGORITHM_MISMATCH`
without retaining a session. Unknown enum values and changes to a live object's
variant are rejected. Initialization preserves a valid preselected enum.

The field consumes one previously reserved 32-bit slot. Object size, function
offsets, and ABI version 4 are unchanged. Zero-initialized old clients continue
using original PatchCore. New variants use manifest schema **2**, preventing old
engines from silently treating whitened features as ordinary PatchCore features.
Schema 1 packages without `variant` keep their original behavior.

## Fitting and model packages

Copy an [MH](../src/model/examples/patchcore-mh.manifest.json) or
[SubAD](../src/model/examples/patchcore-subad.manifest.json) example to the template
directory as `manifest.json`, and supply its ONNX feature graph. Generated statistics
and indexes do not need to exist in the template. The outer C enum chooses the
fitting mode, including when using an original PatchCore template; preprocessing
and feature aggregation are still taken from that template. Changing the enum
does not replace the backbone or alter its feature contract.

`finalize` saves the selected mode and required assets and builds checksums.
For MH/SubAD, it disables inherited normalization and thresholding because the
score distribution has changed. Calibrate the resulting package using the existing
calibration/package-builder APIs before using application thresholds.

| Manifest field | Default | Meaning |
| --- | --- | --- |
| `feature_aggregation` | `concat` | Concatenate feature maps, or `mean` for equal-shaped maps |
| `pca_components` | `0` | MH: min(128, D); SubAD: choose rank by explained variance |
| `explained_variance` | `0.99` | SubAD automatic PCA rank; strictly between 0 and 1 |
| `covariance_shrinkage` | `0.01` | MH shrinkage towards the mean retained variance |
| `covariance_regularization` | `0.000001` | Positive variance floor added before whitening |
| `memory_bank_size` | `1000` | Maximum number of MH reference patches |
| `streaming_chunk_size` | `1024` | Maximum raw patches replayed per MH construction step |
| `top_score_fraction` | `0.01` | SubAD image score: mean of the highest ceil(fraction × patches) residuals |

`embedding_dimension` always describes the graph's aggregated feature channels.
The MH index dimension is read from the fitted PCA statistics and may be smaller.
Original coreset options (`coreset_sampling_ratio`, `projection_dimension`,
`max_exact_distance_evaluations`, `random_seed`) apply to original PatchCore only.
MH uses deterministic farthest-first selection and its explicit bank budget.

MH fits full-covariance PCA with FP64 batch moment updates, regularizes the retained
variances, and whitens projected features. It replays raw features from an owned,
automatically closed temporary disk file, then merges and reduces the transformed
chunks with farthest-first selection. RAM is bounded with respect to the number
of training patches, but includes O(D²) covariance state; temporary disk usage is
O(ND). A statistical checkpoint includes the MH feature cache, so it can be large.
It can be restored after releasing the original object. A transformed deployment
bank is not used to seed a new statistical fit; use a fitting checkpoint instead.

SubAD accumulates mean and covariance without storing all training patches. At
finalization it selects the PCA basis. At inference it explicitly computes the
squared reconstruction residual in FP64 before returning FP32 patch scores. This
avoids subtracting two large squared norms. Both variants require at least two
normal **patches** (a single image can provide many). Statistics/checkpoints reject
incompatible modes, dimensions, truncated payloads, and non-finite values.

## Feature graphs and relation to the papers

The adapter consumes float32 NCHW maps. For SubspaceAD, an exported DINOv2 graph
should remove CLS/register tokens, average the chosen intermediate layers, and
reshape patch tokens to `[N, D, Hpatch, Wpatch]`. Set pooling to kernel 1, stride 1,
padding 0. Alternatively, export equal-shaped layer maps and select `mean`.
The included SubAD manifest describes a 672-pixel DINOv2-G embedding contract;
it does not include pretrained weights or an exported graph. Match resize,
interpolation, normalization, token/layer selection and augmentation to the graph
used during fitting. Rotation augmentation is supplied by the caller through
`add_batch`; the SDK does not rotate images automatically.

These are usable algorithm modes, not claims of reproducing paper benchmark
scores. The MH implementation uses full-covariance PCA and sequential merge/reduce,
instead of the paper's IPCA and hierarchical constructor. SubAD implements PCA
residual scoring; using an existing CNN backbone is an experimental feature
configuration rather than the paper's DINOv2 configuration. The shared PatchCore
plugin still has a build/runtime FAISS dependency even though SubAD inference
does not create or query a FAISS index. CPU ORT/C++ numerical tests do not establish
DINOv2 TensorRT export parity, industrial accuracy, or end-to-end throughput.

References: [MH-PatchCore](https://arxiv.org/abs/2605.27748),
[SubspaceAD](https://arxiv.org/abs/2602.23013).
