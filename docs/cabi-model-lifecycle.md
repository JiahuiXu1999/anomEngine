# C ABI model lifecycle

anomEngine separates model-package assembly from data-driven artifact fitting.
This keeps externally trained graphs such as YOLO and EfficientAD out of the
PatchCore/PaDiM fitting workflow.

## Capability discovery

Include the algorithm-specific C header, such as `anomEngine/patchcore.h`, or
use `anomEngine/anomEngine.h` for all algorithms. Algorithm objects expose
inference and supported fitting operations directly as C function-pointer members (ABI v4).

Use `anom_algorithm_get_count` and `anom_algorithm_get_info` before presenting
an algorithm in an application. `available` reports whether its runtime plugin
was built, while `capabilities` describes the public workflow supported by the
SDK build.

| Algorithm | Package import | Artifact fit | Incremental fit | Calibration |
|---|---:|---:|---:|---:|
| PatchCore | yes | yes | yes | yes |
| PaDiM | yes | yes | yes, from a fitter checkpoint | yes |
| Direct | yes | no | no | yes |
| EfficientAD | yes | no | no | yes |
| DFKDE | yes | not yet exposed | no | yes |
| SPADE | yes | not yet exposed | no | yes |
| YOLO | yes | no | no | yes |

EfficientAD weight distillation and YOLO supervised training remain external
to anomEngine. Their exported graphs are assembled with the package builder.

## Static model inspection

`anom_model_open` parses a package and exposes stable model metadata without
creating a runtime. `anom_model_validate` additionally checks every referenced
deployment artifact. Set `initialize_runtime` to validate plugin discovery,
backend initialization, tensor signatures, and algorithm assets as well.

## Package builder

`anom_package_builder_create` accepts an existing package or manifest path as a
template. `anom_package_builder_add_artifact` overlays files at safe
package-relative paths. `anom_package_builder_commit` then:

1. copies the template into a sibling staging directory;
2. applies artifact overlays;
3. generates SHA-256 entries for every packaged artifact;
4. validates the completed package; and
5. atomically renames the staging directory to the requested output path.

The output directory must not already exist and must not be inside the template
package. The builder never edits the template in place.

For YOLO and EfficientAD this is the complete preparation path: provide a
manifest template, overlay the externally exported ONNX graph, and commit.

## PatchCore and PaDiM fitting

PatchCore and PaDiM expose fitting methods directly on their algorithm structures. Their implementations
share preprocessing and backend machinery, but users never select an algorithm
through a generic fitter. A template may omit its generated index or statistics,
but its graph and output bindings must be usable.

```c
anom_patchcore_fitter_options_t options = {0};
options.struct_size = sizeof(options);
options.template_package_utf8 = "models/patchcore-template";
options.coreset_sampling_ratio = 0.1f;

anom_patchcore_t algorithm = {0};
algorithm.struct_size = sizeof(algorithm);
anom_status_t status = anom_patchcore_init(&algorithm);
if (status != ANOM_STATUS_OK) return status;
status = algorithm.create(&algorithm, &options);
if (status == ANOM_STATUS_OK)
    status = algorithm.add_batch(&algorithm, images, image_count);
if (status == ANOM_STATUS_OK)
    status = algorithm.save_checkpoint(&algorithm, "work/patchcore.ckpt");
if (status == ANOM_STATUS_OK)
    status = algorithm.finalize(&algorithm, "models/patchcore/1.0.0");

/* The same object can load the generated package and predict. */
anom_algorithm_options_t load_options = {0};
load_options.struct_size = sizeof(load_options);
load_options.device = ANOM_DEVICE_AUTO;
load_options.device_id = -1;
anom_prediction_t prediction = {0};
prediction.struct_size = sizeof(prediction);
if (status == ANOM_STATUS_OK) {
    status = algorithm.load(&algorithm, "models/patchcore/1.0.0", &load_options);
    if (status == ANOM_STATUS_OK)
        status = algorithm.predict(&algorithm, &images[0], &prediction);
}
anom_prediction_release(&prediction);
algorithm.release(&algorithm); /* Releases both inference and fitting resources. */
return status;
```

Initialization only fills the method table. `create` and `load` allocate their
own resources on demand and may be called in either order. `finalize` does not
load or replace the inference model, and does not release fitting state. Calling
`create` again while fitting state exists, or `load` while inference exists,
returns an error. `release` clears both so the object can be reused. A failed
creation/load preserves the other runtime. Use separate algorithm objects when
fitting and inference need independent release lifetimes.

PatchCore templates that already contain a FAISS index seed the fitter with
that index. New samples are combined with the imported vectors and coreset
sampling is run again at finalization.

PaDiM fitter checkpoints store the sample count, selected channel indices,
sums, and second moments. These are training-state artifacts and are distinct
from the compact deployment statistics containing means and precision
matrices. Exact PaDiM continuation requires the checkpoint.

If no PaDiM channel indices are provided, the fitter deterministically selects
the first `embedding_dimension` channels. Applications that require randomized
channel selection should provide explicit indices.

Fitting calls are synchronous. The algorithm's `get_progress`
method can be polled from a supervising thread, and `cancel`
requests cancellation between backend batches. Except for cancellation and
progress polling during fitting, serialize operations on the algorithm object.
Do not release or reinitialize the object until all calls have returned.

## Calibration

`anom_calibrator_create` creates an internal inference runtime from any opened model.
Feed representative normal images through `anom_calibrator_add_batch`, then use
`anom_calibrator_compute` to obtain image and pixel score ranges plus thresholds
for the requested target false-positive rate.

Pixel scores use deterministic reservoir sampling bounded by
`max_pixel_samples`; image scores are retained exactly. This prevents large
spatial maps from causing unbounded calibration memory growth.

Pass the result to `anom_package_builder_apply_calibration` before commit. The
builder writes raw score bounds and thresholds into the shared postprocessing
configuration. This is the in-engine preparation step for externally trained
YOLO, EfficientAD, and Direct graphs; they do not enter the artifact fitter.

## ABI ownership

All public structures must be zero-initialized and have `struct_size` set.
Initialize algorithm method tables with the matching `*_init` function.
Both inference and fitting runtimes are released through `object.release(&object)`; opaque model,
builder and calibrator handles use their matching `*_destroy` function. Strings returned in algorithm and model information are
owned by anomEngine and remain valid for the documented handle lifetime.
