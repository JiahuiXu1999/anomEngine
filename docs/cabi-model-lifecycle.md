# C ABI model lifecycle

anomEngine separates model-package assembly from data-driven artifact fitting.
This keeps externally trained graphs such as YOLO and EfficientAD out of the
PatchCore/PaDiM fitting workflow.

## Capability discovery

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

## PatchCore and PaDiM fitter

`anom_fitter_create` loads preprocessing and backend settings from a template
package. The template may omit the generated PatchCore index or PaDiM
statistics, but its graph and output bindings must be usable.

```c
anom_fitter_options_t options = {0};
options.struct_size = sizeof(options);
options.template_package_utf8 = "models/patchcore-template";
options.patchcore_coreset_sampling_ratio = 0.1f;

anom_fitter_t* fitter = NULL;
anom_fitter_create(&options, &fitter);
anom_fitter_add_batch(fitter, images, image_count);
anom_fitter_save_checkpoint(fitter, "work/patchcore.ckpt");
anom_fitter_finalize(fitter, "models/patchcore/1.0.0");
anom_fitter_destroy(fitter);
```

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

Fitter calls are synchronous. `anom_fitter_get_progress` can be polled from a
supervising thread, and `anom_fitter_cancel` requests cancellation between
backend batches. Except for cancellation and progress polling, a fitter must
not be called concurrently.

## Calibration

`anom_calibrator_create` creates an inference session from any opened model.
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

All public structures must be zero-initialized and have `struct_size` set. All
opaque handles are owned by the caller and released with their matching
`*_destroy` function. Strings returned in algorithm and model information are
owned by anomEngine and remain valid for the documented handle lifetime.
