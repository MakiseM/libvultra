# Foveated 3DGS Handoff

This branch carries the current foveated Gaussian Splatting path on top of
`dev-next`.

## Build

Run this after cloning on a new machine:

```powershell
git submodule update --init --recursive
xmake f -y --renderdoc=false
xmake build example-gaussian-splatting
```

If `vshadersystem` binary package download fails, reuse a local XMake package
cache or provide a reachable package mirror. This branch currently uses
`vshadersystem v0.8.3`.

## Core Method

The paper path has three runtime pieces:

1. Log-polar foveated selection: smooth eccentricity-based keep probability.
2. Coverage texture guide: low-resolution alpha-density guide from projected
   Gaussian coverage.
3. Stable temporal release: stable hash thinning plus tile-local capped release
   for gaze-update flicker control.

The current implementation routes through the Gaussian preprocess shader:

- `builtin/shaders/passes/general/gaussian_splat_preprocess.comp.vshader`
- `builtin/shaders/include/common/gaussian_splat_scene.glsl`
- `builtin/shaders/include/common/gpu_scene.glsl`

The renderer-side state and resource plumbing lives in:

- `source/vultra/include/vultra/function/rendering/render_structs.hpp`
- `source/vultra/include/vultra/function/resource/gpu_scene_view.hpp`
- `source/vultra/src/function/rendering/render_system.cpp`
- `source/vultra/src/function/rendering/srp/builtin/passes/general_gaussian_splat_preprocess_pass.cpp`

The desktop entry point is:

- `examples/gaussian_splatting/main.cpp`

The OpenXR eye-gaze path is connected through:

- `examples/openxr/gaussian_splatting/main.cpp`
- `source/vultra/src/function/rendering/backend/render_backend_system.cpp`
- `source/vultra/src/function/openxr/xr_headset.cpp`
- `source/vultra/src/function/openxr/ext/xr_eyetracker.cpp`

## Recommended Interactive Command

Use a resource root containing `resources/scenes/fixed_gaze_smoke_scene.vmanifest`.
Example:

```powershell
xmake run -w D:\formal_3dgs_scene_assets\smoke_outputs\mip360_garden_hf_msplat example-gaussian-splatting `
  --scene resources/scenes/fixed_gaze_smoke_scene.vmanifest `
  --mouse-gaze `
  --shader-antipop-mode delayed-guide-temporal-logpolar-tile-cap `
  --shader-antipop-hash-seed 51001 `
  --coverage-texture-floor true `
  --coverage-texture-size 160,90 `
  --coverage-texture-d-min 4.0 `
  --coverage-texture-sigma-max 0.65 `
  --coverage-texture-strength 0.75 `
  --coverage-texture-history-beta 0.80 `
  --coverage-texture-update-interval 4 `
  --coverage-texture-debug false `
  --delayed-guide-temporal-release-policy tile-local-capped `
  --delayed-guide-temporal-release-cap-ratio 1.0 `
  --no-coverage-compensation `
  --no-foveated-temporal-hysteresis `
  --no-foveated-boundary-smoothing
```

Interaction:

- Mouse controls gaze when `--mouse-gaze` is enabled.
- Hold right mouse button and use `WASD/QE` for free camera movement.
- Shift increases camera speed; Ctrl decreases camera speed.

## Quick Screenshot Validation

Use semantic views for visual checks. Recommended known-good view:

- `mip360_bicycle_hf_msplat`, dataset camera index `141`

Minimum check:

```powershell
xmake build example-gaussian-splatting
```

Then run one short desktop capture on `garden` or `bicycle` and inspect:

- foreground object is recognizable;
- center foveal region is sharp;
- no black frame;
- no white fog;
- no obvious peripheral holes.

This sync was verified with:

```powershell
xmake build -r -j1 example-gaussian-splatting

xmake run -w D:\formal_3dgs_scene_assets\smoke_outputs\mip360_bicycle_hf_msplat example-gaussian-splatting `
  --benchmark `
  --benchmark-frames 8 `
  --benchmark-warmup 2 `
  --benchmark-settle-frames 0 `
  --benchmark-camera-file C:\Users\70982\Documents\gs\build\core_algorithm_validation\camera_overrides\mip360_bicycle_hf_msplat_camera_141.csv `
  --benchmark-view-frames 1 `
  --fixed-gaze 0.5,0.5 `
  --scene resources/scenes/fixed_gaze_smoke_scene.vmanifest `
  --capture-frame-sequence `
  --capture-frame-dir C:\Users\70982\Documents\libvultra_dev_next_sync\build\smoke_tilecap_bicycle_visual\frames `
  --capture-frame-prefix tilecap `
  --capture-frame-limit 1 `
  --benchmark-output C:\Users\70982\Documents\libvultra_dev_next_sync\build\smoke_tilecap_bicycle_visual\benchmark.csv `
  --shader-antipop-mode delayed-guide-temporal-logpolar-tile-cap `
  --coverage-texture-update-interval 4 `
  --coverage-texture-debug false
```

Verified screenshot:

```text
C:\Users\70982\Documents\libvultra_dev_next_sync\build\smoke_tilecap_bicycle_visual\frames\tilecap_frame_000003.png
```

## Data

Datasets and generated resource views stay outside the repository. The current
machine used:

```text
D:\formal_3dgs_scene_assets
```

On a new machine, rebuild resource views with `vasset-cli import` after placing
the scene PLY and camera files under a local asset root.

## Current Validation State

The last local sync build passed:

```powershell
xmake build example-gaussian-splatting
```

The next machine should rerun a short interactive smoke on `garden` or
`bicycle:141` before starting longer experiments.
