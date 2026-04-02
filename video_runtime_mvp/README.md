# video-runtime-mvp

A minimal training-oriented video data runtime.

This MVP focuses on the architecture needed for large-scale video training:

- manifest-driven datasets
- explicit clip requests (`exact` vs `approximate`)
- pluggable decoder backends
- post-processing pipeline (`undistort`, `resize`, `normalize`, `layout pack`)
- PyTorch-friendly iterable dataset and collate path

## What is implemented

### CPU path
- `OpenCVCPUDecoder`: reads indexed frames using OpenCV.
- `VideoIterableDataset`: streams clip samples for training.
- transforms:
  - `UndistortTransform`
  - `ResizeTransform`
  - `NormalizeTransform`
  - `ToTensorTransform`

### GPU path scaffolding
- `PyNvDecoder`: adapter stub for a future `PyNvVideoCodec` integration.
- runtime/backend registry.
- zero-copy/Torch bridge extension point.

## Install

```bash
pip install -e ./video_runtime_mvp
```

## Quick start

```python
from video_runtime_mvp import (
    BackendConfig,
    ClipRequest,
    Compose,
    ManifestDataset,
    NormalizeTransform,
    ResizeTransform,
    ToTensorTransform,
    VideoIterableDataset,
    VideoRuntime,
)

manifest = ManifestDataset.from_jsonl("examples_train_manifest.jsonl")
runtime = VideoRuntime.from_config(BackendConfig(name="opencv_cpu"))
transforms = Compose([
    ResizeTransform((224, 224)),
    NormalizeTransform((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensorTransform(),
])

dataset = VideoIterableDataset(
    manifest=manifest,
    runtime=runtime,
    clip_request=ClipRequest(num_frames=16, stride=2, mode="exact"),
    transforms=transforms,
)

sample = next(iter(dataset))
print(sample["video"].shape)
```

## Suggested next steps

- replace `PyNvDecoder` stub with real `PyNvVideoCodec` decode sessions
- move hot transforms to CUDA/C++
- add worker-local session pools and prefetch queues
- add distributed sharding and telemetry
