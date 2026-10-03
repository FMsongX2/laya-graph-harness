# Optional Decision 2.0 backend

Laya remains the default. Set `backend: "decision2"` in the installed project's
ignored `config/decision.json` to use a separately provisioned local Decision 2.0
package. The resident process, candidate/evidence access and iterative walk are
shared; this switches the choice model rather than creating another graph owner.

The public adapter loads the native package through `Decision2.from_pretrained`.
That loader checks its manifest, model identity and parameter count. It executes
the package's local inference code, so provision trusted pinned source. No weights
or downloaded model source are bundled in this repository.

The tested Sol package is
[`vllm-sr/Decision-2.0-Sol-2B`](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B),
revision `64235bef55dad29387dd16da7c90e038bf2f0972`. Download the complete snapshot
using your asset provisioning process and materialize regular local files; the
native loader rejects symlinks. Provisioning, downloading and trusting that code
are explicit steps; starting this worker does not download or train a model.

Create a separate model environment using `requirements-decision2.txt`. For
CUDA, install the matching PyTorch build from the official
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
The measured server used PyTorch 2.13.0+cu130; macOS used 2.13.0 with MPS.

`knowledgegraph/config/decision2.example.json` is the complete config shape.
Paths are relative to the installed project root or absolute local paths. The
example assumes a separately installed `.venv-decision2` and a local package
under `model-worker/models/Sol-2B`. Use `cpu`, `mps` or `cuda[:index]` explicitly.
MPS uses native FP32, disables CPU fallback and applies the configured per-process
memory fraction. CUDA uses native BF16 linear residency with the FP32 head.
ROCm-specific kernels, graph capture, quantization and shared-context overrides
are disabled. These paths are not equal numerical-precision configurations.

Stop the existing managed services and idle watcher before applying an overlay
or changing its backend config. Preserve the target's graph settings and assets.
Restart the single runtime, then call `kg prepare` to preload/prime the configured
worker. `/health` reports `model_backend` and the worker's model identity.

For source selection use `policy: "auto"` (exact citation links first) or
`policy: "model"` (force the configured choice worker). Legacy `policy: "laya"`
requests Laya explicitly and returns review without inference on another backend.
Source proposals from Decision 2.0 stay `needs_review: true`: the Laya source-task
temperature and threshold do not establish its calibration. Native question
errors such as `max_length_exceeded` yield a review/stop without advancing.

`kg walk` uses the configured local backend with the same request-local visited
hash set and evidence feedback. Choice key → actual graph target and proof →
updated goal/current-node/recent-evidence state → next choice. It still has a hop
cap and no learned STOP/BACK or semantic goal-completion certificate.

See [sanitized measurements](../verification/decision2-results.json) and
[measurement interpretation](measurements.md) for server, Mac and whole-agent
timings. The complete private corpus was not released, so an empty public clone
can test contracts but cannot reproduce those accuracy numbers from supplied data.
