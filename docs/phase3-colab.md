# Phase 3: authorized Colab T4 execution

Status: **preparation only; GPU correctness and performance are not yet verified**.
Prefer [the single-cell launcher](phase3-colab-one-cell.md) for numbered live
progress and automatic failure-log export. The seven manual cells below remain
available for inspecting and running each stage separately.
Run these Python cells in order in the existing GPU notebook. Do not purchase
compute, change the public hosting, mount Drive, enter credentials or install
packages into the notebook's base interpreter. Stop at the first failing cell
and export its logs with the last cell; do not change precision, prompts or
dependencies to make a failure disappear.

## Environment decision

The user reports a Tesla T4, 15,360 MiB VRAM, capability 7.5, driver 580.82.07,
driver-advertised CUDA 13.0 and base PyTorch 2.11.0+cu130. These are **reported**
facts until the cells collect them. `nvidia-smi`'s CUDA version is the driver's
supported version, not necessarily the installed toolkit or PyTorch runtime.
The workflow records all three separately, including `nvcc --version`.

Use FP16, not native BF16 on this Turing GPU. The reviewed baseline is
vLLM **0.17.1**: its published requirements pin Torch 2.10.0, FlashInfer 0.6.4
and Transformers `<5`. This repository uses Transformers 5.17.0. Therefore:

| Environment | Torch wheel | Transformers | Purpose |
| --- | --- | --- | --- |
| Existing notebook kernel | unchanged | unchanged | control cells only |
| New runtime venv | 2.10.0+cu129 | 5.17.0 | project tests, Qwen and client |
| New vLLM venv | 2.10.0+cu129 | 4.57.6 | unmodified vLLM 0.17.1 server |

This is a deliberate older stable baseline with explicitly documented T4
FlashInfer support, not a claim about the latest vLLM release. Both engines
use the same CUDA/Torch wheels, FP16, checkpoint commit and prompt token IDs.
Driver backward compatibility is expected, but **not proof of a working
installation**. Resolution, `pip check`, real CUDA matmul, model correctness,
and the two-server smoke run are separate gates. FlashInfer may compile
kernels at startup: if toolkit/JIT compatibility fails, return the logs.
Do not replace system CUDA, build vLLM from source or patch its implementation.

Child processes preserve the inherited `LD_LIBRARY_PATH`, including Colab's
NVIDIA driver-library paths. The original preparation incorrectly removed it:
the user's T4 diagnostic showed normal CUDA worked while that removal hid the
driver. Do not blanket-unset it to address a hypothetical library mismatch;
diagnose any actual mismatch before making targeted changes. No
`--system-site-packages`, `--no-deps` or dependency-conflict bypass is used.
Core versions are pinned; dry-run reports produce hashed constraints files
locking the selected transitive package versions before installation. Complete
installation reports and freezes are also recorded. Fresh resolution can still
fail as packaging evolves.

Colab's Python 3.13 failed while running `ensurepip` inside the first new venv.
The bootstrap now creates pip-less venvs and uses the base pip's supported
`--python <new-venv>` option to install pip 26.2.2 **only in each new venv**.
This requires host pip >=22.3, not a system package installation or base pip
upgrade. Full bootstrap output is saved in `setup/bootstrap-*.txt`.

Target the authorized 16 GB-class T4, one GPU, Linux x86-64, Python 3.11–3.13
and about 20 GiB of free installation disk as a planning allowance. This is
**not** an experimentally established minimum VRAM/disk requirement. Both
models are never resident together. vLLM reserves up to 70% of GPU memory;
shared/notebook GPU allocations can still cause OOM. Concurrency 8 is optional
only after correctness at 8; the required run uses 1/2/4.

Sources inspected:

- [pip: manage a pip-less environment with --python](https://pip.pypa.io/en/stable/topics/python-option/)
- [vLLM 0.17.1 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.17.1/requirements/cuda.txt)
- [vLLM common requirements](https://github.com/vllm-project/vllm/blob/v0.17.1/requirements/common.txt)
- [vLLM GPU installation and CUDA-library guidance](https://docs.vllm.ai/en/v0.17.1/getting_started/installation/gpu/)
- [T4/SM75 FlashInfer support and head sizes](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/v1/attention/backends/flashinfer.py)
- [Turing dtype policy](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/platforms/cuda.py)
- [Streaming token-ID protocol](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/entrypoints/openai/completion/protocol.py)
- [Qwen2.5-0.5B configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)

## Cell 1 — clone and inventory (no installation)

Use the preparation commit in the accompanying handoff to pin `PREP_COMMIT`.
Leaving it `None` takes one recorded snapshot of main; never pull or edit the
checkout between correctness and measurement. These cells write only new
environments and ignored evidence, not tracked source.

```python
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys

PREP_COMMIT = None  # Optional: paste the full preparation commit from the handoff.
ROOT = Path('/content/llm-inference-runtime-phase3')
if ROOT.exists():
    raise RuntimeError('Clone target already exists. Reuse a verified clean checkout deliberately; do not overwrite it.')
subprocess.run(['git', 'clone', 'https://github.com/amxson/llm-inference-runtime.git', str(ROOT)], check=True)
if PREP_COMMIT:
    subprocess.run(['git', 'checkout', '--detach', PREP_COMMIT], cwd=ROOT, check=True)
os.chdir(ROOT)
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN = ROOT / 'benchmark_results' / 'phase3-colab' / RUN_ID
RUN.mkdir(parents=True, exist_ok=False)
ENVS = Path('/content/phase3-envs') / RUN_ID
RPY, VPY = ENVS / 'runtime/bin/python', ENVS / 'vllm/bin/python'
ENV = os.environ.copy()
ENV.update(TOKENIZERS_PARALLELISM='false', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')

def run_logged(command, name):
    with (RUN / name).open('x', encoding='utf8') as log:
        p = subprocess.run([str(arg) for arg in command], cwd=ROOT, env=ENV,
                           stdout=log, stderr=subprocess.STDOUT)
    print((RUN / name).read_text(encoding='utf8')[-12000:])
    if p.returncode:
        raise RuntimeError(f'Exit {p.returncode}; STOP and export evidence: {RUN / name}')

run_logged(['git', 'status', '--short'], 'initial-status.txt')
run_logged(['git', 'log', '-12', '--oneline'], 'git-log.txt')
run_logged([sys.executable, 'scripts/phase3/preflight.py', '--report-only',
            '--output', RUN / 'base-environment.json'], 'base-environment.txt')
print('Evidence directory:', RUN)
```

Check that inventory actually reports a T4/CUDA device. Do not load another
model in the notebook kernel while these child processes run.

## Cell 2 — resolve before installing

Creates two empty venvs and performs binary-package dry runs. It does not
install either framework stack. Review the displayed versions and report any
resolver errors; the base Colab interpreter must not appear in install commands.

```python
run_logged([sys.executable, 'scripts/phase3/setup_colab.py', '--env-root', ENVS,
            '--output', RUN / 'setup'], 'setup-plan.txt')
for label in ('runtime', 'vllm'):
    report = json.loads((RUN / 'setup' / f'resolve-{label}.json').read_text())
    selected = {item['metadata']['name']: item['metadata']['version']
                for item in report['install']
                if item['metadata']['name'].lower() in
                {'torch', 'transformers', 'vllm', 'flashinfer-python', 'torchaudio', 'torchvision'}}
    print(label, selected)
```

## Cell 3 — install the resolved isolated stacks and verify CUDA

This can download several GB. No kernel restart is needed because the notebook
kernel's packages are not replaced. `pip check`, freezes and actual FP16 CUDA
matmul results are saved. Neither installation nor dry-run success proves that
the subsequent vLLM server/JIT will work.

```python
run_logged([sys.executable, 'scripts/phase3/setup_colab.py', '--env-root', ENVS,
            '--output', RUN / 'setup', '--install'], 'setup-install.txt')
run_logged([VPY, '-m', 'vllm.entrypoints.openai.api_server', '--help'], 'vllm-server-help.txt')
run_logged([RPY, '-m', 'pytest', '-q'], 'python-tests.txt')
run_logged([RPY, '-m', 'compileall', '-q', 'src', 'scripts', 'tests'], 'compileall.txt')
run_logged(['git', 'diff', '--check'], 'git-diff-check.txt')
```

## Cell 4 — freeze the checkpoint and run real GPU correctness

The same immutable HF commit is used by both engines. This downloads the
ungated base model; no Google/HF credential is needed. Source and package
versions are recorded. The gate covers finite full logits, uncached manual
greedy, request-local cache, equal/ragged batches, changing cohorts, zero
budget, test-only forced EOS, ordinary HTTP and genuine TCP SSE. The main
24-request workload uses the full 32/64/128 budgets for cached parity at 1/2/4.
An FP16 batch-shape rounding divergence is a failure to investigate, not a
license to relax internal exact parity or switch to FP32 silently.

```python
resolved = subprocess.check_output([
    str(RPY), '-c',
    "from huggingface_hub import HfApi; import json; "
    "print(json.dumps({'model':'Qwen/Qwen2.5-0.5B', 'revision':HfApi().model_info('Qwen/Qwen2.5-0.5B').sha}))"
], cwd=ROOT, env=ENV, text=True)
MODEL = json.loads(resolved)
with (RUN / 'model.json').open('x') as f:
    json.dump(MODEL, f, indent=2)
print(MODEL)
run_logged([RPY, 'scripts/phase3/correctness.py', '--model', MODEL['model'],
            '--revision', MODEL['revision'], '--concurrencies', '1', '2', '4',
            '--output', RUN / 'correctness.json'], 'correctness.txt')
```

If this fails, do **not** run the benchmark. Export `correctness.json` and its
logs; partial outputs and mismatches are preserved. Do not hand-edit the gate.

## Cell 5 — short two-engine smoke (not publishable measurements)

Starts each real server twice in ABBA order, sequentially, verifies both SSE
protocols, fixed input IDs, stop/count handling and cleanup. Four requests,
four output tokens, two measured repetitions: explicitly **not final evidence**.
Startup/JIT can still take time. The servers bind only loopback, not public
Colab proxies or tunnels.

```python
run_logged([RPY, 'scripts/phase3/benchmark.py', '--correctness', RUN / 'correctness.json',
            '--runtime-python', RPY, '--vllm-python', VPY,
            '--smoke', '--repetitions', '2', '--concurrencies', '1',
            '--output', RUN / 'smoke'], 'smoke.txt')
```

## Cell 6 — final benchmark and validation

Pause other GPU work. Both servers use the same checkpoint, FP16 and Torch/CUDA
set. Default concurrency is 1/2/4. Each engine/configuration gets two warmup
cohorts in each of its two residency sessions and ten measured repetitions in
total (five per session). Config order rotates. Warmups/model loading/startup
are not counted in request latency. Prefix caching is disabled in vLLM so
repeated warmup prompts do not get an unfair cross-request prefix-cache hit.
vLLM keeps its supported native optimizations; CUDA graphs are not disabled.

TTFT starts immediately before HTTP send, after client-concurrency admission,
and ends at the first event containing token IDs. Request latency ends at
terminal SSE completion. Cohort wall time includes all 24 requests and their
client-side queueing. vLLM receives pretokenized prompt IDs; our endpoint
tokenizes text. This preprocessing asymmetry is retained and disclosed.
Chunked SSE limits per-token arrival resolution: raw chunk timestamps and IDs
are stored, not invented token timestamps. Cross-engine divergence is reported
without rejecting it; our internal parity is enforced on every runtime run.

```python
run_logged([RPY, 'scripts/phase3/benchmark.py', '--correctness', RUN / 'correctness.json',
            '--runtime-python', RPY, '--vllm-python', VPY,
            '--repetitions', '10', '--concurrencies', '1', '2', '4',
            '--output', RUN / 'comparison'], 'benchmark.txt')
run_logged([RPY, 'scripts/phase3/validate.py', RUN / 'comparison/gpu_final_comparison.json',
            '--correctness', RUN / 'correctness.json'], 'validation.txt')
data = json.loads((RUN / 'comparison/gpu_final_comparison.json').read_text())
for row in data['aggregates']:
    print('Concurrency', row['concurrency'])
    for engine in ('runtime', 'vllm'):
        m = row[engine]
        print(engine, 'median request ms=', m['request_latency_ms']['median'],
              'p95 ms=', m['request_latency_ms']['p95'],
              'TTFT median ms=', m['ttft_ms']['median'],
              'tokens/sec=', m['generated_token_throughput']['median'])
    print('Cross-engine output parity:', row['comparison']['exact_output_parity_rate'])
```

If any step fails, `failed-comparison.json`, per-cohort raw JSON and server
logs remain available. A successful final filename is emitted only after
the runner's consistency validator passes. Artifacts are ignored locally;
**nothing is automatically committed, uploaded to GitHub or deployed**.

## Cell 7 — export, including after a failed run

Run this cell even after an earlier exception, as long as Cell 1 established
`RUN`. It exports only this run's evidence directory, never weights, HF caches,
venvs, credentials, unrelated files or the complete notebook. Review the ZIP
before attaching it in the chat. Evidence contains public-model IDs, software
versions, GPU identifiers, process IDs and generic `/content` paths.

```python
import hashlib, zipfile
files = sorted(p for p in RUN.rglob('*') if p.is_file())
manifest = {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with (RUN / 'SHA256SUMS.json').open('x') as f:
    json.dump(manifest, f, indent=2)
archive = RUN.parent / f'phase3-colab-{RUN_ID}.zip'
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RUN.rglob('*')):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(RUN)))
print('ZIP:', archive, 'SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())
from google.colab import files
files.download(str(archive))
```

Return the ZIP, even if a correctness/installation/streaming failure prevents
final measurements. Keep the original files unchanged. Real results must be
reviewed, independently consistency/provenance checked and then published in
`benchmarks/results/gpu_final_comparison.json` with UI/docs evidence. The final
public CPU demo/browser verification and project signoff are later steps of
**this same Phase 3**, not a new feature phase. Until then:

**PROJECT STATUS: INCOMPLETE**
