# gpu-idle-predictor

A predictor for idle and oversized GPU allocation on Kubernetes-hosted LLM inference workloads — the layer OpenCost's own maintainers named as a still-open gap after shipping per-token inference cost tracking in [OpenCost v1.121.0](https://github.com/opencost/opencost) (Aug 2026).

## The problem

OpenCost can now tell you what an LLM inference token *already cost*. It can't yet tell you when a GPU allocation is *about to become* idle or oversized — before the next billing cycle confirms it. CPU/memory autoscaling has a mature answer to this (Vertical Pod Autoscaler), but GPU idle risk for LLM serving doesn't follow the same shape: it's driven by batching gaps, KV-cache churn, and bursty request patterns that don't look like a steady CPU load curve.

This project is a first pass at that predictive layer: given real traffic signals (token throughput, request burstiness, inter-request gaps), predict whether a GPU allocation is trending toward idle *before* it happens.

## Current status

Early-stage / local dev. The full pipeline runs end to end locally:

**Inference → Metrics → Cost allocation → Feature extraction → Baseline model**

The first baseline model (gradient-boosted trees on rolling-window traffic features) did **not** find meaningful signal on the first real dataset (ROC-AUC ≈ 0.48, effectively chance) — traced to the synthetic traffic generator producing independent random bursts/idle gaps with no real temporal structure for a model to learn from. Fixed by moving to a regime-based generator (traffic persists in "quiet" / "moderate" / "busy" states for several cycles, creating genuine autocorrelation). Re-evaluation with regime-based data is in progress.

## Architecture (local dev)

```
llama.cpp (CPU inference) ───┐
                             ├──► Prometheus ──► [feature extraction] ──► [model]
OpenCost (cost allocation) ──┘
```

Local dev runs entirely on a `kind` Kubernetes cluster, no cloud dependency.

### Deliberate substitutions for local development

This project targets **vLLM + llm-d** in production (the same stack OpenCost's real integration targets). Locally, on CPU-only hardware, two substitutions were made — documented here rather than glossed over:

- **[llama.cpp server](https://github.com/ggml-org/llama.cpp)** instead of vLLM for the inference layer. vLLM's standard image is CUDA-only and fails outright on CPU (`Failed to infer device type`); llama.cpp is CPU-first and ships equivalent Prometheus token metrics (`llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`) that map cleanly to vLLM's metric shape. A thin adapter layer maps these to the internal feature schema, so swapping the real metrics source back in for vLLM validation is a config change, not a rewrite.
- **[NVIDIA Fake GPU Operator](https://github.com/run-ai/fake-gpu-operator)** to simulate a GPU resource on a CPU-only `kind` node, so Kubernetes will schedule GPU-requesting pods and OpenCost has a real `gpuCount` to price against (using manually-set GPU pricing, since there's no real GPU billing to observe locally).

## Setup

Requires: Docker, `kind`, `kubectl`, `helm`, Python 3.9+.

```bash
# 1. Create cluster
kind create cluster --name gpu-idle-predictor

# 2. Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# 3. Inference server (see vllm-deployment.yaml)
kubectl create namespace inference
kubectl apply -f vllm-deployment.yaml
kubectl apply -f llama-servicemonitor.yaml

# 4. Fake GPU operator (for GPU cost simulation)
kubectl create namespace gpu-operator
kubectl label ns gpu-operator pod-security.kubernetes.io/enforce=privileged
helm upgrade -i fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator \
  --namespace gpu-operator --set topology.nodePools.default.gpuCount=1
kubectl label node gpu-idle-predictor-control-plane run.ai/simulated-gpu-node-pool=default

# 5. OpenCost (see opencost-values.yaml — points at the Prometheus service above)
helm repo add opencost https://opencost.github.io/opencost-helm-chart
helm upgrade -i opencost opencost/opencost --namespace opencost --create-namespace \
  -f opencost-values.yaml
```

Port-forwards for local access: see `start-ports.sh`.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests pandas numpy scikit-learn joblib

# Generate realistic traffic (regime-based bursts/idle periods)
python3 generate_traffic.py

# Extract rolling-window features from the logged traffic
python3 extract_features.py

# Train the baseline idle-risk classifier
python3 train_model.py
```

## Evaluation methodology

Ground truth for "idle" is the traffic log's own request timeline (real inter-request gaps), not a simulated label. The eventual production evaluation will compare predictions against OpenCost's realized per-period cost-per-token, one reporting cycle later — the same ground truth OpenCost itself uses.

## Known limitations

- Local dev traffic is synthetic (regime-based, not real production LLM traffic) — real validation requires a real vLLM/llm-d deployment with genuine user traffic.
- GPU cost is simulated via Fake GPU Operator + manually-set pricing; no real GPU billing was observed.
- First baseline model result was null (no signal) on the initial (non-regime) synthetic dataset — see "Current status" above.

## License

Apache-2.0 (to match OpenCost's ecosystem licensing norms).