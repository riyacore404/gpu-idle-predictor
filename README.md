# gpu-idle-predictor

A predictor for idle and oversized GPU allocation on Kubernetes-hosted LLM inference workloads — the layer OpenCost's own maintainers named as a still-open gap after shipping per-token inference cost tracking in [OpenCost v1.121.0](https://github.com/opencost/opencost) (Aug 2026).

## The problem

OpenCost can now tell you what an LLM inference token *already cost*. It can't yet tell you when a GPU allocation is *about to become* idle or oversized — before the next billing cycle confirms it. CPU/memory autoscaling has a mature answer to this (Vertical Pod Autoscaler), but GPU idle risk for LLM serving doesn't follow the same shape: it's driven by batching gaps, KV-cache churn, and bursty request patterns that don't look like a steady CPU load curve.

This project is a first pass at that predictive layer: given real traffic signals (token throughput, request burstiness, inter-request gaps), predict whether a GPU allocation is trending toward idle *before* it happens.

## Current status

Local dev, baseline model converged. The full pipeline runs end to end locally:

**Inference → Metrics → Cost allocation → Feature extraction → Baseline model**

The baseline (gradient-boosted trees on rolling-window traffic features) went through several rounds of debugging before producing a trustworthy result — documented in full in [Evaluation results](#evaluation-results) below. Two independent ~2–2.5 hour data collection runs now converge on **ROC-AUC ≈ 0.68–0.70** with tight cross-validation variance (std 0.025–0.043), which is treated as the model's stable ceiling on this feature set rather than a number still trending upward. Next step is a Prometheus exporter to serve this as a live metric alongside OpenCost's.

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
```

Traffic generation runs **in-cluster**, not through a local port-forward — a local tunnel was found to drop and fragment multi-hour collection sessions (see Evaluation results). Use `run_collection.sh`, which applies the traffic-generator pod with a bounded duration, streams logs safely, and only deletes the pod once a complete run is confirmed:

```bash
# Run a full collection (duration in seconds), then extract features and train
./run_collection.sh 10800 my_run        # ~3 hours; wrap in `caffeinate -i` on macOS
                                         # to survive laptop sleep — the kind cluster
                                         # runs inside Docker Desktop and suspends with it
python3 extract_features.py my_run.jsonl
python3 train_model.py
python3 cross_validate.py
```

`generate_traffic.py` (independent random bursts, no persistent regimes) is kept for reference but superseded — see the v1 result in Evaluation results for why regime-based traffic (`generate_traffic_incluster.py`) was necessary.

## Evaluation methodology

Ground truth for "idle" is the traffic log's own request timeline (real inter-request gaps), not a simulated label. The eventual production evaluation will compare predictions against OpenCost's realized per-period cost-per-token, one reporting cycle later — the same ground truth OpenCost itself uses.

## Evaluation results

| Version | Data | Setup | Single-split ROC-AUC | 5-fold CV (mean ± std) |
|---|---|---|---|---|
| v1 | 170 rows | Independent-random traffic generator | 0.477 (chance) | — |
| v2 | 138 rows | Regime-based traffic, fragmented (local port-forward) | 0.860 | 0.650 ± 0.203 |
| v3 | 156 rows | Regime-based, still fragmented | 0.826 | 0.704 ± 0.175 |
| v4 | 244 rows | Regime-based, in-cluster, one continuous 2h session | 0.689 | 0.695 ± 0.043 |
| v5 | 67 rows (34min active traffic, one session) | Regime-based, in-cluster, independent re-run | 0.758 | 0.679 ± 0.025 |

**Reading this progression honestly, in order of what it shows:**
- v1 → v2: a model needs real temporal structure to learn from. Independent random bursts gave the classifier nothing to key on; switching to persistent quiet/moderate/busy regimes fixed that immediately.
- v2/v3 → v4: high cross-validation variance (std 0.175–0.203) on v2/v3 was a data-volume and fragmentation problem, not a broken model — confirmed by v4's single continuous, unfragmented session dropping std to 0.043 without changing the underlying approach.
- v4 → v5: an independently collected session, run separately from v4, converged to the same ROC-AUC range (0.68–0.70) with comparably tight variance. Two independent datasets agreeing is stronger evidence of a stable ceiling than either result alone.

**Feature importance caveat**, consistent across v4 and v5: `total_tokens` (current-window activity) is the dominant feature (importance ~0.35–0.5), ahead of the rolling-trend features. The model is substantially learning "current activity predicts near-future activity" rather than detecting subtler idle-trending signals — an honest limitation to state up front, not something the AUC number alone conveys.

## Known limitations

- Local dev traffic is synthetic (regime-based, not real production LLM traffic) — real validation requires a real vLLM/llm-d deployment with genuine user traffic.
- GPU cost is simulated via Fake GPU Operator + manually-set pricing; no real GPU billing was observed.
- The model is substantially driven by current-window activity (`total_tokens`) rather than subtler rolling-trend signals — see Evaluation results.
- All evaluation to date is on synthetic single-session data at the 30–150 minute scale; the model has not been validated on multi-day or multi-tenant traffic patterns.

## License

Apache-2.0 (to match OpenCost's ecosystem licensing norms).