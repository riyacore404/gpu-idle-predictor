# gpu-idle-predictor

A predictor for idle and oversized GPU allocation on Kubernetes-hosted LLM inference workloads — the layer OpenCost's own maintainers named as a still-open gap after shipping per-token inference cost tracking in [OpenCost v1.121.0](https://github.com/opencost/opencost) (Aug 2026).

## The problem

OpenCost can now tell you what an LLM inference token *already cost*. It can't yet tell you when a GPU allocation is *about to become* idle or oversized — before the next billing cycle confirms it. CPU/memory autoscaling has a mature answer to this (Vertical Pod Autoscaler), but GPU idle risk for LLM serving doesn't follow the same shape: it's driven by batching gaps, KV-cache churn, and bursty request patterns that don't look like a steady CPU load curve.

This project is a first pass at that predictive layer: given real traffic signals (token throughput, request burstiness, inter-request gaps), predict whether a GPU allocation is trending toward idle *before* it happens.

## Current status

Local dev, baseline model converged, and a live Prometheus exporter has been built and verified end-to-end. The full pipeline runs end to end locally:

**Inference → Metrics/proxy → Feature extraction → Model → Prometheus (`gpu_idle_risk_score`)**

The baseline (gradient-boosted trees on rolling-window traffic features) went through several rounds of debugging before producing a trustworthy result — documented in full in [Evaluation results](#evaluation-results) below. Two independent ~2–2.5 hour data collection runs converge on **ROC-AUC ≈ 0.68–0.70** with tight cross-validation variance (std 0.025–0.043), treated as the model's stable ceiling on this feature set. That model is now wrapped in a live exporter (`idle_risk_exporter.py`) that serves real-time idle-risk predictions as a Prometheus gauge, verified end-to-end against live traffic — see Evaluation results for the observed score transition.

## Architecture (local dev)

```
                    ┌─────────────────────┐
traffic ──────────► │  idle-risk-exporter  │ ──proxy──► llama.cpp (CPU inference)
                    │  (logs each request,│
                    │   computes rolling  │
                    │   features, scores  │
                    │   the model)        │
                    └──────────┬──────────┘
                               │ /metrics
                               ▼
                          Prometheus ◄────── OpenCost (cost allocation)
```

Local dev runs entirely on a `kind` Kubernetes cluster, no cloud dependency.

### Deliberate substitutions for local development

This project targets **vLLM + llm-d** in production (the same stack OpenCost's real integration targets). Locally, on CPU-only hardware, substitutions were made — documented here rather than glossed over:

- **[llama.cpp server](https://github.com/ggml-org/llama.cpp)** instead of vLLM for the inference layer. vLLM's standard image is CUDA-only and fails outright on CPU (`Failed to infer device type`); llama.cpp is CPU-first and ships equivalent Prometheus token metrics (`llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`) that map cleanly to vLLM's metric shape.
- **[NVIDIA Fake GPU Operator](https://github.com/run-ai/fake-gpu-operator)** to simulate a GPU resource on a CPU-only `kind` node, so Kubernetes will schedule GPU-requesting pods and OpenCost has a real `gpuCount` to price against (using manually-set GPU pricing, since there's no real GPU billing to observe locally).
- **Exporter proxies requests instead of querying Prometheus directly.** llama.cpp's `/metrics` only exposes cumulative token/time counters, not a per-request count or per-request latency — unlike vLLM's real metrics, which include per-request latency histograms (`vllm:e2e_request_latency_seconds`). A production exporter targeting vLLM would query Prometheus directly with no proxy needed; the proxy pattern here exists specifically to reconstruct request-level features (`request_count`, `mean_latency_seconds`) that llama.cpp's counters can't provide, so the live features match what the model was trained on exactly.

## Setup

Requires: Docker, `kind`, `kubectl`, `helm`, Python 3.9+.

```bash
# 1. Model file — any TinyLlama-1.1B-Chat GGUF quantization works; tested with
#    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (Q4_0).
mkdir -p models
curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -o models/tinyllama.gguf

# 2. Create cluster
kind create cluster --name gpu-idle-predictor

# 3. Copy the model into the node directly (NOT a host bind-mount/extraMounts —
#    on macOS + Docker Desktop, bind-mounted files can fail to open inside the
#    container with "Operation not permitted", caused by the com.apple.provenance
#    extended attribute macOS attaches to downloaded files not surviving the
#    virtiofs boundary cleanly. Copying the file directly into the node's own
#    filesystem avoids this entirely.)
docker exec gpu-idle-predictor-control-plane mkdir -p /models
docker cp models/tinyllama.gguf gpu-idle-predictor-control-plane:/models/tinyllama.gguf

# 4. Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# 5. Inference server (see vllm-deployment.yaml)
kubectl create namespace inference
kubectl apply -f vllm-deployment.yaml
kubectl apply -f llama-servicemonitor.yaml
# Note: this must come after step 4 — the ServiceMonitor CRD only exists once
# the Prometheus Operator is installed.

# 6. Fake GPU operator (for GPU cost simulation)
kubectl create namespace gpu-operator
kubectl label ns gpu-operator pod-security.kubernetes.io/enforce=privileged
helm upgrade -i fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator \
  --namespace gpu-operator --set topology.nodePools.default.gpuCount=1
kubectl label node gpu-idle-predictor-control-plane run.ai/simulated-gpu-node-pool=default

# 7. OpenCost (see opencost-values.yaml — points at the Prometheus service above)
helm repo add opencost https://opencost.github.io/opencost-helm-chart
helm upgrade -i opencost opencost/opencost --namespace opencost --create-namespace \
  -f opencost-values.yaml
```

If the model file needs to be re-copied after recreating the cluster (e.g. `kind delete cluster` + `kind create cluster`), just repeat step 3 — the node's `/models` directory does not persist across cluster recreation.

Port-forwards for local access: see `start-ports.sh`.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests pandas numpy scikit-learn joblib
```

Traffic generation runs **in-cluster**, not through a local port-forward — a local tunnel was found to drop and fragment multi-hour collection sessions. Use `run_collection.sh`, which applies the traffic-generator pod with a bounded duration, streams logs safely, and only deletes the pod once a complete run is confirmed:

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

### Live idle-risk exporter

Once `idle_risk_model.joblib` exists (from `train_model.py`), deploy the exporter:

```bash
kubectl create configmap idle-risk-exporter-script \
  -n inference \
  --from-file=idle_risk_exporter.py \
  --from-file=idle_risk_model.joblib \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f idle-risk-exporter.yaml
```

`generate_traffic_incluster.py`'s `SERVER_URL` points at the exporter's proxy (`idle-risk-exporter.inference.svc.cluster.local:8090`) rather than llama-cpu directly, so traffic runs through it and gets scored live. Prometheus scrapes `gpu_idle_risk_score` via the bundled `ServiceMonitor`:

```bash
kubectl port-forward -n inference svc/idle-risk-exporter 8090:8090 &
curl -s http://localhost:8090/metrics | grep gpu_idle
```

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

**Live exporter verification:** deployed as a reverse-proxy Prometheus exporter (`idle_risk_exporter.py`) computing the same rolling-window features as training in real time. Verified twice, independently, on two separately-built clusters:
- Run 1: `gpu_idle_risk_score` measured **0.003 during active (`moderate` regime) traffic**, then rose to **0.999 within one 30s window of a transition into a `quiet` regime**.
- Run 2 (full clean-cluster rebuild — see Setup — including a re-download of the model file and a from-scratch cluster): `gpu_idle_risk_score` measured **0.001–0.003 during active traffic**, then rose to **0.98–0.99** during a subsequent quiet stretch.

Both runs show the same transition pattern, confirming the live feature pipeline reproduces training-time behavior on infrastructure built from scratch, not just on a long-lived dev cluster carrying manual setup history.

## Known limitations

- Local dev traffic is synthetic (regime-based, not real production LLM traffic) — real validation requires a real vLLM/llm-d deployment with genuine user traffic.
- GPU cost is simulated via Fake GPU Operator + manually-set pricing; no real GPU billing was observed.
- The model is substantially driven by current-window activity (`total_tokens`) rather than subtler rolling-trend signals — see Evaluation results.
- All evaluation to date is on synthetic single-session data at the 30–150 minute scale; the model has not been validated on multi-day or multi-tenant traffic patterns.
- The exporter's reverse-proxy design (rather than direct Prometheus querying) is specific to llama.cpp's metrics gap for local dev — a production exporter against vLLM would query Prometheus directly using vLLM's native per-request latency histograms, no proxy required.

## License

Apache-2.0 (to match OpenCost's ecosystem licensing norms).