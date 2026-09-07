#!/bin/bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &
kubectl port-forward -n inference svc/llama-cpu 8080:8080 &
kubectl port-forward -n opencost svc/opencost 9003:9003 &
kubectl port-forward -n opencost svc/opencost 9091:9090 &
echo "All port-forwards started in background. Use 'jobs' to see them, 'kill %1 %2 %3 %4' to stop all."
