"""
Phase 5 — Load test the /classify endpoint and re-run the benchmark harness against
the served endpoint (not the raw model call), for direct comparison against Phase 2/3's
raw single-process numbers.
"""
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, ".")
from bench_common import WORKLOAD, percentile

SERVER_URL = "http://localhost:8080/classify"
N_WARMUP = 2
N_SEQUENTIAL = 12  # single-request-at-a-time baseline against the served endpoint
CONCURRENCY_LEVELS = (1, 4, 8, 16)
N_REQUESTS_PER_CONCURRENCY_LEVEL = 16


def call_classify(clause_text):
    t0 = time.perf_counter()
    resp = requests.post(SERVER_URL, json={"clause": clause_text}, timeout=60.0)
    wall_latency_s = time.perf_counter() - t0
    ok = resp.status_code == 200
    body = resp.json() if ok else {"error": resp.text}
    return {"ok": ok, "wall_latency_s": wall_latency_s, "body": body}


def sequential_benchmark():
    print(f"=== Sequential (single-request) benchmark: {N_WARMUP} warmup + {N_SEQUENTIAL} timed ===")
    for i in range(N_WARMUP):
        call_classify(WORKLOAD[0]["text"])
        print(f"  warmup {i + 1}/{N_WARMUP} done")

    latencies = []
    n_ok = 0
    for i in range(N_SEQUENTIAL):
        clause = WORKLOAD[i % len(WORKLOAD)]
        result = call_classify(clause["text"])
        latencies.append(result["wall_latency_s"])
        n_ok += int(result["ok"])
        print(f"  call {i + 1}/{N_SEQUENTIAL} [{clause['id']}] {result['wall_latency_s']:.3f}s ok={result['ok']}")

    return {
        "n_requests": N_SEQUENTIAL,
        "n_ok": n_ok,
        "latency_s": {
            "median": round(statistics.median(latencies), 4),
            "p90": round(percentile(latencies, 90), 4),
            "mean": round(statistics.mean(latencies), 4),
            "stdev": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        },
    }


def concurrency_benchmark(n_concurrent):
    print(f"\n=== Concurrency level {n_concurrent}: {N_REQUESTS_PER_CONCURRENCY_LEVEL} requests ===")
    clauses = [WORKLOAD[i % len(WORKLOAD)]["text"] for i in range(N_REQUESTS_PER_CONCURRENCY_LEVEL)]

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(call_classify, c) for c in clauses]
        for f in as_completed(futures):
            results.append(f.result())
    wall_time_s = time.perf_counter() - t0

    n_ok = sum(1 for r in results if r["ok"])
    per_request_latencies = [r["wall_latency_s"] for r in results]
    throughput = len(results) / wall_time_s

    print(f"  wall_time={wall_time_s:.2f}s, throughput={throughput:.3f} req/s, ok={n_ok}/{len(results)}")

    return {
        "concurrency": n_concurrent,
        "n_requests": len(results),
        "n_ok": n_ok,
        "wall_time_s": round(wall_time_s, 4),
        "throughput_req_per_s": round(throughput, 4),
        "per_request_latency_s": {
            "median": round(statistics.median(per_request_latencies), 4),
            "mean": round(statistics.mean(per_request_latencies), 4),
            "max": round(max(per_request_latencies), 4),
        },
    }


def main():
    # sanity check server is up
    health = requests.get("http://localhost:8080/health", timeout=5).json()
    print("Health check:", health)

    results = {"sequential": sequential_benchmark(), "concurrency_sweep": []}
    for n in CONCURRENCY_LEVELS:
        results["concurrency_sweep"].append(concurrency_benchmark(n))

    with open("results/phase5_load_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))
    print("\nSaved results/phase5_load_test.json")


if __name__ == "__main__":
    main()
