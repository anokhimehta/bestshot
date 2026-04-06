import requests
import json
import time
import threading

# load input request json
with open("testing_example_input.json") as f:
    data = json.load(f)

# load config
from config import CONFIG

URL = CONFIG["url"]
N_USERS = CONFIG["num_users"]
UPLOADS_PER_USER = CONFIG["uploads_per_user"]
MAX_WORKERS = CONFIG["max_workers"]
N_REQUESTS = N_USERS * UPLOADS_PER_USER

# request functions

def run_sequential(n_requests):
    """one request at a time"""
    latencies = []
    all_results = []

    for i in range(n_requests):
        start = time.time()
        response = requests.post(URL, json=data)
        end = time.time()

        if response.status_code != 200:
            print(f"Request {i+1} failed with status {response.status_code}")
            continue

        latencies.append(end - start)
        all_results.append(response.json()["results"])

    return latencies, all_results


def run_concurrent(num_users, uploads_per_user):
    """multiple users sending requests simultaneously"""
    all_latencies = []
    all_results = []
    lock = threading.Lock()

    def worker():
        for _ in range(uploads_per_user):
            start = time.time()
            response = requests.post(URL, json=data)
            end = time.time()
            with lock:
                if response.status_code == 200:
                    all_latencies.append(end - start)
                    all_results.append(response.json()["results"])

    threads = [threading.Thread(target=worker) for _ in range(num_users)]
    for t in threads: t.start()
    for t in threads: t.join()

    return all_latencies, all_results

# run the right mode based on config

if MAX_WORKERS == 1:
    print(f"Running sequential benchmark ({N_REQUESTS} total requests)...")
    latencies, all_results = run_sequential(N_REQUESTS)
else:
    print(f"Running concurrent benchmark ({N_USERS} users x {UPLOADS_PER_USER} requests)...")
    latencies, all_results = run_concurrent(N_USERS, UPLOADS_PER_USER)

# compute metrics 

flattened_results = [img for batch in all_results for img in batch]

if len(latencies) == 0:
    print("All requests failed. Cannot compute metrics.")
else:
    avg_latency = sum(latencies) / len(latencies)
    throughput = len(latencies) / sum(latencies)

    import numpy as np
    p50 = np.percentile(latencies, 50) * 1000
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000

    metric_sums = {"koniq_score": 0, "sharpness": 0, "exposure": 0, "face_quality": 0}
    for img in flattened_results:
        for metric in metric_sums:
            metric_sums[metric] += img["scores"].get(metric, 0)

    avg_metrics = {k: v / len(flattened_results) for k, v in metric_sums.items()}

    # save results to a text file for later analysis

    with open("benchmark_results.txt", "w") as f:
        f.write(f"------ Benchmark results for config: {CONFIG['mode']} ------\n")
        f.write(f"Users: {N_USERS}, Requests per user: {UPLOADS_PER_USER}, Workers: {MAX_WORKERS}\n")
        f.write(f"Total successful requests: {len(latencies)}/{N_REQUESTS}\n")
        f.write(f"Average latency: {avg_latency*1000:.2f} ms\n")
        f.write(f"Latency p50: {p50:.2f} ms\n")
        f.write(f"Latency p95: {p95:.2f} ms\n")
        f.write(f"Latency p99: {p99:.2f} ms\n")
        f.write(f"Throughput: {throughput:.2f} req/sec\n")
        f.write("Average metrics per image:\n")
        for metric, value in avg_metrics.items():
            f.write(f"  {metric}: {value:.3f}\n")

    # print results to terminal output

    print("\nBenchmark finished!")
    print(f"Config: {CONFIG['mode']} | Workers: {MAX_WORKERS}")
    print(f"Total requests: {len(latencies)}/{N_REQUESTS} succeeded")
    print(f"Average latency : {avg_latency*1000:.2f} ms")
    print(f"Latency p50     : {p50:.2f} ms")
    print(f"Latency p95     : {p95:.2f} ms")
    print(f"Latency p99     : {p99:.2f} ms")
    print(f"Throughput      : {throughput:.2f} req/sec")
    print(f"Avg metrics     : {avg_metrics}")