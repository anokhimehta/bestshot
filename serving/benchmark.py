import requests
import json
import time

URL = "http://127.0.0.1:8000/predict"

# Load input request JSON
with open("testing_example_input.json") as f:
    data = json.load(f)

from config import CONFIG

URL = CONFIG["url"]
N_USERS = CONFIG["num_users"]
UPLOADS_PER_USER = CONFIG["uploads_per_user"]
N_REQUESTS = N_USERS * UPLOADS_PER_USER

latencies = []
all_results = []

for i in range(N_REQUESTS):
    start = time.time()
    response = requests.post(URL, json=data)
    end = time.time()

    if response.status_code != 200:
        print(f"Request {i+1} failed with status {response.status_code}")
        continue

    latencies.append(end - start)
    resp_json = response.json()
    all_results.append(resp_json["results"])

# flatten results for metrics
flattened_results = [img for batch in all_results for img in batch]

# compute average latency and throughput
if len(latencies) == 0:
    print("All requests failed. Cannot compute metrics.") # check to avoid division by zero error
else:
    avg_latency = sum(latencies) / len(latencies)
    throughput = len(latencies) / sum(latencies)

# compute average scores
metric_sums = {"koniq_score": 0, "sharpness": 0, "exposure": 0, "face_quality": 0}
for img in flattened_results:
    for metric in metric_sums:
        metric_sums[metric] += img["scores"].get(metric, 0)

avg_metrics = {k: v / len(flattened_results) for k, v in metric_sums.items()}

# save benchmark results
with open('benchmark_results.txt', 'w') as f:
    f.write(f"------ Benchmark results for config: {CONFIG['mode']} ------\n")
    f.write(f"Sent {N_REQUESTS} requests\n")
    f.write(f"Average latency per request: {avg_latency*1000:.2f} ms\n")
    f.write(f"Throughput: {throughput:.2f} req/sec\n")
    f.write("Average metrics per image:\n")
    for metric, value in avg_metrics.items():
        f.write(f"  {metric}: {value:.3f}\n")

# print results to console too
print("Benchmark finished!\n")
print(f"Ran config: {CONFIG}\n")
print(f"Average latency: {avg_latency*1000:.2f} ms")
print(f"Throughput: {throughput:.2f} req/sec")
print("Average metrics per image:", avg_metrics, "\n")