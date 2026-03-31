import requests
import json
import time

URL = "http://localhost:8000/predict"

# Load example request
with open("testing_example_input.json") as f:
    data = json.load(f)

N_REQUESTS = 100  # number of requests to simulate

start = time.time()

for _ in range(N_REQUESTS):
    response = requests.post(URL, json=data)
    assert response.status_code == 200

end = time.time()

#print(f"Sent {N_REQUESTS} requests")
#print(f"Average latency per request: {(end - start)/N_REQUESTS*1000:.2f} ms")

with open('benchmark_results.txt', 'w') as f:
    print("Benchmark Results with dummy model\n", file=f)
    print(f"Sent {N_REQUESTS} requests", file=f)
    print(f"Average latency per request: {(end - start)/N_REQUESTS*1000:.2f} ms", file=f)
