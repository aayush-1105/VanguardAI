"""Load test for the gateway.

    locust -f locustfile.py --host http://localhost:8000

Every virtual user sends the same prompt, so after the first request most
traffic is a semantic cache hit. That measures the gateway's own path (vault,
embedding, rerank, telemetry); to load the providers instead, vary the content.
"""

from locust import HttpUser, between, task

PAYLOAD = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "user", "content": "Summarize the TCP three-way handshake."}
    ],
}


class GatewayUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat_completion(self):
        self.client.post("/v1/chat/completions", json=PAYLOAD, timeout=60)
