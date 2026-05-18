from __future__ import annotations

import requests


class OllamaClient:
    def __init__(
        self,
        model_name: str = "mistral:latest",
        base_url: str = "http://localhost:11434/api",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()

        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise ValueError(f"Unexpected Ollama response shape: {data}") from exc