import argparse
import json
import urllib.request

URL = "http://localhost:11434"


def ollama(path, body=None):
    data = json.dumps(body).encode() if body else None
    with urllib.request.urlopen(urllib.request.Request(URL + path, data=data)) as r:
        return json.load(r)


def model_specs(name):
    info = ollama("/api/show", {"model": name})["model_info"]

    def get(suffix):
        return next(v for k, v in info.items() if k.endswith(suffix))

    size = next(m["size"] for m in ollama("/api/tags")["models"] if m["name"] == name)
    return (size / (1024 * 1024), get(".block_count"),
            get(".attention.head_count_kv"), get(".attention.key_length"))


def predict_vram(ctx, weights, layers, kv_heads, head_dim, overhead):
    kv_per_token = 2 * layers * kv_heads * head_dim * 2
    return weights + kv_per_token * ctx / (1024 * 1024) + overhead


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--overhead", type=float, default=70)
    a = p.parse_args()
    weights, layers, kv_heads, head_dim = model_specs(a.model)
    print(round(predict_vram(a.context_length, weights, layers, kv_heads, head_dim, a.overhead)))
