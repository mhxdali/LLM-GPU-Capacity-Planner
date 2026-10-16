import argparse
import subprocess

from calculator import model_specs, predict_vram


def read_gpu():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [line.split(",") for line in out.strip().splitlines()]
    return sum(int(r[0]) for r in rows), sum(int(r[1]) for r in rows)


p = argparse.ArgumentParser()
p.add_argument("--model", default="llama3.2:3b",
               help="Ollama model name, as shown by: ollama list")
p.add_argument("--context-length", type=int, default=4096,
               help="tokens kept in memory per user (prompt + answer)")
p.add_argument("--users", type=int, default=1,
               help="concurrent users (requests at the same moment)")
p.add_argument("--overhead", type=float, default=70,
               help="MiB, calibrated on one device")
p.add_argument("--gpu-memory", type=int, default=0,
               help="GPU memory in MiB, for hardware without nvidia-smi "
                    "(the GPU is assumed to be empty)")
a = p.parse_args()

if a.gpu_memory:
    used, total = 0, a.gpu_memory
else:
    try:
        used, total = read_gpu()
    except (FileNotFoundError, subprocess.CalledProcessError):
        total = 0
    if total == 0:
        raise SystemExit("No NVIDIA GPU found (nvidia-smi missing or failed).\n"
                         "On other hardware use: --gpu-memory <MiB> "
                         "(the GPU is assumed to be empty).")

try:
    weights, layers, kv_heads, head_dim = model_specs(a.model)
except (OSError, StopIteration, KeyError):
    raise SystemExit(f"Cannot read '{a.model}' from Ollama. "
                     f"Is Ollama running, and did you run: ollama pull {a.model} ?")

free = total - used
needed = predict_vram(a.context_length * a.users, weights, layers,
                      kv_heads, head_dim, a.overhead)
kv_per_user = (needed - weights - a.overhead) / a.users
word = "user" if a.users == 1 else "users"

print(f"GPU total:     {total} MiB")
print(f"GPU used now:  {used} MiB")
print(f"GPU free:      {free} MiB")
if a.gpu_memory:
    print("Note:          GPU size entered by hand; the GPU is assumed to be empty.")
if used / total > 0.10:
    print("Note:          the GPU is already in use (a loaded model or other programs).")
    print("               The estimate assumes the model is NOT loaded yet.")
    print("               For a clean check run: ollama stop <model>")
print(f"Needed (est.): {needed:.0f} MiB")
print(f"               to run {a.model} for {a.users} {word} at the same time,")
print(f"               each with a context length of {a.context_length} tokens")
print(f"               = model {weights:.0f} + context memory {a.users} x {kv_per_user:.0f}"
      f" + overhead {a.overhead:.0f}")
if needed <= free:
    spare = free - needed
    if spare / free < 0.10:
        print(f"Verdict:       FITS, but only {spare / free:.0%} spare ({spare:.0f} MiB)")
    else:
        print(f"Verdict:       FITS, {spare:.0f} MiB to spare")
    if a.users > 1:
        print(f"To serve them in parallel, set OLLAMA_NUM_PARALLEL={a.users}")
else:
    print(f"Verdict:       DOES NOT FIT, short by {needed - free:.0f} MiB")
