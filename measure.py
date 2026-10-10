import subprocess
import psutil

MIB = 1024 * 1024
GIB = 1024 * MIB


def read_gpu():
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    used, total = out.strip().split(",")
    return int(used), int(total)


def read_ram():
    mem = psutil.virtual_memory()
    return mem.used // MIB, mem.total // MIB


def read_cpu():
    return psutil.cpu_percent(interval=1)


def read_disk():
    return psutil.disk_usage("/").free // GIB


if __name__ == "__main__":
    gpu_used, gpu_total = read_gpu()
    ram_used, ram_total = read_ram()
    print(f"VRAM: {gpu_used} / {gpu_total} MiB")
    print(f"RAM:  {ram_used} / {ram_total} MiB")
    print(f"CPU:  {read_cpu()} %")
    print(f"Disk free: {read_disk()} GiB")
