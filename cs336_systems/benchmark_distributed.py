import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size=world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    time_start = time.perf_counter()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_end = time.perf_counter()
    print(f"rank {rank} data (after all-reduce): {data}")
    print(f"rank {rank} all-reduce time: {time_end - time_start:.6f} seconds")


if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)