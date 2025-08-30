import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def data_distributed_parallel(rank, world_size):
    
    setup(rank, world_size=world_size)

    torch.manual_seed(42)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    scatter_list = None
    my_data_to_train = torch.empty(1,4)

    if rank == 0:
        data_gen = torch.Generator()
        data_gen.manual_seed(1234)
        data = torch.rand([4, 4], generator=data_gen)
        print(f"Initial data:{data} \n")
        scatter_list = list(torch.chunk(data, world_size, dim=0))

    dist.scatter(tensor=my_data_to_train, scatter_list=scatter_list, src=0)

    print(f"Rank {rank} received slice:{my_data_to_train} \n")

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(my_data_to_train)
        outputs = torch.log_softmax(outputs, dim=1)
        loss = torch.mean(outputs)
        loss.backward()

        global_loss = loss.clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        avg_global_loss = global_loss / world_size
        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch}, Global Average Loss: {avg_global_loss.item()}")

        # update the gradients on all machines

        dist.all_reduce(model.weight.grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(model.bias.grad, op=dist.ReduceOp.SUM)

        model.weight.grad /= world_size
        model.bias.grad /= world_size

        optimizer.step()
        
        print(f"Rank {rank}, Epoch {epoch}, Local Loss: {loss.item()}")

    dist.barrier()

    if rank == 0:
        torch.save(model.state_dict(), "distributed_model.pt")
        print(f"Rank 0: DDP model saved.")


def run_single_node_training(world_size):
    print("--- Starting Single-Node Training ---")
    torch.manual_seed(42) # Use the same seed

    # Match DDP data by using the same dedicated RNG
    data_gen = torch.Generator()
    data_gen.manual_seed(1234)
    data = torch.rand([4, 4], generator=data_gen)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(data)
        outputs = torch.log_softmax(outputs, dim=1)
        loss = torch.mean(outputs) # Simplified loss for this example
        loss.backward()
        # No all_reduce here!
        optimizer.step()
        print(f"Single-Node, Epoch {epoch}, Loss: {loss.item()} , shape: {loss.shape}")

    torch.save(model.state_dict(), "single_node_model.pt")
    print("--- Single-Node Training Finished. Model saved. ---")

if __name__ == "__main__":
    world_size = 4
    print("\\n--- Starting Distributed Training ---")
    mp.spawn(data_distributed_parallel, args=(world_size,), nprocs=world_size, join=True)
    print("--- Distributed Training Finished. ---")

    print("\\n--- Starting Single-Node Training for Comparison ---")
    run_single_node_training(world_size)
    print("--- Single-Node Training Finished. ---")

    print("\\n--- Comparing Models ---")
    single_node_sd = torch.load("single_node_model.pt")
    dist_sd = torch.load("distributed_model.pt")

    models_match = True
    for key in single_node_sd:
        if not torch.allclose(single_node_sd[key], dist_sd[key]):
            print(f"Mismatch found in parameter: {key}")
            models_match = False
            break

    if models_match:
        print("✅ Success! The models are identical.")
    else:
        print("❌ Failure! The models are different.")
