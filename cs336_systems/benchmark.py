import math
import time
import torch
from cs336_basics.model import BasicsTransformerLM

def benchmark_model(model, dummy_input):
    """
    A placeholder function to benchmark the model.
    In a real scenario, this would involve running the model on a dataset
    and measuring metrics like throughput and latency.
    """
    print("Starting benchmark...")
    
    # Warm-up
    for _ in range(5):
        _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    overall_start = time.time()

    step_times = []  # store duration of every forward pass

    for _ in range(10):
        step_start = time.time()

        _ = model(dummy_input)

        # ensure all GPU work is finished before stopping timer # Important thing to do 
        # when doing benchmarking because otherwise cpu can still move forward and we will get weird timings as o/p
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_times.append(time.time() - step_start)
        
    overall_end = time.time()

    avg_time_per_step = sum(step_times) / len(step_times)

    variance = sum((t - avg_time_per_step) ** 2 for t in step_times) / len(step_times)
    std_deviation = math.sqrt(variance)

    print(f"Average time per forward pass : {avg_time_per_step:.4f} seconds")
    print(f"Standard deviation per pass   : {std_deviation:.4f} seconds")
    print(f"Total time taken              : {overall_end - overall_start:.4f} seconds")


if __name__ == "__main__":
    # Model parameters
    vocab_size = 1000
    context_length = 128
    d_model = 256
    num_layers = 4
    num_heads = 4
    d_ff = 1024
    rope_theta = 10000.0

    # Create a model instance
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    
    # Create a dummy input tensor
    dummy_input = torch.randint(0, vocab_size, (1, context_length))
    
    # Move model and data to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        dummy_input = dummy_input.to('cuda')
    
    # Run benchmark
    benchmark_model(model, dummy_input)
