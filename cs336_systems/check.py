import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def main():
    # Pick device and matching autocast dtype (FP16 on CUDA, BF16 on CPU fallback)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    device = torch.device(device_type)
    model = ToyModel(10, 10).to(device)
    model.train()

    batch_size = 4
    inputs = torch.randn(batch_size, 10, device=device, dtype=torch.float32)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Capture intermediate output dtypes with forward hooks
    captured_dtypes = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured_dtypes[name] = out.dtype
        return hook

    hooks = [
        model.fc1.register_forward_hook(make_hook("fc1_output")),
        model.ln.register_forward_hook(make_hook("layernorm_output")),
        model.register_forward_hook(make_hook("logits")),
    ]

    # Autocast context
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        # Model parameter dtypes (these remain FP32 under autocast unless explicitly converted)
        param_dtypes = {name: p.dtype for name, p in model.named_parameters()}

        logits = model(inputs)
        loss = loss_fn(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Clean up hooks
    for h in hooks:
        h.remove()

    grad_dtypes = {name: (p.grad.dtype if p.grad is not None else None) for name, p in model.named_parameters()}

    # Print a concise report
    print(f"device_type: {device_type}")
    print(f"autocast dtype: {autocast_dtype}")
    print(f"parameter dtypes (unique): {sorted({str(dt) for dt in param_dtypes.values()})}")
    print(f"fc1 output dtype: {captured_dtypes.get('fc1_output')}")
    print(f"layernorm output dtype: {captured_dtypes.get('layernorm_output')}")
    print(f"logits dtype: {captured_dtypes.get('logits')}")
    print(f"loss dtype: {loss.dtype}")
    print(f"parameter grad dtypes (unique): {sorted({str(dt) for dt in grad_dtypes.values() if dt is not None})}")

if __name__ == "__main__":
    main()