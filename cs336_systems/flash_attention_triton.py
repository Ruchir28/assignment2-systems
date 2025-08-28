import math
import torch

# Make Triton optional so this module can import on unsupported platforms.
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_AVAILABLE = True
except Exception:  # ImportError or any env-related error
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0,0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0,0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0),
        )

        query = tl.load(Q_block_ptr,boundary_check=(0,),padding_option="zero") #(Q_TILE_SIZE,D)
        output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        max_running_score = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
        l_sum = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

        for tile_k in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            keys = tl.load(K_block_ptr,boundary_check=(0,),padding_option="zero") #(K_TILE_SIZE,D)
            values = tl.load(V_block_ptr,boundary_check=(0,),padding_option="zero") #(K_TILE_SIZE,D)

            keys_T = tl.trans(keys)
            scores = tl.dot(query, keys_T) * scale #(Q_TILE_SIZE,K_TILE_SIZE)
            max_block = tl.max(scores, axis=1) #(Q_TILE_SIZE)
            max_new = tl.maximum(max_running_score, max_block) #(Q_TILE_SIZE)
            alpha = tl.exp(max_running_score - max_new) #(Q_TILE_SIZE)
            p_ = tl.exp(scores - max_new[..., None]) #(Q_TILE_SIZE,K_TILE_SIZE)
            l_sum = l_sum * alpha + tl.sum(p_, axis=1) #(Q_TILE_SIZE)
            output = output * alpha[..., None] + tl.dot(p_, values) #(Q_TILE_SIZE,D)
            max_running_score = max_new

            # advancing block pointer
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))    

        output = output / l_sum[..., None]
        tl.store(O_block_ptr, output, boundary_check=(0,))

        L = max_running_score + tl.log(l_sum)
        tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise ImportError("Triton is not available on this platform. Install the optional 'triton' extra on Linux x86_64 with CUDA.")

        B, Nq, D = Q.shape
        Nk = K.shape[1]
        device, dtype = Q.device, Q.dtype

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # Outputs
        O = torch.empty_like(Q)
        L = torch.zeros((B, Nq), device=device, dtype=torch.float32)

        T_q = tl.cdiv(Nq, Q_TILE_SIZE)

        flash_fwd_kernel[(T_q, B)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Nq,
            Nk,
            1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        if not _TRITON_AVAILABLE:
            raise ImportError("Triton is not available on this platform.")
        # Not yet implemented
        pass
