import torch
import math

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        device, dtype = Q.device, Q.dtype

        BQ = 64
        BK = 64

        scale = 1.0 / math.sqrt(D)
        q_scaled = Q * scale

        # Outputs
        O = torch.empty(B, Nq, D, device=device, dtype=dtype) # outputs
        L = torch.zeros((B, Nq), device=device, dtype=dtype)


        for qs in range(0,Nq, BQ):
            qe = min(qs + BQ, Nq) # let's call it's dimension bq
            query_block = q_scaled[:,qs:qe,:] #(B,bq,D)
            max_running_score = torch.full((B,qe-qs),float("-inf"),device=device,dtype=dtype)
            output_block = torch.zeros((B,qe-qs,D),device=device,dtype=dtype) # (B,qe-qs,D)
            # running sum of exponentiated logits
            l = torch.zeros((B, qe - qs), device=device, dtype=dtype) #(B,bq)

            for ks in range(0,Nk,BK):
                ke = min(ks + BK,Nk)
                keys_block = K[:,ks:ke,:]
                value_block = V[:,ks:ke,:]
                scores_block = torch.matmul(query_block, keys_block.transpose(-1, -2))  # (B, bq, bk)
                max_block = scores_block.max(dim=-1).values #(B,bq)
                max_new = torch.maximum(max_running_score,max_block) #(B,bq)

                # exponentiated scores for the current block
                p_ = torch.exp(scores_block - max_new[..., None]) # (B, bq, bk)
                
                # Scaling factor to update running sums from the old max to the new max
                alpha = torch.exp(max_running_score - max_new)  # (B, bq)
                
                l = l * alpha + p_.sum(dim=-1) # (B, bq)
                
                output_block = output_block * alpha[..., None] + torch.einsum('bqk,bkd->bqd', p_, value_block)

                max_running_score = max_new
            
            O[:,qs:qe,:] = output_block / l[..., None]

            L[:,qs:qe] = max_running_score + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        pass