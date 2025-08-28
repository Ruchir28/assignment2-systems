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

        ctx.save_for_backward(L, Q, K, V, O, q_scaled)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):

        L, Q, K, V, O, q_scaled = ctx.saved_tensors

        B, Nq, D = Q.shape
        
        Nk = K.shape[1]
        
        device, dtype = Q.device, Q.dtype
        BQ = 64
        BK = 64

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        D = torch.sum(dO * O, dim=-1) #(B,Nq)


        for ks in range(0,Nk,BK):
            ke = min(ks + BK,Nk)
            k_j = K[:,ks:ke,:] #(B,bk,D)
            v_j = V[:,ks:ke,:] #(B,bk,D)
            dK_j = torch.zeros_like(k_j)
            dV_j = torch.zeros_like(v_j)
            
            for qs in range(0,Nq,BQ):
                qe = min(qs + BQ,Nq)
                q_i = q_scaled[:,qs:qe,:]
                o_i = O[:,qs:qe,:]
                dO_i = dO[:,qs:qe,:] # (B,bq,D)
                dQ_i = torch.zeros_like(q_i) # (B,bq,D)
                L_i = L[:,qs:qe] # (B,bq)
                D_i = D[:,qs:qe] # (B,bq)

                S_ij = torch.matmul(q_i, k_j.transpose(-1,-2)) #(B,bq,bk)
                P_ij = torch.exp(S_ij - L_i[...,None]) #(B,bq,bk)
                dV_j += torch.einsum('bqk,bqd->bkd', P_ij, dO_i) #(B,bk,D)
                dP_ij = torch.einsum('bqd,bkd->bqk', dO_i, v_j) #(B,bq,bk)
                dS_ij = P_ij * (dP_ij - D_i[...,None]) #(B,bq,bk)

                dQ_i += (1/math.sqrt(Q.shape[-1])) * torch.matmul(dS_ij, k_j) #(B,bq,D)
                dQ[:,qs:qe,:] += dQ_i

                dK_j += torch.matmul(dS_ij.transpose(-1,-2), q_i) #(B,bk,D)\

            dK[:,ks:ke,:] += dK_j
            dV[:,ks:ke,:] += dV_j

        return dQ, dK, dV, None