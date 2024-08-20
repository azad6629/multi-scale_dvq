import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class VectorQuantizer(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, commitment_cost,decay=0.99,epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self._ema_w.data.normal_()
        
        self.decay = decay
        self.epsilon = epsilon
        
    @torch.no_grad()
    def compute_distances(self, flat_x):
        flat_x = F.normalize(flat_x, p=2, dim=1)
        weight = self.embeddings.weight
        weight = F.normalize(weight, p=2, dim=1)
        
        distances = (
                        torch.sum(flat_x ** 2, dim=1, keepdim=True) 
                        + torch.sum(weight ** 2, dim=1) 
                        - 2. * torch.matmul(flat_x, weight.t())
                    )  # [N, M]

        return distances
    
    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=True):
        distances = self.compute_distances(x)
        soft_code = F.softmax(-distances / temp, dim=-1)
        
        if stochastic:
            soft_code_flat = soft_code.view(-1, soft_code.shape[-1])
            code = torch.multinomial(soft_code_flat, 1)
            code = code.view(*soft_code.shape[:-1])
        else:
            code = distances.argmin(dim=-1)

        return soft_code, code

    def forward(self, inputs,temp=1.0, stochastic=True):        
        bs,channel = inputs.shape[0],inputs.shape[1]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape        
        flat_x = inputs.view(-1, self.embedding_dim)
        distances = self.compute_distances(flat_x)
        
        # Get soft codes
        soft_codes = F.softmax(-distances / temp, dim=-1)        
        #Soft codes represent the probabilities (or softmax scores) of each input vector 
        #belonging to each embedding vector in the codebook.
        
        if stochastic:
            soft_code_flat = soft_codes.view(-1, soft_codes.shape[-1])
            encoding_indices = torch.multinomial(soft_code_flat, 1)
            encoding_indices = encoding_indices.view(*soft_codes.shape[:-1])
        else:
            encoding_indices = distances.argmin(dim=-1)
                
        
        """Returns embedding tensor for a batch of indices."""
        encoding_indices = encoding_indices.unsqueeze(1) 
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encoding_indices, 0)
            
        # Laplace smoothing of the cluster size
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon)
                                 / (n + self.num_embeddings * self.epsilon) * n)
        
        dw = torch.matmul(encodings.t().to(flat_x.dtype), flat_x)
        self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)
        self.embeddings.weight = nn.Parameter(self._ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss          = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0,3, 2, 1).contiguous()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        soft_codes = soft_codes.view(bs,channel,-1 )

        return quantized,loss,perplexity,encoding_indices,soft_codes