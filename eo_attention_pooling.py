# attention_pooling.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os

class PatchAttentionPooling(nn.Module):
    """
    Module that directs Attention to important areas such as "farmland" from a variable-length patch sequence,
    aggregating them into a single fixed-length vector (County Embedding).
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Lightweight MLP to calculate "importance score" for each patch (trainable)
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, patch_embeddings):
        # patch_embeddings shape: [Batch, Num_Patches, Embed_Dim]
        
        # 1. Calculate score for each patch
        # Output shape: [Batch, Num_Patches, 1]
        attn_scores = self.attention_net(patch_embeddings)
        
        # 2. Normalize weights with Softmax (sum becomes 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 3. Calculate weighted sum for pooling
        # [Batch, 1, Num_Patches] @ [Batch, Num_Patches, Embed_Dim] -> [Batch, 1, Embed_Dim]
        pooled_embedding = torch.bmm(attn_weights.transpose(1, 2), patch_embeddings)
        
        # Remove extra dimension and return
        return pooled_embedding.squeeze(1), attn_weights

def load_and_aggregate_patches(pt_dir=".", embed_dim=1024):
    """
    Function to load multiple saved extracted patches (Q) and apply Attention Pooling
    """
    # 1. Find all saved patch files
    file_pattern = os.path.join(pt_dir, "q_patch/extracted_q_patch_*.pt")
    patch_files = sorted(glob.glob(file_pattern))
    
    if not patch_files:
        print("Patch files not found. Please run extract_eo_features.py first.")
        return None

    print(f"Loading {len(patch_files)} patch files...")
    
    patch_vectors = []
    for f in patch_files:
        # shape: [1, Num_Tokens(e.g. 197), Embed_Dim(e.g. 1024)]
        q_tensor = torch.load(f)
        
        # As a vector representing the patch, extract the first CLS token
        # (Or average of all tokens torch.mean(q_tensor, dim=1) is also possible)
        cls_token = q_tensor[:, 0, :] # shape: [1, Embed_Dim]
        patch_vectors.append(cls_token)
    
    # Combine list into tensor of [Batch=1, Num_Patches, Embed_Dim]
    # This corresponds to "varying-length sequence of valid geographical patches" in the paper
    county_patches_tensor = torch.stack(patch_vectors, dim=1)
    
    # 2. Initialize Attention Pooling model
    model = PatchAttentionPooling(embed_dim=embed_dim)
    model.eval() # Inference mode for now (can be integrated into training loop later)
    
    # 3. Execute Pooling!
    with torch.no_grad():
        county_embedding, weights = model(county_patches_tensor)
        
    print("-" * 30)
    print(f"Input patches Shape : {county_patches_tensor.shape}")
    print(f"Final County Embedding (Q) Shape: {county_embedding.shape}")
    print(f"   -> This becomes the Q passed to Cross-Modal Attention!")
    
    # Display top weights for debugging
    top_weights, top_indices = torch.topk(weights.squeeze(), k=min(3, len(patch_files)))
    print(f"Indices of patches with highest Attention (weights): {top_indices.tolist()} (Weights: {top_weights.tolist()})")
    
    return county_embedding

if __name__ == "__main__":
    # Execute specifying directory with extracted files
    final_q = load_and_aggregate_patches(pt_dir=".")

    torch.save(final_q, "final_county_embedding_q.pt")
    print("Final Q saved as 'final_county_embedding_q.pt'!")