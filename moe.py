import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2, alpha_noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"

        """
        nn.Linear(self.bundle_sum_emb.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, self.embedding_size)
        """
        self.experts = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
            # nn.Linear(input_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        self.w_gate = nn.Parameter(
            torch.zeros(input_dim, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_dim, num_experts), requires_grad=True 
        )
        self.alpha_noise = alpha_noise

        self.softplus = nn.Softplus()

    def forward(self, x, noise_epsilon=1e-2):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # gate_logits = self.gate(x)
        # add noise to gate logits for exploration
        # 1e-1: good

        # noise = torch.randn_like(gate_logits) * self.alpha_noise
        # noise = torch.randn_like(gate_logits)
        # gate_logits = gate_logits + noise

        clean_logits = x @ self.w_gate
        if self.training:
            raw_noise_std = x @ self.w_noise
            noise_std = self.softplus(raw_noise_std) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            logits = clean_logits

        aux_loss = self._compute_load_balancing_loss(logits)

        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=1)
        
        topk_weights = F.softmax(topk_logits, dim=1)  # [batch_size, top_k]
        
        selected_experts = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        topk_weights = topk_weights.unsqueeze(-1)  # [batch_size, top_k, 1]
        output = torch.sum(topk_weights * selected_experts, dim=1)  # [batch_size, output_dim]
        
        return output, aux_loss
        # return output 
    
    def _compute_load_balancing_loss(self, gate_logits):
        gates = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        
        importance_per_expert = gates.mean(dim=0)  # [num_experts]

        target_importance = torch.ones_like(importance_per_expert) / self.num_experts
        importance_loss = F.kl_div(
                importance_per_expert.log(),
                target_importance,
                reduction='sum'
        )
        
        return importance_loss
    
if __name__ == "__main__":
    batch_size = 4
    input_dim = 16
    output_dim = 16
    num_experts = 4
    top_k = 2

    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = MoE_Layer(input_dim, output_dim, num_experts, top_k)
    x = torch.randn(batch_size, input_dim)
    output, aux_loss = model(x)

    print(model.w_noise)

    n_epochs = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output, aux_loss = model(x)
        loss = output.mean() + aux_loss * 0.01
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")
    
    print(model.w_noise)
    