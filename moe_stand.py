import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super.__init__()
        # first layer
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # activation
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        # Router has 2 layers = FFNN (linear layer) + Softmax
        self.gate = nn.Linear(input_dim, hidden_dim)
        # mixture of experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device
        # gate instance
        logits = self.gate(x)
        # probability
        probs = torch.softmax(logits, dim=-1)
        # top k
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # Auxiliary loss = Coefficient Variation (CV) = std / mean, load balancing for each sample's capacity
        # if during training, implement experts load balancing
        if self.training:
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)
            # load balance loss
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()

            # aux loss, add up two loss functions
            aux_loss = importance_loss + load_balance_loss
        # if not training, during reasoning
        else:
            aux_loss = 0

        # expert assignment by 1-dim expert index with corresponding probability
        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)

        # sample index, with same batch and relevant topic
        sample_indices = torch.arange(batch_size, device=device)[:, None]
        # 1-1 mapping on expert index probability, so that we get which sample is assigned to which expert with how much weight
        sample_indices = sample_indices.expand(-1, self.top_k).flatten()

        # initialization
        outputs = torch.zeros(batch_size, self.expert[0].net[-1].out_features, device=device)

        # iterate through each expert for calculation
        for expert_idx in range(self.num_experts):
            # each batch has an index, for each sample being assigned to current expert
            expert_mask = flat_indices == expert_idx
            # get the expert sample and index with their weights
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            # control Expert Capacity = this expert can only process how many tokens
            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue

            # calculation: expert_samples be the input of def forward()
            expert_input = x[expert_samples]
            # pass the input params to current expert defined in class Expert()
            expert_output = self.experts[expert_idx](expert_input)
            # multiply to get the weights
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            # Keep Top-K: after we get top-k experts, aggregate each expert's output
            outputs.index_add_(0, expert_samples, weighted_output)
        return outputs, aux_loss

if __name__ == '__main__':
    # hyperparams
    input_dim = 128
    output_dim = 256
    hidden_dim = 512
    num_experts = 8
    top_k = 2
    expert_capacity = 32 # max sample for each expert to get
    batch_size = 64

    device = torch.device("npu:3" if torch.npu.is_available() else "cpu")
    x = torch.rand(batch_size, input_dim).to(device)
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)

    # iterations for training
    for _ in range(1000):
        moe.train()
        output, loss = moe(x)
        print(f"Training output shape: {output.shape}")
        print(f"Training Auxiliary loss: {loss.item():.f4}")
    print("=" * 50)
    # reasoning
    moe.eval()
    output, _ = moe(x)
    print(f"eval output shape: {output.shape}")