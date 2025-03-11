import torch
import torch.nn as nn
import torch.nn.functional as F

# distributed computing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torch_npu
from torch_npu.contrib import transfer_to_npu # replace cuda strings to npu automatically

# export MASTER_ADDR=""         # main IP
# export MASTER_PORT=""         # any available port

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)

# generate random simulation data in gaussian distribution
def gen_data(batch_size, input_dim):
    data = torch.randn(batch_size, input_dim)
    labels = torch.randn(batch_size, input_dim)
    return data, labels

# Parallel experts
class MoEEP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, top_k, capacity_factor):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([
            # .to: each expert goes to which device
            Expert(input_dim, hidden_dim, output_dim).to(f'cuda: {i}')
            for i in range(num_experts)
        ])

        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])

        # calculate router
        logits = self.router(x)
        # softmax
        probs = F.softmax(logits, dim=-1)
        # choose top k, and get index
        expert_weights, expert_indices = torch.topk(probs, self.top_k, dim=-1)

        # distributed computing
        world_size = dist.get_world_size()
        # expert capacity, based on # of clusters, capacity_factor is hyperparam,
        capacity = int(self.capacity_factor * batch_size / (self.top_k * world_size))
        capacity = max(capacity, 1) # cannot > 1

        # multiple-devices communication
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)
        # count # of each expert being selected
        expert_count = expert_mask.sum(dim=0)
        # all_reduce: aggregate the count of each expert being used in all processes, for global load balancing
        # ReduceOp.SUM: each expert is evenly used, if one expert is used too much, load balancer will control it
        dist.all_reduce(expert_count, op=dist.ReduceOp.SUM)

        # load balancing
        # each expert mean router probability
        density = probs.mean(dim=0)
        # each expert usage ratio, expert_count has been aggregated globally, so divide by batch size
        usage = expert_count / (batch_size * world_size)
        # load balance loss
        balance_loss = (density * usage).sum() * self.num_experts

        # distributed expert computation
        outputs = []
        for expert_id in range(self.num_experts):
            # locate each expert on which device
            device = f'cuda:{expert_id % torch.cuda.device_count()}'
            print(f"Current device: {device}")

            # which index is assigned to current expert
            idx_mask = (expert_indices == expert_id).any(dim=-1)
            if idx_mask.sum() == 0:
                continue

            # capacity control
            selected = torch.nonzero(idx_mask).flatten()
            selected = selected[capacity]
            # if after capacity control still empty, skip this iteration
            if selected.numel() == 0:
                continue

            # inter-device communication
            # pass input x to each device and each expert
            expert_input = x[selected].to(device)
            expert_output = self.experts[expert_id](expert_input)

            # then we create a weighted activation for each expert's calculation, then accumulate all weights and pass back to original device
            weights = expert_weights[selected, (expert_indices[selected] == expert_id).nonzero()[:, 1]]
            weights_output = (expert_output * weights.unsqueeze(-1)).to(x.device)
            outputs.append((selected, weights_output))

        # aggregate all outputs, each expert output be aggregated by original index to recover the original status
        final_output = torch.zeros_like(x)
        for selected, out in outputs:
            final_output[selected] += out

        # importance loss
        important = probs.sum(dim=0)
        # all_reduce: each expert has different importance in each device, inter-device/process aggregation
        dist.all_reduce(important, op=dist.ReduceOp.SUM)
        # load balance to avoid some experts being ignored, if so then loss is big
        important_loss = (important ** 2).mean()
        aux_loss = balance_loss + important_loss

        return final_output.view(*orig_shape), aux_loss

# distributed training
def step(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# training iteration, *** the core code ***
def train(rank, world_size):
    setup(rank, world_size)
    batch_size = 64
    input_dim = 128
    output_dim = 256
    hidden_dim = 512
    top_k = 2
    num_experts = 8 # 8 devices

    # to(rank), assign expert to corresponding device
    model = MoEEP(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        top_k=top_k,
        capacity_factor=1.2
    ).to(rank)
    # DistributedDataParallel
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # load data, generate simulation data
    data, label = gen_data(batch_size, input_dim)
    # pytorch format
    dataset = list(zip(torch.tensor(data), torch.tensor(label)))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_experts):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x = x.to(rank)
            y = y.to(rank)
            outputs, aux_loss = model(x)
            main_loss = F.mes_loss(outputs, y)
            # AdamW loss function calculation + mes_loss, aggregate balance loss + important loss, choose 0.2 as weight
            total_loss = main_loss + 0.2 * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

if __name__ == "__main__":
    world_size = 8 # 8 devices
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)




