from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class DirectedMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.parent_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.child_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, adjacency: Tensor, node_mask: Tensor) -> Tensor:
        if adjacency.ndim == 3:
            incoming_degree = adjacency.transpose(1, 2).sum(dim=-1, keepdim=True).clamp_min(1.0)
            outgoing_degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            parents = torch.bmm(adjacency.transpose(1, 2), hidden) / incoming_degree
            children = torch.bmm(adjacency, hidden) / outgoing_degree
        elif adjacency.ndim == 2:
            if hidden.shape[0] != 1 or adjacency.shape[0] != 2:
                raise ValueError("sparse edge_index currently supports one variable-size DAG per batch")
            source, target = adjacency.long()
            flat_hidden = hidden[0]
            node_count = flat_hidden.shape[0]
            parent_sum = torch.zeros_like(flat_hidden)
            child_sum = torch.zeros_like(flat_hidden)
            parent_sum.index_add_(0, target, flat_hidden[source])
            child_sum.index_add_(0, source, flat_hidden[target])
            degree_dtype = flat_hidden.dtype
            parent_degree = torch.zeros(node_count, 1, device=hidden.device, dtype=degree_dtype)
            child_degree = torch.zeros(node_count, 1, device=hidden.device, dtype=degree_dtype)
            ones = torch.ones(source.shape[0], 1, device=hidden.device, dtype=degree_dtype)
            parent_degree.index_add_(0, target, ones)
            child_degree.index_add_(0, source, ones)
            parents = (parent_sum / parent_degree.clamp_min(1.0)).unsqueeze(0)
            children = (child_sum / child_degree.clamp_min(1.0)).unsqueeze(0)
        else:
            raise ValueError("adjacency must be dense [B,N,N] or sparse edge_index [2,E]")
        updated = self.self_linear(hidden) + self.parent_linear(parents) + self.child_linear(children)
        updated = self.normalization(torch.nn.functional.silu(updated))
        return updated * node_mask.unsqueeze(-1)


class FiLM(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, max(8, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(8, hidden_dim // 2), 2 * hidden_dim),
        )

    def forward(self, hidden: Tensor, delay_weight: Tensor) -> Tensor:
        gamma, beta = self.network(delay_weight).chunk(2, dim=-1)
        while gamma.ndim < hidden.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return hidden * (1.0 + gamma) + beta


class HierarchicalConstrainedPolicy(nn.Module):
    """Shared inductive DAG GNN for one fixed delivery-first objective."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        num_graph_layers: int,
        num_executor_actions: int,
        num_macro_actions: int,
        num_uavs: int = 1,
        cost_keys: Sequence[str],
        action_feature_dim: int | None = None,
        causal_executor_prior_scale: float = 0.0,
        causal_macro_prior_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if num_graph_layers <= 0 or hidden_dim <= 0:
            raise ValueError("GNN depth and hidden dimension must be positive")
        self.cost_keys = tuple(cost_keys)
        self.num_executor_actions = int(num_executor_actions)
        self.num_macro_actions = int(num_macro_actions)
        self.num_uavs = int(num_uavs)
        if self.num_uavs <= 0:
            raise ValueError("the hierarchical policy requires at least one UAV")
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.graph_layers = nn.ModuleList(
            DirectedMessagePassingLayer(hidden_dim) for _ in range(num_graph_layers)
        )
        self.action_feature_dim = action_feature_dim
        self.causal_executor_prior_scale = float(causal_executor_prior_scale)
        self.causal_macro_prior_scale = float(causal_macro_prior_scale)
        if not math.isfinite(self.causal_executor_prior_scale) or self.causal_executor_prior_scale < 0.0:
            raise ValueError("causal executor prior scale must be finite and non-negative")
        if not math.isfinite(self.causal_macro_prior_scale) or self.causal_macro_prior_scale < 0.0:
            raise ValueError("causal macro prior scale must be finite and non-negative")
        if self.causal_macro_prior_scale > 0.0 and (
            self.num_macro_actions != 9 or global_feature_dim < 9
        ):
            raise ValueError(
                "causal macro prior requires nine macro zones and nine demand features"
            )
        if self.causal_executor_prior_scale > 0.0 and (
            action_feature_dim is None or action_feature_dim < 12
        ):
            raise ValueError("causal executor prior requires the twelve executor features")
        if action_feature_dim is None:
            self.executor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_executor_actions),
            )
        else:
            self.executor_query = nn.Linear(hidden_dim, hidden_dim)
            self.executor_action_encoder = nn.Sequential(
                nn.Linear(action_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.executor_node_bias = nn.Linear(hidden_dim, 1)
        centralized_dim = hidden_dim + global_feature_dim
        self.central_encoder = nn.Sequential(
            nn.Linear(centralized_dim, hidden_dim),
            nn.SiLU(),
        )
        self.macro_head = nn.Linear(hidden_dim, num_macro_actions)
        self.reward_critic = nn.Linear(hidden_dim, 1)
        self.cost_critics = nn.ModuleDict({key: nn.Linear(hidden_dim, 1) for key in self.cost_keys})

    @staticmethod
    def _validate_inputs(
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
    ) -> None:
        if node_features.ndim != 3 or adjacency.ndim not in (2, 3) or node_mask.ndim != 2:
            raise ValueError("batched graph tensors have invalid rank")
        batch, nodes, _ = node_features.shape
        dense_valid = adjacency.ndim == 3 and adjacency.shape == (batch, nodes, nodes)
        sparse_valid = adjacency.ndim == 2 and adjacency.shape[0] == 2 and batch == 1
        if not (dense_valid or sparse_valid) or node_mask.shape != (batch, nodes):
            raise ValueError("graph tensor shapes are inconsistent")
        if sparse_valid and adjacency.numel():
            if adjacency.dtype not in (torch.int32, torch.int64):
                raise ValueError("sparse edge_index must use integer indices")
            if bool(((adjacency < 0) | (adjacency >= nodes)).any()):
                raise ValueError("sparse edge_index contains an invalid node index")
        if global_features.shape[0] != batch or delay_weight.shape != (batch, 1):
            raise ValueError("global feature or compatibility-scalar batch shape is invalid")
        if not bool(torch.isfinite(delay_weight).all()):
            raise ValueError("compatibility scalar must be finite")
        if torch.any((delay_weight < 0.0) | (delay_weight > 1.0)):
            raise ValueError("compatibility scalar must lie in [0, 1]")

    def encode_graph(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        delay_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.node_encoder(node_features) * node_mask.unsqueeze(-1)
        del delay_weight
        for layer in self.graph_layers:
            hidden = layer(hidden, adjacency, node_mask)
            hidden = hidden * node_mask.unsqueeze(-1)
        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        pooled = hidden.sum(dim=1) / denominator
        return hidden, pooled

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
        *,
        executor_features: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        self._validate_inputs(node_features, adjacency, node_mask, global_features, delay_weight)
        node_hidden, pooled = self.encode_graph(
            node_features,
            adjacency,
            node_mask,
            delay_weight,
        )
        if self.action_feature_dim is None:
            executor_logits = self.executor_head(node_hidden)
        else:
            if executor_features is None or executor_features.ndim != 3:
                raise ValueError("shared executor scoring requires [B,A,F] action features")
            if (
                executor_features.shape[0] != node_hidden.shape[0]
                or executor_features.shape[1] != self.num_executor_actions
                or executor_features.shape[2] != self.action_feature_dim
            ):
                raise ValueError("executor action feature shape is inconsistent")
            queries = self.executor_query(node_hidden)
            keys = self.executor_action_encoder(executor_features)
            executor_logits = torch.einsum("bnh,bah->bna", queries, keys) / math.sqrt(
                queries.shape[-1]
            )
            executor_logits = executor_logits + self.executor_node_bias(node_hidden)
            if self.causal_executor_prior_scale > 0.0:
                cpu = executor_features[..., 4]
                rate = executor_features[..., 6]
                queue = executor_features[..., 7]
                locality = executor_features[..., 8]
                missing = executor_features[..., 9]
                margin = executor_features[..., 10]
                feasible = executor_features[..., 11]
                local = executor_features[..., 0]
                remote = executor_features[..., 1] + executor_features[..., 2]
                defer = executor_features[..., 3]
                # The residual prior expresses the causal contract, not a
                # generic "fastest server" heuristic.  A remote executor is
                # attractive only when the currently missing predecessor
                # results can be delivered in the forecast contact window.
                # The same rule applies when a nominally local action still
                # needs predecessor results currently held by infrastructure;
                # otherwise a long DAG can strand an entire successor chain at
                # its owner after the RSU contact has ended.  An infeasible
                # forecast remains a soft penalty, so pause/resume is legal.
                delivery_safety = remote * (
                    8.0 * feasible - 4.0 + 2.0 * margin
                ) + local * (1.0 - feasible) * (
                    -4.0 + 2.0 * margin
                )
                causal_prior = (
                    2.0 * cpu
                    + 3.0 * locality
                    + delivery_safety
                    + 0.50 * rate
                    - 2.0 * queue
                    - missing
                    - 2.0 * defer
                )
                executor_logits = executor_logits + (
                    self.causal_executor_prior_scale
                    * causal_prior[:, None, :]
                )
        executor_logits = executor_logits.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
        central = self.central_encoder(torch.cat((pooled, global_features), dim=-1))
        macro_logits = self.macro_head(central)
        if self.causal_macro_prior_scale > 0.0:
            zone_demand = global_features[..., -9:]
            coverage_scores = torch.stack(
                tuple(
                    sum(
                        zone_demand[..., (action + uav_index) % 9]
                        / float(uav_index + 1)
                        for uav_index in range(self.num_uavs)
                    )
                    for action in range(self.num_macro_actions)
                ),
                dim=-1,
            )
            macro_logits = macro_logits + (
                self.causal_macro_prior_scale * coverage_scores
            )
        return {
            "executor_logits": executor_logits,
            "macro_logits": macro_logits,
            "reward_value": self.reward_critic(central).squeeze(-1),
            "cost_values": {
                key: critic(central).squeeze(-1) for key, critic in self.cost_critics.items()
            },
        }

    @torch.no_grad()
    def act(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
        *,
        decision_mask: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
        executor_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = self(
            node_features,
            adjacency,
            node_mask,
            global_features,
            delay_weight,
            executor_features=executor_features,
        )
        logits = output["executor_logits"]
        if (
            not isinstance(logits, Tensor)
            or action_mask.shape != logits.shape
            or decision_mask.shape != node_mask.shape
        ):
            raise ValueError("executor action mask shape is invalid")
        active_mask = node_mask & decision_mask
        defer_only_mask = torch.zeros_like(action_mask)
        defer_only_mask[..., -1] = True
        safe_action_mask = torch.where(
            active_mask.unsqueeze(-1), action_mask, defer_only_mask
        )
        active_allowed = safe_action_mask.any(dim=-1) | ~active_mask
        if not bool(active_allowed.all()):
            raise ValueError("every active DAG node must retain at least one legal action")
        masked_logits = logits.masked_fill(~safe_action_mask, -torch.inf)
        masked_logits = torch.where(
            active_mask.unsqueeze(-1),
            masked_logits,
            torch.zeros_like(masked_logits),
        )
        distribution = Categorical(logits=masked_logits)
        actions = masked_logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probability = distribution.log_prob(actions)
        entropy = distribution.entropy()
        actions = actions.masked_fill(~active_mask, -1)
        log_probability = log_probability.masked_fill(~active_mask, 0.0)
        entropy = entropy.masked_fill(~active_mask, 0.0)
        return actions, log_probability, entropy

    def evaluate_actions(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
        *,
        decision_mask: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        executor_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor | dict[str, Tensor]]]:
        """Re-evaluate stored node actions and aggregate them per graph for PPO."""
        output = self(
            node_features,
            adjacency,
            node_mask,
            global_features,
            delay_weight,
            executor_features=executor_features,
        )
        logits = output["executor_logits"]
        if (
            not isinstance(logits, Tensor)
            or action_mask.shape != logits.shape
            or decision_mask.shape != node_mask.shape
        ):
            raise ValueError("executor action mask shape is invalid")
        if actions.shape != node_mask.shape:
            raise ValueError("recorded action shape is invalid")
        active_mask = node_mask & decision_mask
        defer_only_mask = torch.zeros_like(action_mask)
        defer_only_mask[..., -1] = True
        safe_action_mask = torch.where(
            active_mask.unsqueeze(-1), action_mask, defer_only_mask
        )
        if not bool((safe_action_mask.any(dim=-1) | ~active_mask).all()):
            raise ValueError("every active DAG node must retain at least one legal action")
        safe_actions = actions.masked_fill(~active_mask, logits.shape[-1] - 1)
        active_actions = safe_actions[active_mask]
        if active_actions.numel() and bool(
            ((active_actions < 0) | (active_actions >= logits.shape[-1])).any()
        ):
            raise ValueError("recorded action index is outside the executor action space")
        selected_legal = safe_action_mask.gather(
            -1, safe_actions.unsqueeze(-1)
        ).squeeze(-1)
        if not bool((selected_legal | ~active_mask).all()):
            raise ValueError("recorded action is illegal under its causal action mask")
        masked_logits = logits.masked_fill(~safe_action_mask, -torch.inf)
        masked_logits = torch.where(
            active_mask.unsqueeze(-1),
            masked_logits,
            torch.zeros_like(masked_logits),
        )
        distribution = Categorical(logits=masked_logits)
        node_log_prob = distribution.log_prob(safe_actions).masked_fill(~active_mask, 0.0)
        node_entropy = distribution.entropy().masked_fill(~active_mask, 0.0)
        denominator = active_mask.sum(dim=-1).clamp_min(1).to(logits.dtype)
        return (
            node_log_prob.sum(dim=-1) / denominator,
            node_entropy.sum(dim=-1) / denominator,
            output,
        )
