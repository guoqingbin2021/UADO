from __future__ import annotations

import json
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


UPSTREAM_COMMITS = {
    "MAPPO": "de66d7a4b23fac2513f56f96f73b3f5cb96695ac",
    "HAPPO": "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2",
    "AMCoEdge": "fd921f8e11f7b3ca1cf0b4a7baeffb45d0a7ffe6",
    "FDEdge": "551988866f934b5cdb672e9122279a73eba60259",
    "MEC-UARA": "304a60a0a9db5d0a41242e150dca6e383b1c68c7",
}

UPSTREAM_REQUIRED_FILES = {
    "MAPPO": (
        "onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py",
        "onpolicy/algorithms/r_mappo/r_mappo.py",
    ),
    "HAPPO": (
        "harl/algorithms/actors/happo.py",
        "harl/algorithms/actors/on_policy_base.py",
    ),
    "AMCoEdge": ("AdaDQN.py", "environment.py", "main.py"),
    "FDEdge": ("fdedge_main.py", "fdsac_model.py", "feedback_diffusion.py"),
    "MEC-UARA": ("agent.py", "env.py", "main.py"),
}

UPSTREAM_ADAPTER_CLASSES = {
    "MAPPO": "MAPPOAdaptedPolicy",
    "HAPPO": "HAPPOAdaptedPolicy",
    "AMCoEdge": "AMCoEdgeRuntime",
    "FDEdge": "FDEdgeRuntime",
    "MEC-UARA": "MECUARARuntime",
}


class Box:
    """Minimal gym-compatible observation-space descriptor used by frozen upstream code."""

    def __init__(self, dimension: int) -> None:
        self.shape = (int(dimension),)


class Discrete:
    """Minimal gym-compatible discrete action-space descriptor used by frozen upstream code."""

    def __init__(self, actions: int) -> None:
        self.n = int(actions)


def _enable_frozen_marl_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative in ("third_party/on-policy", "third_party/HARL"):
        source_root = str(project_root / relative)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)


def _require_frozen_component(component, relative_root: str) -> None:
    module = sys.modules[component.__module__]
    source = Path(module.__file__).resolve()
    expected_root = (Path(__file__).resolve().parents[1] / relative_root).resolve()
    if not source.is_relative_to(expected_root):
        raise RuntimeError(
            f"{component.__name__} was imported outside the frozen source tree: {source}"
        )


def verify_upstream_sources(project_root: str | Path | None = None) -> dict[str, dict[str, object]]:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    manifest_path = root / "third_party/SOURCES.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = payload.get("methods", {})
    if set(methods) != set(UPSTREAM_COMMITS):
        raise RuntimeError(
            "formal SOTA source manifest must contain exactly "
            "MAPPO, HAPPO, AMCoEdge, FDEdge, and MEC-UARA"
        )
    third_party_root = (root / "third_party").resolve()
    report: dict[str, dict[str, object]] = {}
    for name, expected_commit in UPSTREAM_COMMITS.items():
        record = methods[name]
        if record.get("commit") != expected_commit:
            raise RuntimeError(f"{name} source commit does not match its executable adapter")
        source_root = (root / str(record.get("local_path", ""))).resolve()
        if not source_root.is_relative_to(third_party_root):
            raise RuntimeError(f"{name} source path escapes third_party")
        missing = [
            filename
            for filename in UPSTREAM_REQUIRED_FILES[name]
            if not (source_root / filename).is_file()
        ]
        if missing:
            raise RuntimeError(f"{name} official source files are missing: {missing}")
        report[name] = {
            "commit": expected_commit,
            "source_path": str(source_root),
            "source_present": True,
            "adapter": UPSTREAM_ADAPTER_CLASSES[name],
        }
    return report


class _AdaptedPolicyBase(nn.Module, ABC):
    def __init__(
        self,
        *,
        name: str,
        global_feature_dim: int,
        hidden_dim: int,
        num_executor_actions: int,
        num_macro_actions: int,
        cost_keys: Sequence[str],
        action_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.method_name = name
        self.upstream_commit = UPSTREAM_COMMITS[name]
        self.cost_keys = tuple(cost_keys)
        self.global_feature_dim = int(global_feature_dim)
        self.num_executor_actions = int(num_executor_actions)
        self.action_feature_dim = action_feature_dim
        if action_feature_dim is not None:
            self.executor_query = nn.Linear(hidden_dim, hidden_dim)
            self.executor_action_encoder = nn.Sequential(
                nn.Linear(action_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.executor_node_bias = nn.Linear(hidden_dim, 1)
        self.central = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim + 1, hidden_dim),
            nn.SiLU(),
        )
        self.macro_head = nn.Linear(hidden_dim, num_macro_actions)
        self.reward_critic = nn.Linear(hidden_dim, 1)
        self.cost_critics = nn.ModuleDict(
            {key: nn.Linear(hidden_dim, 1) for key in self.cost_keys}
        )

    @abstractmethod
    def encode_nodes(
        self,
        node_features: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
    ) -> Tensor:
        """Encode ready-node context using the method-specific upstream design."""

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
        del adjacency
        if node_features.ndim != 3 or node_mask.shape != node_features.shape[:2]:
            raise ValueError("baseline graph tensor shapes are inconsistent")
        if global_features.shape != (node_features.shape[0], self.global_feature_dim):
            raise ValueError("baseline global feature shape is inconsistent")
        if delay_weight.shape != (node_features.shape[0], 1):
            raise ValueError("baseline preference shape is inconsistent")
        hidden = self.encode_nodes(node_features, node_mask, global_features, delay_weight)
        if self.action_feature_dim is None:
            logits = self.executor_head(hidden)
        else:
            if (
                executor_features is None
                or executor_features.ndim != 3
                or executor_features.shape
                != (hidden.shape[0], self.num_executor_actions, self.action_feature_dim)
            ):
                raise ValueError("adapted baseline scoring requires [B,A,F] action features")
            queries = self.executor_query(hidden)
            keys = self.executor_action_encoder(executor_features)
            logits = torch.einsum("bnh,bah->bna", queries, keys) / math.sqrt(queries.shape[-1])
            logits = logits + self.executor_node_bias(hidden)
        logits = logits.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        pooled = hidden.sum(dim=1) / denominator
        central = self.central(torch.cat((pooled, global_features, delay_weight), dim=-1))
        return {
            "executor_logits": logits,
            "macro_logits": self.macro_head(central),
            "reward_value": self.reward_critic(central).squeeze(-1),
            "cost_values": {
                key: critic(central).squeeze(-1) for key, critic in self.cost_critics.items()
            },
        }

    @staticmethod
    def _masked_distribution(
        logits: Tensor,
        node_mask: Tensor,
        decision_mask: Tensor,
        action_mask: Tensor,
    ) -> Categorical:
        if action_mask.shape != logits.shape or decision_mask.shape != node_mask.shape:
            raise ValueError("baseline action mask shape is invalid")
        active_mask = node_mask & decision_mask
        defer_only_mask = torch.zeros_like(action_mask)
        defer_only_mask[..., -1] = True
        safe_action_mask = torch.where(
            active_mask.unsqueeze(-1), action_mask, defer_only_mask
        )
        if not bool((safe_action_mask.any(dim=-1) | ~active_mask).all()):
            raise ValueError("every active DAG node must retain a legal action")
        masked = logits.masked_fill(~safe_action_mask, -torch.inf)
        masked = torch.where(active_mask.unsqueeze(-1), masked, torch.zeros_like(masked))
        return Categorical(logits=masked)

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
        if not isinstance(logits, Tensor):
            raise TypeError("executor logits must be a tensor")
        active_mask = node_mask & decision_mask
        defer_only_mask = torch.zeros_like(action_mask)
        defer_only_mask[..., -1] = True
        safe_action_mask = torch.where(
            active_mask.unsqueeze(-1), action_mask, defer_only_mask
        )
        distribution = self._masked_distribution(
            logits, node_mask, decision_mask, action_mask
        )
        masked_logits = logits.masked_fill(~safe_action_mask, -torch.inf)
        actions = masked_logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probability = distribution.log_prob(actions).masked_fill(~active_mask, 0.0)
        entropy = distribution.entropy().masked_fill(~active_mask, 0.0)
        return actions.masked_fill(~active_mask, -1), log_probability, entropy

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
            or actions.shape != node_mask.shape
            or decision_mask.shape != node_mask.shape
        ):
            raise ValueError("recorded baseline action shape is invalid")
        active_mask = node_mask & decision_mask
        defer_only_mask = torch.zeros_like(action_mask)
        defer_only_mask[..., -1] = True
        safe_action_mask = torch.where(
            active_mask.unsqueeze(-1), action_mask, defer_only_mask
        )
        safe_actions = actions.masked_fill(~active_mask, logits.shape[-1] - 1)
        selected_legal = safe_action_mask.gather(
            -1, safe_actions.unsqueeze(-1)
        ).squeeze(-1)
        if not bool((selected_legal | ~active_mask).all()):
            raise ValueError("recorded baseline action violates its causal mask")
        distribution = self._masked_distribution(
            logits, node_mask, decision_mask, action_mask
        )
        denominator = active_mask.sum(dim=-1).clamp_min(1).to(logits.dtype)
        log_probability = distribution.log_prob(safe_actions).masked_fill(~active_mask, 0.0)
        entropy = distribution.entropy().masked_fill(~active_mask, 0.0)
        return (
            log_probability.sum(dim=-1) / denominator,
            entropy.sum(dim=-1) / denominator,
            output,
        )


class _OfficialMARLAdaptedPolicy(_AdaptedPolicyBase):
    """Common tensor contract for official MARL actors adapted to DAG executor choices."""

    def __init__(
        self,
        *,
        name: str,
        node_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        action_feature_dim: int | None,
        **kwargs,
    ) -> None:
        if action_feature_dim is None or action_feature_dim < 4:
            raise ValueError("general MARL baselines require executor type and resource features")
        super().__init__(
            name=name,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            action_feature_dim=None,
            **kwargs,
        )
        self.node_feature_dim = int(node_feature_dim)
        self.executor_feature_dim = int(action_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.pair_observation_dim = (
            self.node_feature_dim + self.global_feature_dim + 1 + self.executor_feature_dim
        )

    def encode_nodes(self, node_features, node_mask, global_features, delay_weight):
        raise NotImplementedError("official MARL adapters construct executor-conditioned node encodings")

    def _pair_observations(
        self,
        node_features: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
        executor_features: Tensor | None,
    ) -> Tensor:
        if node_features.ndim != 3 or node_mask.shape != node_features.shape[:2]:
            raise ValueError("general MARL graph tensor shapes are inconsistent")
        batch_size, node_count, _ = node_features.shape
        if global_features.shape != (batch_size, self.global_feature_dim):
            raise ValueError("general MARL global feature shape is inconsistent")
        if delay_weight.shape != (batch_size, 1):
            raise ValueError("general MARL preference shape is inconsistent")
        if (
            executor_features is None
            or executor_features.shape
            != (batch_size, self.num_executor_actions, self.executor_feature_dim)
        ):
            raise ValueError("general MARL policies require [B,A,F] executor features")
        action_count = self.num_executor_actions
        nodes = node_features.unsqueeze(2).expand(-1, -1, action_count, -1)
        global_context = global_features[:, None, None, :].expand(
            -1, node_count, action_count, -1
        )
        preference = delay_weight[:, None, None, :].expand(
            -1, node_count, action_count, -1
        )
        executors = executor_features[:, None, :, :].expand(-1, node_count, -1, -1)
        return torch.cat((nodes, global_context, preference, executors), dim=-1)

    def _finish_forward(
        self,
        *,
        hidden: Tensor,
        logits: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        delay_weight: Tensor,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        hidden = hidden * node_mask.unsqueeze(-1)
        logits = logits.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        pooled = hidden.sum(dim=1) / denominator
        central = self.central(torch.cat((pooled, global_features, delay_weight), dim=-1))
        return {
            "executor_logits": logits,
            "macro_logits": self.macro_head(central),
            "reward_value": self.reward_critic(central).squeeze(-1),
            "cost_values": {
                key: critic(central).squeeze(-1) for key, critic in self.cost_critics.items()
            },
        }


class MAPPOAdaptedPolicy(_OfficialMARLAdaptedPolicy):
    """Official shared MAPPO actor and centralized critic adapted to masked DAG actions."""

    official_components = ("R_MAPPOPolicy", "R_MAPPO")
    update_mode = "simultaneous"

    def __init__(self, *, hidden_dim: int, **kwargs) -> None:
        super().__init__(name="MAPPO", hidden_dim=hidden_dim, **kwargs)
        _enable_frozen_marl_imports()
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

        _require_frozen_component(R_MAPPOPolicy, "third_party/on-policy")
        args = SimpleNamespace(
            lr=3e-4,
            critic_lr=3e-4,
            opti_eps=1e-5,
            weight_decay=0.0,
            hidden_size=hidden_dim,
            gain=0.01,
            use_orthogonal=True,
            use_policy_active_masks=True,
            use_naive_recurrent_policy=False,
            use_recurrent_policy=False,
            recurrent_N=1,
            algorithm_name="rmappo",
            use_popart=False,
            use_feature_normalization=True,
            use_ReLU=True,
            stacked_frames=1,
            layer_N=1,
        )
        centralized_dimension = hidden_dim + self.global_feature_dim + 1
        official_policy = R_MAPPOPolicy(
            args,
            Box(self.pair_observation_dim),
            Box(centralized_dimension),
            Discrete(self.num_executor_actions),
        )
        self.shared_actor = official_policy.actor
        self.central = official_policy.critic.base
        self.reward_critic = official_policy.critic.v_out

    def actor_update_groups(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        return (
            tuple(self.shared_actor.parameters()) + tuple(self.macro_head.parameters()),
        )

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
        del adjacency
        pair_observations = self._pair_observations(
            node_features,
            node_mask,
            global_features,
            delay_weight,
            executor_features,
        )
        batch_size, node_count, action_count, _ = pair_observations.shape
        pair_hidden = self.shared_actor.base(
            pair_observations.reshape(-1, self.pair_observation_dim)
        ).reshape(batch_size, node_count, action_count, self.hidden_dim)
        all_logits = self.shared_actor.act.action_out.linear(
            pair_hidden.reshape(-1, self.hidden_dim)
        ).reshape(batch_size, node_count, action_count, action_count)
        logits = all_logits.diagonal(dim1=2, dim2=3)
        hidden = pair_hidden.mean(dim=2)
        return self._finish_forward(
            hidden=hidden,
            logits=logits,
            node_mask=node_mask,
            global_features=global_features,
            delay_weight=delay_weight,
        )


class HAPPOAdaptedPolicy(_OfficialMARLAdaptedPolicy):
    """Official HAPPO actors adapted as fixed-order heterogeneous executor agents."""

    official_components = ("HAPPO",)
    update_mode = "sequential"
    actor_group_names = ("local", "rsu", "uav", "defer")

    def __init__(self, *, hidden_dim: int, **kwargs) -> None:
        super().__init__(name="HAPPO", hidden_dim=hidden_dim, **kwargs)
        _enable_frozen_marl_imports()
        from harl.algorithms.actors.happo import HAPPO

        _require_frozen_component(HAPPO, "third_party/HARL")
        args = {
            "data_chunk_length": 1,
            "use_recurrent_policy": False,
            "use_naive_recurrent_policy": False,
            "use_policy_active_masks": True,
            "action_aggregation": "prod",
            "lr": 3e-4,
            "opti_eps": 1e-5,
            "weight_decay": 0.0,
            "hidden_sizes": [hidden_dim, hidden_dim],
            "gain": 0.01,
            "initialization_method": "orthogonal_",
            "activation_func": "relu",
            "use_feature_normalization": True,
            "recurrent_n": 1,
            "clip_param": 0.2,
            "ppo_epoch": 1,
            "actor_num_mini_batch": 1,
            "entropy_coef": 0.01,
            "use_max_grad_norm": True,
            "max_grad_norm": 0.5,
        }
        actors = []
        for _ in self.actor_group_names:
            algorithm = HAPPO(
                args,
                Box(self.pair_observation_dim),
                Discrete(self.num_executor_actions),
            )
            actors.append(algorithm.actor)
        self.heterogeneous_actors = nn.ModuleList(actors)

    def actor_update_groups(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        groups = [tuple(actor.parameters()) for actor in self.heterogeneous_actors]
        groups[-1] = groups[-1] + tuple(self.macro_head.parameters())
        return tuple(groups)

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
        del adjacency
        pair_observations = self._pair_observations(
            node_features,
            node_mask,
            global_features,
            delay_weight,
            executor_features,
        )
        if executor_features is None:
            raise AssertionError("validated executor features unexpectedly missing")
        batch_size, node_count, action_count, _ = pair_observations.shape
        group_ids = executor_features[..., :4].argmax(dim=-1)
        logits_by_batch = []
        hidden_by_batch = []
        for batch_index in range(batch_size):
            routed_logits = node_features.new_zeros((node_count, action_count))
            routed_hidden = node_features.new_zeros(
                (node_count, action_count, self.hidden_dim)
            )
            for group_index, actor in enumerate(self.heterogeneous_actors):
                action_indices = torch.nonzero(
                    group_ids[batch_index] == group_index,
                    as_tuple=False,
                ).squeeze(-1)
                if action_indices.numel() == 0:
                    continue
                selected = pair_observations[
                    batch_index, :, action_indices, :
                ].reshape(-1, self.pair_observation_dim)
                selected_hidden = actor.base(selected).reshape(
                    node_count, action_indices.numel(), self.hidden_dim
                )
                all_logits = actor.act.get_logits(
                    selected_hidden.reshape(-1, self.hidden_dim)
                )
                selected_action_ids = action_indices.unsqueeze(0).expand(
                    node_count, -1
                ).reshape(-1, 1)
                selected_logits = all_logits.gather(
                    dim=-1,
                    index=selected_action_ids,
                ).reshape(node_count, action_indices.numel())
                routed_logits = routed_logits.index_copy(
                    1, action_indices, selected_logits
                )
                routed_hidden = routed_hidden.index_copy(
                    1, action_indices, selected_hidden
                )
            logits_by_batch.append(routed_logits)
            hidden_by_batch.append(routed_hidden)
        routed_logits = torch.stack(logits_by_batch, dim=0)
        routed_hidden = torch.stack(hidden_by_batch, dim=0)
        hidden = routed_hidden.mean(dim=2)
        return self._finish_forward(
            hidden=hidden,
            logits=routed_logits,
            node_mask=node_mask,
            global_features=global_features,
            delay_weight=delay_weight,
        )


def build_adapted_baseline_policy(name: str, **kwargs) -> _AdaptedPolicyBase:
    classes = {
        "MAPPO": MAPPOAdaptedPolicy,
        "HAPPO": HAPPOAdaptedPolicy,
    }
    try:
        policy_class = classes[name]
    except KeyError as exc:
        raise ValueError(f"unsupported adapted baseline policy: {name}") from exc
    return policy_class(**kwargs)
