# Adapted from qwen2.py
import logging
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.utils import add_prefix, is_cuda

import numpy as np

import math
from pathlib import Path
from sglang.srt.kernels.dfloat11.decode import get_decode_kernel

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_dfloat11: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        if is_dfloat11:
            self.qkv_proj = None
            self.o_proj = None
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=attention_bias,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                reduce_results=False,
                prefix=add_prefix("o_proj", prefix),
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream
        self.is_dfloat11 = is_dfloat11

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        df11_q_weight: Optional[torch.Tensor] = None,
        df11_k_weight: Optional[torch.Tensor] = None,
        df11_v_weight: Optional[torch.Tensor] = None,
        df11_o_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if df11_q_weight is not None:
            qkv_weight = torch.cat([df11_q_weight, df11_k_weight, df11_v_weight], dim=0)
            qkv = torch.nn.functional.linear(hidden_states, qkv_weight)
        else:
            qkv, _  = self.qkv_proj(hidden_states)
            
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        if df11_o_weight is not None:
            output = torch.nn.functional.linear(attn_output, df11_o_weight)
        else:
            output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.is_dfloat11 = hasattr(config, "dfloat11_config")
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_dfloat11=self.is_dfloat11
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            is_dfloat11=self.is_dfloat11
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        self.df11_weight_shapes = {
            "mlp.down_proj": (hidden_size, intermediate_size),
            "mlp.gate_proj": (intermediate_size, hidden_size),
            "mlp.up_proj": (intermediate_size, hidden_size),

            "self_attn.k_proj": (num_key_value_heads * head_dim, hidden_size),
            "self_attn.o_proj": (hidden_size, num_attention_heads * head_dim),
            "self_attn.q_proj": (num_attention_heads * head_dim, hidden_size),
            "self_attn.v_proj": (num_key_value_heads * head_dim, hidden_size),
        }

    def decode(self):
        decoded = {}

        encoded_exponent = getattr(self, "df11_encoded_exponent")
        sign_mantissa = getattr(self, "df11_sign_mantissa")
        luts = getattr(self, "df11_luts")
        gaps = getattr(self, "df11_gaps")
        output_positions = getattr(self, "df11_output_positions")
        split_positions = getattr(self, "df11_split_positions")
        shared_mem_size = self.df11_shared_mem_size
        n_luts = luts.shape[0]
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        threads_per_block = (512,  )
        bytes_per_thread = 8
        blocks_per_grid = (int(np.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        if not luts.is_cuda:
            target_device = torch.device("cuda")
            luts = luts.to(target_device)
            encoded_exponent = encoded_exponent.to(target_device)
            sign_mantissa = sign_mantissa.to(target_device)
            output_positions = output_positions.to(target_device)
            gaps = gaps.to(target_device)
        device = luts.device
        output = torch.empty(n_elements, dtype = torch.bfloat16, device=device)
        decode_kernel = get_decode_kernel()
        import cupy as cp
        torch_stream = torch.cuda.current_stream(device)
        with cp.cuda.Device(device.index):
            with cp.cuda.ExternalStream(torch_stream.cuda_stream):
                decode_kernel(
                    grid=blocks_per_grid,
                    block=threads_per_block,
                    shared_mem=shared_mem_size,
                    args=[
                        luts.data_ptr(),
                        encoded_exponent.data_ptr(),
                        sign_mantissa.data_ptr(),
                        output_positions.data_ptr(),
                        gaps.data_ptr(),
                        output.data_ptr(),
                        n_luts,
                        n_bytes,
                        n_elements,
                    ],
                )

        pieces = torch.tensor_split(output, split_positions.tolist())
        return {
            "mlp.down_proj": pieces[0].reshape(self.df11_weight_shapes["mlp.down_proj"]),
            "mlp.gate_proj": pieces[1].reshape(self.df11_weight_shapes["mlp.gate_proj"]),
            "mlp.up_proj": pieces[2].reshape(self.df11_weight_shapes["mlp.up_proj"]),
            "self_attn.k_proj": pieces[3].reshape(self.df11_weight_shapes["self_attn.k_proj"]),
            "self_attn.o_proj": pieces[4].reshape(self.df11_weight_shapes["self_attn.o_proj"]),
            "self_attn.q_proj": pieces[5].reshape(self.df11_weight_shapes["self_attn.q_proj"]),
            "self_attn.v_proj": pieces[6].reshape(self.df11_weight_shapes["self_attn.v_proj"]),
        }

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoded = None
        # Testing
        if self.is_dfloat11:
            decoded = self.decode() #TODO
            down_w = decoded["mlp.down_proj"]
            gate_w = decoded["mlp.gate_proj"]
            up_w = decoded["mlp.up_proj"]
            k_w = decoded["self_attn.k_proj"]
            o_w = decoded["self_attn.o_proj"]
            q_w = decoded["self_attn.q_proj"]
            v_w = decoded["self_attn.v_proj"]

            # DFloat11 stores and decodes the original, unsharded weights.
            # Match QKVParallelLinear/MergedColumnParallelLinear/
            # RowParallelLinear by selecting this rank's shard before GEMM.
            attn_rank = self.self_attn.attn_tp_rank
            attn_size = self.self_attn.attn_tp_size
            q_shard_size = self.self_attn.q_size
            kv_shard_size = self.self_attn.kv_size

            q_w = q_w.narrow(0, attn_rank * q_shard_size, q_shard_size)

            total_kv_heads = self.self_attn.total_num_kv_heads
            if total_kv_heads >= attn_size:
                kv_shard_rank = attn_rank
            else:
                # QKVParallelLinear replicates KV heads when there are fewer
                # KV heads than TP ranks.
                kv_replicas = attn_size // total_kv_heads
                kv_shard_rank = attn_rank // kv_replicas
            kv_start = kv_shard_rank * kv_shard_size
            k_w = k_w.narrow(0, kv_start, kv_shard_size)
            v_w = v_w.narrow(0, kv_start, kv_shard_size)

            # o_proj is row-parallel: each rank consumes its local attention
            # heads, so shard the input (column) dimension of the weight.
            o_w = o_w.narrow(1, attn_rank * q_shard_size, q_shard_size)

            mlp_rank = get_tensor_model_parallel_rank()
            mlp_size = get_tensor_model_parallel_world_size()
            intermediate_size = gate_w.shape[0]
            if intermediate_size % mlp_size != 0:
                raise ValueError(
                    f"DFloat11 MLP intermediate size {intermediate_size} is not "
                    f"divisible by TP size {mlp_size}."
                )
            mlp_shard_size = intermediate_size // mlp_size
            mlp_start = mlp_rank * mlp_shard_size
            gate_w = gate_w.narrow(0, mlp_start, mlp_shard_size)
            up_w = up_w.narrow(0, mlp_start, mlp_shard_size)
            down_w = down_w.narrow(1, mlp_start, mlp_shard_size)
        else:
            down_w = gate_w = up_w = None
            k_w = o_w = q_w = v_w = None
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                df11_q_weight=q_w,
                df11_k_weight=k_w,
                df11_v_weight=v_w,
                df11_o_weight=o_w,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlp(
            hidden_states,
            df11_gate_weight=gate_w,
            df11_down_weight=down_w,
            df11_up_weight=up_w,
        )
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.is_dfloat11 = hasattr(config, "dfloat11_config")

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer
    DF11_KEYS = {
    "encoded_exponent",
    "sign_mantissa",
    "luts",
    "gaps",
    "output_positions",
    "split_positions",
    }

    def is_df11_tensor_name(self, name: str) -> bool:
        return name.split(".")[-1] in self.DF11_KEYS

    def attach_df11_buffer(self, layer, key: str, tensor: torch.Tensor):
        buffer_name = f"df11_{key}"
        if hasattr(layer, buffer_name):
            delattr(layer, buffer_name)  

        if key == "output_positions":
            output_positions_np = tensor.view(torch.uint32).numpy()
            threads_per_block = (512,  )
            shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
            layer.df11_shared_mem_size = shared_mem_size

        if key != "split_positions":
            tensor = tensor.to(torch.device("cuda"))
        layer.register_buffer(buffer_name, tensor, persistent=True)
        

    def dump_large_tensors(self, model, path="/tmp/df11_after_load.txt", threshold_mb=1):
        lines = []

        param_total = 0
        buffer_total = 0

        lines.append("=== PARAMETERS ===")
        for name, p in model.named_parameters():
            mb = p.numel() * p.element_size() / 1024**2
            param_total += p.numel() * p.element_size()
            if mb >= threshold_mb:
                lines.append(
                    f"{name} shape={tuple(p.shape)} dtype={p.dtype} "
                    f"device={p.device} MB={mb:.2f}"
                )

        lines.append("=== BUFFERS ===")
        for name, b in model.named_buffers():
            mb = b.numel() * b.element_size() / 1024**2
            buffer_total += b.numel() * b.element_size()
            if mb >= threshold_mb:
                lines.append(
                    f"{name} shape={tuple(b.shape)} dtype={b.dtype} "
                    f"device={b.device} MB={mb:.2f}"
                )

        lines.append(f"PARAM TOTAL MB: {param_total / 1024**2:.2f}")
        lines.append(f"BUFFER TOTAL MB: {buffer_total / 1024**2:.2f}")
        lines.append(f"TOTAL MB: {(param_total + buffer_total) / 1024**2:.2f}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        #self.dump_large_tensors(self.model, "/nfs/home/hhmoon/tmp/df11_before_load.txt")

        stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            # DFloat11 weight mapping
            if self.is_dfloat11 and self.is_df11_tensor_name(name):
                parts = name.split(".")
                layer_id = int(parts[2])
                key = parts[-1]

                layer = self.model.layers[layer_id]
                self.attach_df11_buffer(layer, key, loaded_weight)
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
        print(f"Used Memory after loading weights: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        #self.dump_large_tensors(self.model, "/nfs/home/hhmoon/tmp/df11_after_load.txt")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM
