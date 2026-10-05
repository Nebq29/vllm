# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thor (SM110) sparse-MLA backend for GLM-5.3 Flash.

GLM-5.3's sparse-MLA layers are rope-free BF16 (``qk_rope_head_dim == 0``,
``head_size == kv_lora_rank == 512``). On sm_110 none of the precompiled
sparse-MLA libraries (FlashMLA, FlashAttention-3, FlashInfer TRTLLM-gen)
ship sm_110 code, so the backend selector rejects every CUDA/ROCm-adjacent
option.

The one portable route is the rope-free BF16 **Triton** path that already
lives in :mod:`vllm.v1.attention.ops.rocm_aiter_mla_sparse`
(``_rocm_sparse_attn_prefill_ragged_triton`` /
``_rocm_sparse_attn_decode_ragged_triton``). Those kernels are pure Triton
and are already imported and run on CUDA by our DeepSeek-V4 Thor backend
(``vllm/models/deepseek_v4/nvidia/thor.py``).

This module wires that same Triton path to GLM by subclassing the ROCm
AITER sparse backend. The Triton path in ``ROCMAiterMLASparseImpl`` is
CUDA-safe end-to-end; the ONLY AMD dependency it would otherwise hit is the
unconditional ``from aiter import dtypes, get_mla_metadata_info_v1`` in
``ROCMAiterMLASparseMetadataBuilder.__init__``, which exists solely to size
the *persistent* work-splitting metadata buffers. The Triton path never
reads those buffers (``build()`` skips them whenever
``use_triton_sparse`` is true), so this builder subclass simply skips that
block and forces ``_use_persistent_metadata = False``.

No shared ROCm code is modified: the base classes keep their exact
behaviour on ROCm; only this sm_110 subclass changes the allocation path.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadata,
    ROCMAiterMLASparseMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class ThorMLASparseGLMBackend(ROCMAiterMLASparseBackend):
    """Sparse-MLA backend for GLM-5.3 on Thor (SM110), via the Triton path.

    Accepts: compute capability major == 11, MLA + sparse, head_size ==
    kv_lora_rank (512), KV cache ``auto``/``bfloat16``, model dtype bf16.
    Rejects: fp8 KV and rope-carrying models (see validate_configuration).
    """

    # BF16 only on this path (the Triton rope-free kernels are bf16/fp16;
    # we restrict to bf16 to match GLM-5.3 and avoid fp8 scale plumbing).
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "THOR_MLA_SPARSE_GLM"

    @staticmethod
    def get_builder_cls() -> type["ThorMLASparseGLMMetadataBuilder"]:
        return ThorMLASparseGLMMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[ROCMAiterMLASparseImpl]:
        # The base impl's Triton branch is CUDA-safe; reuse it unchanged.
        return ROCMAiterMLASparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 11

    @classmethod
    def supports_sink(cls) -> bool:
        # The Triton rope-free path we force does not carry attention sinks
        # for GLM (GLM-5.3 has none). Reject sinks explicitly so a
        # sink-carrying model never silently lands here.
        return False

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Same as the ROCm base: the Triton kernels index by strides, but
        # the global-index conversion assumes contiguous pages per layer.
        return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC)

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        *args,
        **kwargs,
    ) -> list[str]:
        """Add GLM-specific accept/reject reasons on top of the base checks.

        The base ``validate_configuration`` already checks dtype / kv_cache
        dtype / mla / sparse / compute-capability. We layer on the
        rope-free + head==kv_lora_rank requirement with explicit reason
        strings, per the round spec.
        """
        invalid_reasons = super().validate_configuration(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            *args,
            **kwargs,
        )
        # rope-free only: head_size must equal kv_lora_rank (512). A
        # rope-carrying model has head_size = kv_lora_rank + rope_dim > 512.
        # We read kv_lora_rank from the current config when available.
        try:
            from vllm.config import get_current_vllm_config

            kv_lora_rank = get_current_vllm_config().model_config.hf_text_config.kv_lora_rank
        except Exception:
            kv_lora_rank = 512
        if head_size != kv_lora_rank:
            invalid_reasons.append(
                f"rope-carrying model: head_size {head_size} != kv_lora_rank "
                f"{kv_lora_rank} (this backend is rope-free BF16 only)"
            )
        if kv_cache_dtype is not None and str(kv_cache_dtype).startswith("fp8"):
            invalid_reasons.append(
                f"fp8 KV cache ({kv_cache_dtype}) not supported; use bfloat16"
            )
        return invalid_reasons


class ThorMLASparseGLMMetadataBuilder(ROCMAiterMLASparseMetadataBuilder):
    """Builder that skips the aiter persistent-metadata init.

    The base builder unconditionally does
    ``from aiter import dtypes, get_mla_metadata_info_v1`` to size the
    persistent work-splitting buffers. Those buffers are only consumed by
    the AITER non-Triton decode path; the Triton rope-free path we force
    never reads them. This subclass reproduces the CUDA-safe portion of the
    base ``__init__`` and omits the aiter block, setting
    ``_use_persistent_metadata = False`` so ``build()`` skips persistent
    metadata entirely.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        # --- CUDA-safe portion (mirrors the base builder up to the aiter
        # import). Kept in sync with
        # ROCMAiterMLASparseMetadataBuilder.__init__ minus the aiter block.
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.model_dtype = vllm_config.model_config.dtype
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        parallel_config = vllm_config.parallel_config
        self.device = device
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.vllm_config = vllm_config
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        from vllm.model_executor.layers.attention.mla_attention import get_mla_dims

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_text_config.index_topk

        # Force the non-persistent path: the Triton kernels need no
        # persistent work-splitting metadata.
        self._use_persistent_metadata = False

        # num_compute_units is used only by _sparse_decode_max_split (aiter
        # path); harmless to set here.
        self._num_compute_units = 20  # Thor has 20 SMs; not used on Triton path
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=device
        )

        self.req_id_per_token_buffer = torch.zeros(
            (max_num_batched_tokens,), dtype=torch.int32, device=device
        )
        self.qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.paged_kv_indices = torch.zeros(
            [max_num_batched_tokens * self.topk_tokens],
            dtype=torch.int32,
            device=device,
        )
        self.paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )

        # --- aiter persistent-metadata buffers: NOT allocated (Triton path
        # never reads them). Set to None so any accidental access is loud.
        self._num_attention_heads = self.num_heads
        self._mla_work_meta_data = None
        self._mla_work_indptr = None
        self._mla_work_info_set = None
        self._mla_reduce_indptr = None
        self._mla_reduce_final_map = None
        self._mla_reduce_partial_map = None

        self._prev_req_extent: int = 0
        self._prev_indices_extent: int = 0
        self._prev_metadata_key: tuple | None = None
