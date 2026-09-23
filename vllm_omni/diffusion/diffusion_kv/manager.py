# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.utils import free_kv_blocks_in_physical_order
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVContextMetadata,
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVContext, DiffusionKVRequest


class DiffusionKVAdmissionError(ValueError):
    """A request-specific KV admission failure safe to return to the caller."""


class _ContextKVRequest:
    """Native ``KVCacheManager`` request surface for one shared context."""

    def __init__(self, public_request_id: str, context: DiffusionKVContext) -> None:
        self.request_id = f"{public_request_id}/diffusion-kv/context/{context.context_id}"
        self.num_tokens = context.num_tokens
        self.num_prompt_tokens = context.num_tokens
        self.num_computed_tokens = 0
        self.block_hashes = list(context.block_hashes)
        self.skip_reading_prefix_cache = not self.block_hashes
        self.status = RequestStatus.WAITING
        self.num_preemptions = 0
        self.num_in_flight_tokens = 0
        self.shared_prefix_boundary = 0
        self.kv_transfer_params: dict[str, object] | None = None
        self.prompt_token_ids: list[int] | None = None


class DiffusionKVCacheManager:
    """Atomic public-request facade over native vLLM KVCacheManager."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        *,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_rows_per_request: int,
        max_in_flight_tokens: int | None = None,
    ) -> None:
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        if kv_cache_config.num_blocks <= 0 or not kv_cache_config.kv_cache_groups:
            raise ValueError("Diffusion KVCacheConfig must contain a positive block pool and at least one group")
        if scheduler_block_size <= 0 or hash_block_size <= 0:
            raise ValueError("scheduler_block_size and hash_block_size must be positive")
        if type(max_rows_per_request) is not int or max_rows_per_request <= 0:
            raise ValueError(f"max_rows_per_request must be a positive integer, got {max_rows_per_request!r}")
        if max_in_flight_tokens is not None and max_in_flight_tokens <= 0:
            raise ValueError(f"max_in_flight_tokens must be positive when provided, got {max_in_flight_tokens}")
        self.native_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            max_in_flight_tokens=max_in_flight_tokens,
            enable_caching=False,
        )
        self.max_model_len = max_model_len
        self.max_rows_per_request = max_rows_per_request
        # Native vLLM may reserve a null block, so an idle BlockPool does not
        # necessarily report ``kv_cache_config.num_blocks`` free blocks.
        self._empty_pool_num_free_blocks = self.native_manager.block_pool.get_num_free_blocks()
        # Keep sequence requests separate from context-only native shims.
        # Sequence policies such as prefix publication and KV transfer must
        # never interpret a context allocation as an execution sequence.
        self._requests: dict[str, tuple[DiffusionKVRequest, ...]] = {}
        self._context_requests: dict[str, tuple[_ContextKVRequest, ...]] = {}
        self._metadata: dict[str, DiffusionKVMetadata] = {}
        self._internal_request_ids: set[str] = set()
        self._next_allocation_generation = 1

    def _get_empty_pool_required_blocks(
        self,
        requests: Sequence[DiffusionKVRequest | _ContextKVRequest],
    ) -> int:
        """Return the native full-sequence reservation for one public request.

        vLLM performs this calculation inside ``allocate_slots`` when
        ``full_sequence_must_fit`` is enabled. Diffusion CFG adds one semantic
        requirement that native requests do not have: every execution sequence
        must fit atomically. Sum the native per-sequence result so a request
        that can never fit is rejected independently of current pool usage.
        """

        coordinator = self.native_manager.coordinator
        empty_blocks = self.native_manager.empty_kv_cache_blocks.blocks
        return sum(
            coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=request.num_tokens,
                new_computed_blocks=empty_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=0,
                num_local_computed_tokens=0,
                num_tokens_main_model=request.num_tokens,
                apply_admission_cap=True,
            )
            for request in requests
        )

    def has_request(self, public_request_id: str) -> bool:
        return public_request_id in self._requests

    @staticmethod
    def _collect_contexts(
        requests: Sequence[DiffusionKVRequest],
    ) -> tuple[tuple[DiffusionKVContext, ...], tuple[tuple[str, ...], ...]]:
        """Deduplicate request-scoped contexts and retain sequence references."""

        contexts_by_id: dict[str, DiffusionKVContext] = {}
        sequence_context_ids: list[tuple[str, ...]] = []
        for request in requests:
            context_ids: list[str] = []
            for context in request.kv_contexts:
                existing = contexts_by_id.get(context.context_id)
                if existing is not None and existing != context:
                    raise ValueError(
                        "Diffusion KV sequences define conflicting request-scoped contexts: "
                        f"context_id={context.context_id!r}"
                    )
                contexts_by_id.setdefault(context.context_id, context)
                context_ids.append(context.context_id)
            sequence_context_ids.append(tuple(context_ids))
        return tuple(contexts_by_id.values()), tuple(sequence_context_ids)

    def reserve_request(
        self,
        public_request_id: str,
        kv_requests: Sequence[DiffusionKVRequest],
    ) -> DiffusionKVMetadata | None:
        """Reserve all CFG sequences or roll back the complete public request."""

        if not public_request_id:
            raise ValueError("public_request_id must be non-empty")
        if public_request_id in self._requests:
            raise ValueError(f"Diffusion KV request {public_request_id!r} is already allocated")
        requests = tuple(kv_requests)
        if not requests:
            raise ValueError("Diffusion KV allocation requires at least one sequence")
        contexts, sequence_context_ids = self._collect_contexts(requests)
        num_rows = len(requests) + len(contexts)
        if num_rows > self.max_rows_per_request:
            raise DiffusionKVAdmissionError(
                f"Diffusion KV request {public_request_id!r} requires {num_rows} rows; "
                f"adapter limit is {self.max_rows_per_request}"
            )
        context_requests = tuple(_ContextKVRequest(public_request_id, context) for context in contexts)
        native_requests: tuple[DiffusionKVRequest | _ContextKVRequest, ...] = (*requests, *context_requests)

        sequence_ids = [request.sequence_id for request in requests]
        internal_ids = [request.request_id for request in native_requests]
        expected_sequence_ids = list(range(len(requests)))
        if sequence_ids != expected_sequence_ids:
            raise ValueError(
                "Diffusion KV requests must be ordered by contiguous sequence_id values: "
                f"expected={expected_sequence_ids}, got={sequence_ids}"
            )
        if len(internal_ids) != len(set(internal_ids)):
            raise ValueError(f"Diffusion KV internal request IDs must be unique, got {internal_ids}")
        conflicts = self._internal_request_ids.intersection(internal_ids)
        if conflicts:
            raise ValueError(f"Diffusion KV internal request IDs are already allocated: {sorted(conflicts)}")
        for request in requests:
            if request.seq_len > self.max_model_len:
                raise DiffusionKVAdmissionError(
                    f"Diffusion KV sequence {request.request_id!r} exceeds max_model_len: "
                    f"seq_len={request.seq_len}, max_model_len={self.max_model_len}"
                )
        for context in contexts:
            if context.num_tokens > self.max_model_len:
                raise DiffusionKVAdmissionError(
                    f"Diffusion KV context {context.context_id!r} exceeds max_model_len: "
                    f"num_tokens={context.num_tokens}, max_model_len={self.max_model_len}"
                )

        required_blocks = self._get_empty_pool_required_blocks(native_requests)
        if required_blocks > self._empty_pool_num_free_blocks:
            raise DiffusionKVAdmissionError(
                f"Diffusion KV request {public_request_id!r} cannot fit even when the block pool is empty: "
                f"required_blocks={required_blocks}, available_blocks={self._empty_pool_num_free_blocks}; "
                "increase KV cache capacity or reduce the request sequence count/length"
            )

        allocated: list[DiffusionKVRequest | _ContextKVRequest] = []
        sequence_metadata: list[DiffusionKVSequenceMetadata] = []
        context_metadata: list[DiffusionKVContextMetadata] = []
        try:
            for request, context_ids in zip(requests, sequence_context_ids, strict=True):
                blocks = self.native_manager.allocate_slots(
                    request,
                    num_new_tokens=request.seq_len,
                    delay_cache_blocks=True,
                    full_sequence_must_fit=True,
                )
                if blocks is None:
                    self._rollback(allocated)
                    allocated.clear()
                    return None
                allocated.append(request)
                sequence_metadata.append(
                    DiffusionKVSequenceMetadata(
                        sequence_id=request.sequence_id,
                        prefix_len=request.prefix_len,
                        target_len=request.target_len,
                        seq_len=request.seq_len,
                        block_ids=blocks.get_block_ids(),
                        context_ids=context_ids,
                    )
                )
            for context, context_request in zip(contexts, context_requests, strict=True):
                blocks = self.native_manager.allocate_slots(
                    context_request,
                    num_new_tokens=context_request.num_tokens,
                    delay_cache_blocks=True,
                    full_sequence_must_fit=True,
                )
                if blocks is None:
                    self._rollback(allocated)
                    allocated.clear()
                    return None
                allocated.append(context_request)
                context_metadata.append(
                    DiffusionKVContextMetadata(
                        context_id=context.context_id,
                        cache_role=context.cache_role,
                        num_tokens=context.num_tokens,
                        block_ids=blocks.get_block_ids(),
                    )
                )
        except Exception:
            self._rollback(allocated)
            raise

        metadata = DiffusionKVMetadata(
            request_id=public_request_id,
            allocation_generation=self._next_allocation_generation,
            sequences=tuple(sequence_metadata),
            contexts=tuple(context_metadata),
        )
        self._next_allocation_generation += 1
        self._requests[public_request_id] = requests
        self._context_requests[public_request_id] = context_requests
        self._metadata[public_request_id] = metadata
        self._internal_request_ids.update(internal_ids)
        return metadata

    def get_metadata(self, public_request_id: str) -> DiffusionKVMetadata:
        try:
            return self._metadata[public_request_id]
        except KeyError as exc:
            raise KeyError(f"Diffusion KV request {public_request_id!r} is not allocated") from exc

    def free_request(self, public_request_id: str) -> None:
        requests = self._requests.pop(public_request_id, ())
        context_requests = self._context_requests.pop(public_request_id, ())
        self._metadata.pop(public_request_id, None)
        for context_request in reversed(context_requests):
            self.native_manager.free(context_request)
            self._internal_request_ids.discard(context_request.request_id)
        for request in reversed(requests):
            free_kv_blocks_in_physical_order(self.native_manager, request)
            self._internal_request_ids.discard(request.request_id)

    def close(self) -> None:
        for public_request_id in tuple(self._requests):
            self.free_request(public_request_id)

    def _rollback(self, requests: Sequence[DiffusionKVRequest | _ContextKVRequest]) -> None:
        for request in reversed(requests):
            free_kv_blocks_in_physical_order(self.native_manager, request)
