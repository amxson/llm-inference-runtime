"""Explicit lifecycle and isolated runtime state for one inference request."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class RequestState:
    prompt: str
    max_new_tokens: int
    request_id: str = field(default_factory=lambda: uuid4().hex)
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    runtime_inputs: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    kv_cache: Any = field(default=None, repr=False, compare=False)
    # Logical allocator ownership only; these IDs never index HF cache tensors.
    kv_block_ids: list[int] = field(default_factory=list, init=False)
    status: RequestStatus = field(default=RequestStatus.WAITING, init=False)
    stop_reason: str | None = field(default=None, init=False)
    model_execution_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0 or not self.prompt.strip() or not self.request_id:
            raise ValueError("A request needs a nonempty prompt/ID and max_new_tokens >= 0.")

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_token_ids)

    @property
    def logical_context_length(self) -> int:
        return len(self.prompt_token_ids) + self.generated_token_count

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED

    def start_prefill(self, token_ids: list[int], runtime_inputs=None) -> None:
        if self.status is not RequestStatus.WAITING:
            raise ValueError("Only a waiting request can start prefill.")
        if not token_ids:
            raise ValueError("The prompt must contain at least one token.")
        self.prompt_token_ids = list(token_ids)
        self.runtime_inputs = {} if runtime_inputs is None else dict(runtime_inputs)
        self.status = RequestStatus.PREFILL
        if self.max_new_tokens == 0:
            self.finish("max_new_tokens")

    def record_token(self, token_id: int, eos_token_id: int | None = None) -> None:
        """Record one model execution's selected token and apply stopping rules."""
        if self.status not in {RequestStatus.PREFILL, RequestStatus.DECODING}:
            raise ValueError("Tokens can only be recorded during prefill or decoding.")
        self.generated_token_ids.append(int(token_id))
        self.model_execution_count += 1
        if eos_token_id is not None and token_id == eos_token_id:
            self.finish("eos")
        elif self.generated_token_count >= self.max_new_tokens:
            self.finish("max_new_tokens")
        else:
            self.status = RequestStatus.DECODING

    def finish(self, reason: str) -> None:
        if self.is_finished:
            raise ValueError("A finished request cannot transition again.")
        if reason not in {"eos", "max_new_tokens", "cancelled", "error"}:
            raise ValueError("Unknown stop reason.")
        if self.status is RequestStatus.WAITING and reason not in {"cancelled", "error"}:
            raise ValueError("A waiting request must start prefill before completing.")
        self.stop_reason = reason
        self.status = RequestStatus.FINISHED
        self.kv_cache = None
        self.kv_block_ids.clear()
        self.runtime_inputs.clear()

    def to_result(self, tokenizer) -> dict[str, Any]:
        """Export small inspectable data, never runtime tensors or cache state."""
        full_ids = self.prompt_token_ids + self.generated_token_ids
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "full_token_ids": full_ids,
            "generated_text": tokenizer.decode(self.generated_token_ids, skip_special_tokens=True),
            "full_text": tokenizer.decode(full_ids, skip_special_tokens=True),
            "generated_token_count": self.generated_token_count,
            "model_execution_count": self.model_execution_count,
            "stop_reason": self.stop_reason,
        }
