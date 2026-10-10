"""The interface every engine (fake, HF baseline, ours, vLLM adapter) implements for the runner."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None:
        """Queue a request. It must produce exactly output_len tokens (EOS is ignored)."""
        ...

    def step(self) -> list[tuple[int, int, bool]]:
        """Run one forward pass worth of work and return (req_id, token_id, finished) per sampled token.

        Returns only after the sampled tokens are on the CPU, so the runner can timestamp on return.
        Returns [] when there is nothing to do.
        """
        ...

    def has_unfinished(self) -> bool:
        """True while any added request has not produced its last token."""
        ...
