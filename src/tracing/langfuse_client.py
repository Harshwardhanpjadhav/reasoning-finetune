"""
Thin wrapper around Langfuse so eval/inference code just calls trace_generation()
without knowing Langfuse's specific API. If you ever swap observability tools,
only this file changes.

Requires env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
(set these on the Lightning Studio, not committed to the repo).
"""

import os
import time
from contextlib import contextmanager


def _client():
    from langfuse import Langfuse
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


@contextmanager
def trace_generation(name: str, input_text: str, metadata: dict | None = None):
    """
    Usage:
        with trace_generation("gsm8k_eval", question) as trace:
            output = model.generate(...)
            trace.set_output(output)

    If Langfuse env vars aren't set, this silently no-ops rather than crashing
    the eval run — tracing is observability, it shouldn't block the pipeline.
    """
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        yield _NullTrace()
        return

    client = _client()
    start = time.time()
    generation = client.generation(name=name, input=input_text, metadata=metadata or {})
    wrapper = _TraceWrapper(generation, start)
    try:
        yield wrapper
    finally:
        wrapper.finalize()


class _TraceWrapper:
    def __init__(self, generation, start_time):
        self._generation = generation
        self._start = start_time
        self._output = None

    def set_output(self, output: str):
        self._output = output

    def finalize(self):
        latency = time.time() - self._start
        self._generation.end(output=self._output, metadata={"latency_sec": latency})


class _NullTrace:
    """No-op used when Langfuse isn't configured, so code doesn't need if/else checks."""
    def set_output(self, output: str):
        pass
