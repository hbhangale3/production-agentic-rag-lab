# Week 6 M06: Langfuse observability foundation

This milestone adds the provider-neutral foundation for explicit RAG tracing.
It does not create production traces yet; cache, retrieval, evidence,
generation, and validation instrumentation belong to M07.

## Configuration

Langfuse is disabled by default and requires no account or network access:

```text
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_CAPTURE_CONTENT=false
LANGFUSE_TIMEOUT_SECONDS=5
```

When enabled, both keys must be nonblank. The secret key uses `SecretStr` and
is redacted from settings representations. Invalid local configuration fails
fast. Unexpected SDK construction failure instead produces an unavailable
no-op provider so RAG can continue. `LANGFUSE_HOST` is passed as the SDK's
`base_url`, supporting Langfuse Cloud and self-hosted installations without
adding a Compose service.

The resolved official SDK is Langfuse 4.17.0 for Python 3.12. The adapter uses
its current `start_observation`, batched `flush`, and `shutdown` APIs. The SDK
owns its batching/export worker for each enabled application worker; disabled
mode does not construct the SDK or start its resources. SDK defaults are used
for batching. The configured SDK HTTP timeout is bounded to 1–30 seconds, and
application flush/shutdown calls run outside the event loop with the same
upper bound.

## Provider-neutral architecture

Application code depends on `ObservabilityProvider` and `Observation`, not
Langfuse SDK types. The deliberately small interface supports a root trace,
nested spans, metadata updates, safe error classification, completion, flush,
and close. `NoOpObservabilityProvider` makes all calls deterministic and
harmless when disabled. `LangfuseObservabilityProvider` catches SDK operation
and lifecycle failures so loss of telemetry never becomes loss of RAG.

The provider is created once per FastAPI worker, stored at
`app.state.observability`, flushed and then closed at shutdown. Application-
created SDK clients are shut down; explicitly injected caller-owned clients are
not.

Internal status means:

- `disabled`: observability was intentionally disabled.
- `configured`: a local Langfuse client was constructed; this does not claim
  remote service reachability.
- `unavailable`: construction failed or the owned adapter has shut down.

No remote call is added to `/health`; public observability health exposure is
deferred. Langfuse network or export failure is fail-open.

## Privacy policy

Raw content capture defaults to disabled. M06 sends no user questions, prompts,
evidence, answers, source chunks, cache payloads, user/session identifiers, or
arbitrary exception strings. The abstraction accepts structured metadata and a
sanitized error type/classification only. It does not expose general raw
input/output parameters. M07 may attach non-content metadata such as model,
tokens, retrieval mode, cache outcome, source count, duration, and safe error
classification. Enabling future content capture must remain an explicit policy
decision through `LANGFUSE_CAPTURE_CONTENT`.

No automatic FastAPI, HTTP, Groq, or environment instrumentation is enabled.
The existing provider-neutral Groq implementation, cache behavior, cache
identity, response schemas, SSE protocol, and Gradio UI are unchanged. This
milestone adds no trace IDs, trace URLs, evaluation, scoring, prompt management,
sampling policy, or custom cost calculation.
