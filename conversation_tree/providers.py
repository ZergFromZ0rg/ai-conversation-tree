"""Where replies come from.

The conversation logic needs a reply for each turn; this module is the only
place that knows how one is produced. A host plugs in its own model layer with
``init(..., replier=...)`` (see ``Replier``); the default, ``BuiltinReplier``,
is the environment-driven Ollama / OpenAI / Anthropic / Gemini / stub routing
the bundled viewer uses.
"""

import functools
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import httpx

from .conversationErrors import InvalidInputError, ProviderError

STUB_MODEL_SPEC = "stub"


def _chatMessages(userText: str, history) -> list[dict]:
    """Earlier turns plus the new message, in the {"role", "content"} shape Ollama and Anthropic take."""
    return [
        *({"role": message["role"], "content": message["content"]} for message in history),
        {"role": "user", "content": userText},
    ]


def _openAiInput(userText: str, history) -> list[dict]:
    """The Responses API `input`: earlier turns as plain-text messages, then the new one."""
    return [
        *({"role": message["role"], "content": message["content"]} for message in history),
        {"role": "user", "content": [{"type": "input_text", "text": userText}]},
    ]


def _geminiContents(userText: str, history) -> list[dict]:
    """Gemini calls the assistant role "model"."""
    return [
        *(
            {"role": "model" if message["role"] == "assistant" else "user", "parts": [{"text": message["content"]}]}
            for message in history
        ),
        {"role": "user", "parts": [{"text": userText}]},
    ]

MODEL_SPEC_HINT = "model must be 'stub', 'ollama:<name>', 'openai:<name>', 'anthropic:<name>', or 'gemini:<name>'."


def _translatesTransportErrors(function):
    """Raise ProviderError for any httpx failure inside `function`, so callers
    never need to know which HTTP client the providers use. Works for plain
    functions and for generators (errors that surface mid-stream)."""
    if inspect.isgeneratorfunction(function):

        @functools.wraps(function)
        def streaming(*args, **kwargs):
            try:
                yield from function(*args, **kwargs)
            except httpx.HTTPError as error:
                raise ProviderError(str(error)) from error

        return streaming

    @functools.wraps(function)
    def blocking(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except httpx.HTTPError as error:
            raise ProviderError(str(error)) from error

    return blocking


def extractResponseText(responseBody: dict) -> str:
    outputItems = responseBody.get("output", [])
    textParts: list[str] = []

    for outputItem in outputItems:
        for contentItem in outputItem.get("content", []):
            if contentItem.get("type") == "output_text":
                textParts.append(contentItem.get("text", ""))

    return "\n".join(part for part in textParts if part).strip()


def extractOllamaText(responseBody: dict) -> str:
    message = responseBody.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return ""


KNOWN_PROVIDERS = ("ollama", "openai", "anthropic", "gemini", "stub")


def parseModelSpec(modelSpec: str | None) -> tuple[str | None, str | None]:
    """Split "provider:model" into (provider, model).

    Ollama tags contain colons ("llama3:8b"), so only the first colon is a
    separator. "stub" and a bare provider yield model=None. A falsy spec yields
    (None, None) -> fall back to the environment-configured provider.
    """
    if not modelSpec:
        return None, None
    provider, _, model = modelSpec.partition(":")
    return provider, (model or None)


def ollamaBaseUrl() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def generateOllamaText(userText: str, model: str, history=()) -> str:
    response = httpx.post(
        f"{ollamaBaseUrl()}/api/chat",
        json={
            "model": model,
            "messages": _chatMessages(userText, history),
            "stream": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    aiText = extractOllamaText(response.json())
    if not aiText:
        raise ProviderError("Ollama returned no text output.")
    return aiText


def generateOpenAiText(userText: str, model: str, apiKey: str | None = None, history=()) -> str:
    apiKey = apiKey or os.environ.get("OPENAI_API_KEY")
    if not apiKey:
        raise ProviderError("OPENAI_API_KEY is not set.")
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": _openAiInput(userText, history),
        },
        timeout=60.0,
    )
    response.raise_for_status()
    aiText = extractResponseText(response.json())
    if not aiText:
        raise ProviderError("LLM returned no text output.")
    return aiText


def generateOpenAiTextStream(userText: str, model: str, apiKey: str | None = None, history=()):
    """Yield text deltas from the OpenAI Responses SSE stream."""
    apiKey = apiKey or os.environ.get("OPENAI_API_KEY")
    if not apiKey:
        raise ProviderError("OPENAI_API_KEY is not set.")
    with httpx.stream(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json={
            "model": model,
            "input": _openAiInput(userText, history),
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("type") == "response.output_text.delta" and event.get("delta"):
                yield event["delta"]
            if event.get("type") == "response.completed":
                break


anthropicApiVersion = "2023-06-01"


def extractAnthropicText(responseBody: dict) -> str:
    blocks = responseBody.get("content", [])
    return "\n".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    ).strip()


def generateAnthropicText(userText: str, model: str, apiKey: str | None = None, history=()) -> str:
    apiKey = apiKey or os.environ.get("ANTHROPIC_API_KEY")
    if not apiKey:
        raise ProviderError("ANTHROPIC_API_KEY is not set.")
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": apiKey,
            "anthropic-version": anthropicApiVersion,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "messages": _chatMessages(userText, history),
        },
        timeout=60.0,
    )
    response.raise_for_status()
    aiText = extractAnthropicText(response.json())
    if not aiText:
        raise ProviderError("LLM returned no text output.")
    return aiText


def extractGeminiText(responseBody: dict) -> str:
    candidates = responseBody.get("candidates", [])
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts).strip()


def generateGeminiText(userText: str, model: str, apiKey: str | None = None, history=()) -> str:
    apiKey = apiKey or os.environ.get("GEMINI_API_KEY")
    if not apiKey:
        raise ProviderError("GEMINI_API_KEY is not set.")
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": apiKey},
        json={"contents": _geminiContents(userText, history)},
        timeout=60.0,
    )
    response.raise_for_status()
    aiText = extractGeminiText(response.json())
    if not aiText:
        raise ProviderError("LLM returned no text output.")
    return aiText


def envDefaultModelSpec() -> str:
    """The model spec that matches the current environment-based routing."""
    ollamaModelName = os.environ.get("OLLAMA_MODEL")
    if ollamaModelName:
        return f"ollama:{ollamaModelName}"
    if os.environ.get("AI_CONVERSATION_TREE_STUB_LLM", "1") == "1":
        return "stub"
    if os.environ.get("OPENAI_API_KEY"):
        return f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-5-mini')}"
    return "stub"


@_translatesTransportErrors
def generateAiText(userText: str, modelSpec: str | None = None, apiKey: str | None = None, history=()) -> str:
    """One reply. `history` is the earlier turns as [{"role", "content"}, ...] (the stub ignores it)."""
    provider, model = parseModelSpec(modelSpec or envDefaultModelSpec())

    if provider == "stub":
        return f"Stub LLM response for: {userText}"
    if provider == "ollama":
        if not model:
            raise ProviderError("An ollama model spec needs a model name, e.g. 'ollama:llama3'.")
        return generateOllamaText(userText, model, history)
    if provider == "openai":
        return generateOpenAiText(userText, model or os.environ.get("OPENAI_MODEL", "gpt-5-mini"), apiKey, history)
    if provider == "anthropic":
        if not model:
            raise ProviderError("An anthropic model spec needs a model name, e.g. 'anthropic:claude-sonnet-5'.")
        return generateAnthropicText(userText, model, apiKey, history)
    if provider == "gemini":
        if not model:
            raise ProviderError("A gemini model spec needs a model name, e.g. 'gemini:gemini-2.5-flash'.")
        return generateGeminiText(userText, model, apiKey, history)
    raise ProviderError(f"Unknown model provider: {provider!r}. Expected one of {list(KNOWN_PROVIDERS)}.")


# ---- streaming ----
# Splits a reply into incremental text chunks as it's generated, instead of
# waiting for the whole thing. Classification still needs the complete text
# (it embeds userText + " " + aiText), so streaming only changes what the
# composer shows while the reply is in flight — persistence happens the same
# way afterward, once the stream ends (see streamChatMessage).

stubStreamWordsPerChunk = 3
stubStreamDelaySeconds = 0.05


def _streamStubText(userText: str):
    words = f"Stub LLM response for: {userText}".split(" ")
    for index in range(0, len(words), stubStreamWordsPerChunk):
        chunk = " ".join(words[index : index + stubStreamWordsPerChunk])
        if index > 0:
            chunk = " " + chunk
        time.sleep(stubStreamDelaySeconds)  # otherwise a local stub reply streams instantly
        yield chunk


def generateOllamaTextStream(userText: str, model: str, history=()):
    with httpx.stream(
        "POST",
        f"{ollamaBaseUrl()}/api/chat",
        json={
            "model": model,
            "messages": _chatMessages(userText, history),
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            content = (chunk.get("message") or {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break


def generateAnthropicTextStream(userText: str, model: str, apiKey: str | None = None, history=()):
    apiKey = apiKey or os.environ.get("ANTHROPIC_API_KEY")
    if not apiKey:
        raise ProviderError("ANTHROPIC_API_KEY is not set.")
    with httpx.stream(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": apiKey,
            "anthropic-version": anthropicApiVersion,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "messages": _chatMessages(userText, history),
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:") :].strip())
            if event.get("type") == "content_block_delta":
                text = (event.get("delta") or {}).get("text", "")
                if text:
                    yield text


@_translatesTransportErrors
def streamAiText(userText: str, modelSpec: str | None = None, apiKey: str | None = None, history=()):
    """Yield a reply as it's generated. Same provider routing as generateAiText.

    Gemini's `streamGenerateContent` (a JSON array, not line-delimited) still
    falls back to the blocking call; OpenAI and Ollama use their native SSE or
    NDJSON streams.
    """
    provider, model = parseModelSpec(modelSpec or envDefaultModelSpec())

    if provider == "stub":
        yield from _streamStubText(userText)
        return
    if provider == "ollama":
        if not model:
            raise ProviderError("An ollama model spec needs a model name, e.g. 'ollama:llama3'.")
        yield from generateOllamaTextStream(userText, model, history)
        return
    if provider == "openai":
        yield from generateOpenAiTextStream(
            userText, model or os.environ.get("OPENAI_MODEL", "gpt-5-mini"), apiKey, history
        )
        return
    if provider == "anthropic":
        if not model:
            raise ProviderError("An anthropic model spec needs a model name, e.g. 'anthropic:claude-sonnet-5'.")
        yield from generateAnthropicTextStream(userText, model, apiKey, history)
        return
    if provider == "gemini":
        if not model:
            raise ProviderError("A gemini model spec needs a model name, e.g. 'gemini:gemini-2.5-flash'.")
        yield generateGeminiText(userText, model, apiKey, history)
        return
    raise ProviderError(f"Unknown model provider: {provider!r}. Expected one of {list(KNOWN_PROVIDERS)}.")


# A small curated set of models worth offering as one-click downloads —
# Ollama has no API for browsing its library, only for pulling a name you
# already know, so this is a hand-picked list rather than something fetched.
OLLAMA_LIBRARY = [
    {"name": "llama3.2:1b", "sizeLabel": "~1.3 GB"},
    {"name": "llama3.2:3b", "sizeLabel": "~2.0 GB"},
    {"name": "qwen2.5:1.5b", "sizeLabel": "~1.0 GB"},
    {"name": "qwen2.5:3b", "sizeLabel": "~1.9 GB"},
    {"name": "phi3:mini", "sizeLabel": "~2.3 GB"},
    {"name": "gemma2:2b", "sizeLabel": "~1.6 GB"},
    {"name": "mistral:7b", "sizeLabel": "~4.1 GB"},
]


def listOllamaLibrary() -> dict:
    installed = set()
    ollamaReachable = False
    try:
        response = httpx.get(f"{ollamaBaseUrl()}/api/tags", timeout=2.0)
        response.raise_for_status()
        ollamaReachable = True
        installed = {entry.get("name") for entry in response.json().get("models", [])}
    except Exception:
        pass

    return {
        "ollamaReachable": ollamaReachable,
        "models": [entry for entry in OLLAMA_LIBRARY if entry["name"] not in installed],
    }


@_translatesTransportErrors
def pullOllamaModel(model: str):
    """Yield Ollama's own pull progress lines as {"status", "completed", "total"}
    dicts, straight through — Ollama already reports exactly what a progress
    bar needs (bytes completed/total per layer) so there's nothing to compute,
    just forward each NDJSON line as it arrives."""
    with httpx.stream(
        "POST",
        f"{ollamaBaseUrl()}/api/pull",
        json={"model": model, "stream": True},
        timeout=None,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            yield {
                "status": chunk.get("status", ""),
                "completed": chunk.get("completed"),
                "total": chunk.get("total"),
            }
            if chunk.get("status") == "success":
                break


def listAvailableModels() -> dict:
    options: list[dict] = []

    ollamaReachable = False
    try:
        response = httpx.get(f"{ollamaBaseUrl()}/api/tags", timeout=2.0)
        response.raise_for_status()
        ollamaReachable = True
        for entry in sorted(response.json().get("models", []), key=lambda m: m.get("name", "")):
            name = entry.get("name")
            if name:
                options.append({"id": f"ollama:{name}", "label": name, "provider": "ollama"})
    except Exception:
        pass

    if os.environ.get("OPENAI_API_KEY"):
        openAiModel = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        options.append(
            {"id": f"openai:{openAiModel}", "label": f"OpenAI · {openAiModel}", "provider": "openai"}
        )

    stubEnabled = os.environ.get("AI_CONVERSATION_TREE_STUB_LLM", "1") == "1"
    if stubEnabled or not options:
        options.append({"id": "stub", "label": "Stub (canned replies)", "provider": "stub"})

    return {
        "models": options,
        "default": envDefaultModelSpec(),
        "ollamaReachable": ollamaReachable,
    }


def isValidModelSpec(modelSpec: str) -> bool:
    provider, model = parseModelSpec(modelSpec)
    if provider not in KNOWN_PROVIDERS:
        return False
    if provider in ("ollama", "openai", "anthropic", "gemini") and not model:
        return False
    return True


# ---- pluggable replies ----

@dataclass(frozen=True)
class ReplyRequest:
    """Everything a replier is told about the reply it is asked for.

    One object, so a field can be added later without changing every replier's
    signature.
    """

    userText: str
    """The new message to answer."""

    modelSpec: str | None = None
    """The model the request or the conversation chose; None means the replier's default."""

    apiKey: str | None = None
    """A per-request provider key, when the caller supplied one."""

    conversationId: int | None = None
    """The conversation this message belongs to, or None for a request that isn't
    a turn in one (a summary)."""

    history: list[dict] = field(default_factory=list)
    """The turns before this message, oldest first, as ``{"role": "user" | "assistant",
    "content": str}``. Only turns that fit the context window (see
    ``Settings.contextWindowTokens``) are included, and always whole turns, so the
    list alternates user / assistant and starts with a user message. Empty for the
    first turn, and for requests that aren't a turn in a conversation."""


class Replier:
    """How the package obtains a reply for a user message.

    A host with its own model layer subclasses this and overrides ``generate``;
    the other methods have working defaults. ``generate`` should raise
    ``ProviderError`` for failures the caller ought to see as "the model
    provider failed" (an HTTP layer reports those as 502); anything else is
    treated as a bug and propagates.

    The request carries the conversation's earlier turns (``request.history``)
    and its id (``request.conversationId``): use the history as given, or ignore
    it and look the conversation up in the host's own store.
    """

    modelSpecHint = "model must be a non-empty model name."

    def generate(self, request: ReplyRequest) -> str:
        raise NotImplementedError

    def stream(self, request: ReplyRequest) -> Iterator[str]:
        """Yield the reply in pieces. By default the whole reply is one piece."""
        yield self.generate(request)

    def isValidModelSpec(self, modelSpec: str) -> bool:
        return bool(modelSpec.strip())

    def defaultModelSpec(self) -> str | None:
        """The model used when neither the request nor the conversation names one."""
        return None


class BuiltinReplier(Replier):
    """The environment-driven providers: Ollama, OpenAI, Anthropic, Gemini, or a stub."""

    modelSpecHint = MODEL_SPEC_HINT

    def generate(self, request):
        return generateAiText(request.userText, request.modelSpec, request.apiKey, request.history)

    def stream(self, request):
        return streamAiText(request.userText, request.modelSpec, request.apiKey, request.history)

    def isValidModelSpec(self, modelSpec):
        return isValidModelSpec(modelSpec)

    def defaultModelSpec(self):
        return envDefaultModelSpec()


class CallableReplier(Replier):
    """Adapts a plain ``function(request) -> str``.

    Any exception the function raises is reported as a ProviderError, since a
    failing model call is a provider failure by definition.
    """

    def __init__(self, function: Callable[[ReplyRequest], str]):
        self._function = function

    def generate(self, request):
        try:
            return self._function(request)
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(str(error) or type(error).__name__) from error


_replier: Replier = BuiltinReplier()


def setReplier(replier: "Replier | Callable | None") -> Replier:
    """Install the replier the package uses. ``None`` restores the built-in one;
    a plain function is wrapped in a CallableReplier."""
    global _replier
    if replier is None:
        replier = BuiltinReplier()
    elif not isinstance(replier, Replier):
        if not callable(replier):
            raise TypeError("replier must be a Replier, a callable(ReplyRequest) -> str, or None")
        replier = CallableReplier(replier)
    _replier = replier
    return replier


def getReplier() -> Replier:
    return _replier


def requireValidModelSpec(model: str | None) -> None:
    """None means "use the default"; anything else must be a spec the replier accepts."""
    replier = getReplier()
    if model is not None and not replier.isValidModelSpec(model):
        raise InvalidInputError(replier.modelSpecHint)
