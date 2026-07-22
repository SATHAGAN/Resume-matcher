"""
Wrapper around NVIDIA NIM (https://integrate.api.nvidia.com/v1).

NIM's chat/completions endpoint is OpenAI-compatible, so we just POST to it
with `requests` instead of pulling in the openai SDK. The embeddings endpoint
is *mostly* OpenAI-compatible but adds two NVIDIA-specific fields:

  - input_type: "query" | "passage"   (this model is asymmetric - queries and
                                        the documents they search over are
                                        embedded slightly differently)
  - truncate:   "NONE" | "START" | "END"

Get an API key (starts with "nvapi-") at https://build.nvidia.com
"""
import json
import re
import requests
from flask import current_app


class NIMError(RuntimeError):
    """Raised for any non-2xx response or unparsable payload from NIM."""


def _headers():
    api_key = current_app.config["NVIDIA_API_KEY"]
    if not api_key:
        raise NIMError(
            "NVIDIA_API_KEY is not set. Add it to your .env file "
            "(get a free key at https://build.nvidia.com)."
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def chat_completion(messages, model=None, temperature=0.2, max_tokens=2048, json_mode=True, timeout=60):
    """
    Calls NIM's /chat/completions. Returns the raw string content of the
    first choice. Caller is responsible for JSON-parsing if json_mode=True
    (use extract_json below - some NIM-hosted models ignore response_format
    and wrap output in markdown fences anyway).
    """
    base_url = "https://integrate.api.nvidia.com/v1"
    payload = {
        "model": model or current_app.config["NVIDIA_CHAT_MODEL"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=_headers(),
        data=json.dumps(payload),
        timeout=timeout,
    )
    if not resp.ok:
        raise NIMError(f"NIM chat completion failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise NIMError(f"Unexpected NIM chat response shape: {data}") from exc


def get_embedding(text, input_type="passage", timeout=30):
    """
    Calls NIM's /embeddings with nvidia/nv-embedqa-e5-v5 (1024-dim, asymmetric).
    input_type must be "query" (for the JD, when it's used to search resumes)
    or "passage" (for resumes, which are the documents being searched over).
    """
    base_url = current_app.config["NVIDIA_BASE_URL"]
    payload = {
        "model": current_app.config["NVIDIA_EMBED_MODEL"],
        "input": [text[:8000]],  # guard against pathologically long inputs
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "END",
    }
    resp = requests.post(
        f"{base_url}/embeddings",
        headers=_headers(),
        data=json.dumps(payload),
        timeout=timeout,
    )
    if not resp.ok:
        raise NIMError(f"NIM embedding call failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError) as exc:
        raise NIMError(f"Unexpected NIM embedding response shape: {data}") from exc


def run_structured_extraction(document_text, schema_model, task_description, system_preamble, max_retries=1):
    """
    Sends document_text to the chat model, forcing its response to validate
    against schema_model (a Pydantic BaseModel class). The model is given the
    JSON Schema up front; if its response fails to parse/validate, we feed
    the error back and give it one more try before raising.
    """
    from pydantic import ValidationError  # local import: keeps this module dependency-light for callers that don't need it

    schema_json = json.dumps(schema_model.model_json_schema(), indent=2)
    system_prompt = (
        f"{system_preamble}\n\n"
        f"Task: {task_description}\n\n"
        f"Your JSON object must validate against this JSON Schema:\n{schema_json}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": document_text[:20000]},
    ]

    last_error = None
    for _ in range(max_retries + 1):
        raw = chat_completion(messages, temperature=0.1)
        try:
            parsed = extract_json(raw)
            return schema_model.model_validate(parsed)
        except (ValidationError, NIMError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That response did not validate against the schema. Error: {exc}\nReturn a corrected JSON object only.",
            })

    raise last_error


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(raw_text):
    """
    Best-effort extraction of a JSON object from an LLM response. Some
    NIM-hosted open-weight models don't perfectly honor response_format and
    wrap output in ```json fences or add a stray sentence before/after.
    """
    raw_text = raw_text.strip()

    # Try straight parse first.
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try stripping a markdown code fence.
    fence_match = _JSON_FENCE_RE.search(raw_text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: grab the outermost {...} span.
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise NIMError(f"Could not parse JSON from model response: {raw_text[:500]}")
