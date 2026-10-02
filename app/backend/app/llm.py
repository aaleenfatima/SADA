"""
Natural-language report generation.

Design rule: the LLM is given ONLY structured, measured values and is told it
may not introduce any fact not present in them. Left unconstrained, a language
model will happily invent a planet radius, an equilibrium temperature, or a
stellar type that nothing in this pipeline measured — which would be fabricated
data presented as a result.

If no API key is configured, a deterministic template is used instead. The
template is not a degraded fallback to apologise for: it never hallucinates, and
the demo works offline.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger("sada.llm")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("SADA_LLM_MODEL", "claude-sonnet-4-6")

OLLAMA_HOST = os.environ.get("SADA_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("SADA_OLLAMA_MODEL")  # e.g. "qwen2.5:7b-instruct"

# Explicit override ("anthropic" | "ollama" | "template"), or auto-resolved from
# whatever is configured. Anthropic wins if both are set, since it's the one
# whose constraint-following on this prompt has actually been checked.
LLM_PROVIDER = os.environ.get("SADA_LLM_PROVIDER")

SYSTEM_PROMPT = """You write short vetting summaries for an exoplanet transit \
classifier.

Rules, without exception:
- Use ONLY the numbers and flags in the JSON provided. Introduce no other facts.
- Never estimate or state planet radius, mass, temperature, habitability, \
stellar type, or distance. None of these are measured by this pipeline.
- Never convert a model probability into a claim of discovery. This is a \
screening tool, not a confirmation.
- If the decision is "escalate", state plainly why human review is needed.
- If odd/even depths differ significantly, say that this is characteristic of \
an eclipsing binary rather than a planet.
- If rf_probability is null, check rf_unavailable_reason in the JSON and state \
that specific reason. Do not assume or guess why the Random Forest model was \
not used -- the JSON always tells you if a reason is known.
- 3-5 sentences, plain prose, no headings, no bullet points.
- Never state the same fact twice in different wording — e.g. if you explain
  that the models disagree, do not separately restate that as the reason for
  escalation; say it once.
- Do not report the CNN probability as a calibrated percentage; refer to the \
confidence band instead. This model is known to be overconfident mid-range."""


def generate_report(candidate: dict, star_id: int, rf_unavailable_reason=None) -> dict:
    """Returns {'text': str, 'source': 'llm'|'template'}.

    `rf_unavailable_reason` comes from the agent's top-level result
    (`analyze_star`'s `rf_unavailable_reason`), not from the candidate itself.
    Without it, this function cannot tell "no catalog ephemeris existed" apart
    from "catalog ephemeris existed but no RF artifact is loaded" -- both leave
    `rf_probability` as None, but they are different facts, and reporting the
    wrong one is a false statement in the output, not a harmless simplification.
    """
    payload = _report_payload(candidate, star_id, rf_unavailable_reason)
    provider = _resolve_provider()

    if provider == "anthropic":
        return _call_anthropic(payload)
    if provider == "ollama":
        return _call_ollama(payload)
    return {"text": _template_report(payload), "source": "template"}


def _resolve_provider() -> str:
    if LLM_PROVIDER:
        return LLM_PROVIDER
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OLLAMA_MODEL:
        return "ollama"
    return "template"


def _call_anthropic(payload: dict) -> dict:
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": "Write the vetting summary for this candidate:\n"
                                + json.dumps(payload, indent=2),
                }],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", [])
                        if block.get("type") == "text").strip()
        if not text:
            logger.warning("Anthropic returned an empty response; using template.")
            return {"text": _template_report(payload), "source": "template"}
        return {"text": text, "source": "llm"}
    except Exception as e:  # noqa: BLE001 - never let the report layer break a result
        logger.warning("Anthropic call failed (%s: %s); falling back to template.",
                       type(e).__name__, e)
        return {"text": _template_report(payload), "source": "template"}


def _call_ollama(payload: dict) -> dict:
    """Local Ollama server. Same constraints prompt as Anthropic.

    Not verified against hallucination for this specific prompt+model pair --
    a 7B instruct model is more likely than Claude to slip and mention a planet
    radius or similar despite being told not to. Review real output before
    trusting it unattended. Also: this only works wherever Ollama is actually
    running. It will not be reachable on Railway/Render in Week 8's deployment
    unless Ollama is separately hosted there -- the deployed API falls back to
    the template in that case, silently and correctly, not as an error.
    """
    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": "Write the vetting summary(no more that 4 lines) for this candidate:\n"
                                + json.dumps(payload, indent=2)},
                ],
                "stream": False,
            },
            # Local CPU inference on a 7B model is slower than a cloud API call;
            # 30s (the Anthropic timeout) was cutting it close in testing.
            timeout=90.0,
        )
        response.raise_for_status()
        text = response.json().get("message", {}).get("content", "").strip()
        if not text:
            logger.warning("Ollama returned an empty response; using template.")
            return {"text": _template_report(payload), "source": "template"}
        return {"text": text, "source": "llm"}
    except Exception as e:  # noqa: BLE001 - never let the report layer break a result
        logger.warning("Ollama call to %s (model=%s) failed (%s: %s); falling "
                       "back to template. Is `ollama serve` running, and is the "
                       "model pulled (`ollama list`)?",
                       OLLAMA_HOST, OLLAMA_MODEL, type(e).__name__, e)
        return {"text": _template_report(payload), "source": "template"}


def _report_payload(candidate: dict, star_id: int, rf_unavailable_reason=None) -> dict:
    oe = candidate.get("odd_even") or {}
    return {
        "star_id": star_id,
        "orbital_period_days": candidate.get("period"),
        "transit_duration_days": candidate.get("duration_days"),
        "confidence_band": candidate.get("confidence_band"),
        "decision": candidate.get("decision"),
        "decision_reason": candidate.get("reason"),
        "cnn_probability": candidate.get("cnn_probability"),
        "rf_probability": candidate.get("rf_probability"),
        "rf_unavailable_reason": rf_unavailable_reason,
        "models_agree": candidate.get("models_agree"),
        "period_was_refined": candidate.get("refined"),
        "odd_even_depth_difference_sigma": oe.get("sigma"),
        "odd_even_flagged_as_suspicious": oe.get("is_suspicious"),
    }


def _template_report(p: dict) -> str:
    period = p.get("orbital_period_days")
    period_str = f"{period:.4f} days" if isinstance(period, (int, float)) else "an unresolved period"
    band = p.get("confidence_band") or "undetermined"

    parts = [f"KIC {p['star_id']} shows a candidate signal at {period_str}, "
             f"which the classifier rates as {band} confidence."]

    decision_reason = p.get("decision_reason") or ""
    # The decision reason already states some facts outright (e.g. "CNN and
    # Random Forest disagree...", "...odd and even transit depths differ by
    # Xsigma..."). Track which categories it already covers so the sentences
    # built below don't restate the same fact in different words right next to
    # it -- that produced a real duplicated sentence in production ("CNN and
    # Random Forest disagree on this candidate." twice in a row).
    reason_covers_agreement = decision_reason.startswith(
        "CNN and Random Forest disagree")
    reason_covers_odd_even = decision_reason.startswith(
        "Model leans planet candidate, but odd and even")

    if p.get("rf_probability") is not None:
        if not reason_covers_agreement:
            agree = "agree" if p.get("models_agree") else "disagree"
            parts.append(f"The CNN and Random Forest {agree} on this candidate.")
    elif p.get("rf_unavailable_reason"):
        parts.append(p["rf_unavailable_reason"])
    else:
        parts.append("Only the CNN was applied here, since DR25 catalog "
                     "parameters were not available for this target.")

    sigma = p.get("odd_even_depth_difference_sigma")
    if not reason_covers_odd_even and sigma is not None:
        if p.get("odd_even_flagged_as_suspicious"):
            parts.append(f"Odd and even transit depths differ by {sigma:.2f} "
                         "sigma, which is characteristic of an eclipsing binary "
                         "rather than a planet.")
        else:
            parts.append(f"Odd and even transit depths are consistent to "
                         f"within {sigma:.2f} sigma.")

    if p.get("period_was_refined"):
        parts.append("The period was refined from the initial BLS estimate before "
                     "this verdict was reached.")

    parts.append(decision_reason)
    return " ".join(part for part in parts if part).strip()