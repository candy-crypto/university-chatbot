# judge.py
"""LLM-as-judge for the CS department chatbot evaluation pipeline.

Scores each question on five criteria (0-3 scale each) using a single
structured OpenAI call. Returns scores plus a plain-English reasoning string
for human review.
"""

import os
import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── Rubric ────────────────────────────────────────────────────────────────────

RUBRIC = """
Score each criterion on a 0–3 integer scale.

Degree name alias: "Master of Data Analytics", "MS in Data Analytics", and "PMS in Data Analytics"
are catalog aliases for the "Professional M.S. in Computational Data Analytics". Treat these as
equivalent — do not penalize an answer for using the catalog name instead of the web name or vice versa.

Semester abbreviation convention used in rotation tables:
  SP = Spring, FA = Fall, followed by 2-digit year.
  Examples: SP26 = Spring 2026, FA26 = Fall 2026, SP27 = Spring 2027, FA27 = Fall 2027.
  Current date context: the evaluation is running in May 2026. SP26 (Spring 2026) is the
  current/ending semester; FA26 (Fall 2026) is the upcoming semester. A rotation table that
  lists SP26 is citing a real scheduled offering — do NOT treat it as a hallucination or
  inconsistency even though Spring 2026 is nearly over.

FAITHFULNESS — Every factual claim in the answer traces to the retrieved context.
  3 = All claims are supported by the provided context.
  2 = Minor unsupported details; core claims are grounded.
  1 = Some significant claims lack support in context.
  0 = Answer invents facts not present in the context.
  Note: Facts that match the Ground Truth Key Facts are considered grounded — do NOT
  penalize faithfulness for facts that align with the key facts even if they do not
  appear verbatim in the Retrieved Context.

COMPLETENESS — The answer covers the key facts from the ground truth.
  3 = All key facts are present.
  2 = Most key facts present; minor gaps.
  1 = Several key facts missing.
  0 = Core question missed entirely.

SOURCE_PREFERENCE — The answer draws from the correct source type for the question.
  Domain rules:
    - Course descriptions, prerequisites, degree requirements, gen ed, VWW → prefer CATALOG.
    - Advising contacts, financial aid, assistantships, faculty directory → prefer WEB.
    - Seat/enrollment questions (open seats, registration, "can I still get in?") → answer must
      include the NMSU course search URL, not content. These are the ONLY questions that warrant
      a Banner redirect.
    - Offering-frequency questions ("Is X offered every semester?", "Is X offered in spring?",
      "When is X next offered?") → prefer the three-year course rotation page (WEB). Do NOT
      redirect these to Banner — they are answered from the rotation table, not real-time enrollment.
    - Either source acceptable when the question spans both.
  3 = Correct source type used throughout.
  2 = Mostly correct; minor preference error.
  1 = Wrong source type used for primary content.
  0 = Answer ignores the correct source entirely.

CITATION_QUALITY — Citations are present, correctly formatted, and accurate.
  Catalog chunks → cite catalog year and page range (e.g., "NMSU Academic Catalog 2025-2026, pp. 584-585"). The page range cited should match the retrieved chunk's range; do not penalize a broader range (e.g., pp. 584-587) when the relevant fact falls within that range.
  Web chunks → cite source URL. A bare URL is the correct and complete citation format for web sources. Do NOT penalize for lack of a title or label alongside the URL. Do NOT expect or require catalog-style formatting when the source is a web page — a bare URL alone is a full-credit citation for web content.
  If the answer draws only from web sources, there will be no catalog citation — this is correct and expected. Do not penalize the absence of a catalog citation when the source type is web.
  Citation placement: the system is designed to collect all citations in a "Sources:" section at the end of the response, with no inline citations in the body. This is the correct and expected format — do NOT penalize for placing citations at the end rather than tying them inline to individual claims.
  3 = All citations correct and complete.
  2 = Citations present but minor formatting issues.
  1 = Citations missing for some sources or noticeably inaccurate.
  0 = No citations, or citations are fabricated.

HALLUCINATION — No invented specifics (course numbers, names, URLs, page numbers, dates).
  3 = No hallucinated specifics detected.
  2 = Trivial error unlikely to mislead.
  1 = At least one clear hallucination.
  0 = Multiple hallucinations or a seriously misleading invented fact.
  Note: This is an inverted criterion — higher is better (no hallucinations).
  Note: Inferring the current or upcoming semester from the student's own language (e.g. "this Fall" → "Fall 2026") is valid reasoning, not hallucination.

RESPONSE_QUALITY — Direct, professional, well-organized; no filler or extraneous preamble.
  3 = Leads with the answer; concise; appropriate tone; no filler phrases.
  2 = Mostly clear; minor verbosity or slight tone issue (e.g., one filler opener).
  1 = Noticeably indirect, padded, or contains filler ("Great question!", "Certainly!", "As an AI").
  0 = Dominated by filler or preamble; evasive; fails to deliver useful information promptly.
  Filler phrases to flag: "Great question", "Certainly!", "Of course!", "I'd be happy to help",
  "As an AI", "I'm here to help", "Absolutely!", "Sure!".
"""

# ── Judge prompt builder ───────────────────────────────────────────────────────

def build_judge_prompt(
    question: str,
    key_facts: list[str],
    context: str,
    answer: str,
    banner_redirect_expected: bool,
) -> str:
    key_facts_text = "\n".join(f"- {f}" for f in key_facts)

    banner_note = ""
    if banner_redirect_expected:
        banner_note = (
            "\nIMPORTANT: This question requires a course search redirect. "
            "The correct answer must include the NMSU course search URL "
            "(https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/termSelection?mode=search) "
            "and should NOT attempt to answer from context. "
            "If the answer includes that URL and instructs the user to check there, "
            "you MUST score faithfulness=3, source_preference=3, citation_quality=3, and hallucination=3. "
            "Do NOT apply the hallucination rubric to this answer — override it entirely. "
            "In particular, do not penalize the answer for naming the current or upcoming semester "
            "(e.g. 'Fall 2026') when the student's question references 'this Fall' or 'this semester' — "
            "that is valid temporal reasoning from the student's own words, not hallucination."
        )

    return f"""You are evaluating a university department chatbot response.
Score the SYSTEM ANSWER against the GROUND TRUTH KEY FACTS using the rubric below.
Return a JSON object with integer scores (0-3) and a reasoning string.

## Student Question
{question}

## Ground Truth Key Facts
{key_facts_text}

## Retrieved Context (what the chatbot was given)
{context}

## System Answer (what the chatbot said)
{answer}
{banner_note}

## Scoring Rubric
{RUBRIC}

## Required Output Format
Return ONLY a JSON object with these exact keys:
{{
  "faithfulness": <0-3>,
  "completeness": <0-3>,
  "source_preference": <0-3>,
  "citation_quality": <0-3>,
  "hallucination": <0-3>,
  "response_quality": <0-3>,
  "reasoning": "<plain English explanation of scores, 2-4 sentences>"
}}
"""


# ── Judge caller ──────────────────────────────────────────────────────────────

def judge_question(
    question: str,
    key_facts: list[str],
    context: str,
    answer: str,
    banner_redirect_expected: bool = False,
) -> dict[str, Any]:
    """Call the LLM judge and return scores + reasoning.

    On any failure, returns all scores as -1 and records the error.
    Never raises — the harness continues even if the judge fails.
    """
    prompt = build_judge_prompt(
        question=question,
        key_facts=key_facts,
        context=context,
        answer=answer,
        banner_redirect_expected=banner_redirect_expected,
    )

    try:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = response.choices[0].message.content
        scores = json.loads(raw)
        tokens_used = response.usage.total_tokens if response.usage else None

        return {
            "faithfulness":      int(scores.get("faithfulness", -1)),
            "completeness":      int(scores.get("completeness", -1)),
            "source_preference": int(scores.get("source_preference", -1)),
            "citation_quality":  int(scores.get("citation_quality", -1)),
            "hallucination":     int(scores.get("hallucination", -1)),
            "response_quality":  int(scores.get("response_quality", -1)),
            "reasoning":         scores.get("reasoning", ""),
            "judge_tokens_used": tokens_used,
            "judge_error":       None,
        }

    except Exception as e:
        return {
            "faithfulness":      -1,
            "completeness":      -1,
            "source_preference": -1,
            "citation_quality":  -1,
            "hallucination":     -1,
            "response_quality":  -1,
            "reasoning":         "",
            "judge_tokens_used": None,
            "judge_error":       str(e),
        }


def judge_total(scores: dict[str, Any]) -> float | None:
    """Compute normalized mean of the six judge scores (0.0–1.0).

    Returns None if any score is -1 (judge failed).
    """
    keys = ["faithfulness", "completeness", "source_preference",
            "citation_quality", "hallucination", "response_quality"]
    values = [scores.get(k, -1) for k in keys]
    if any(v == -1 for v in values):
        return None
    return sum(values) / (len(values) * 3)  # max per criterion is 3
