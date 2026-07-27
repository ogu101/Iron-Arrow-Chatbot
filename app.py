 """
app.py — Iron Arrow History Chatbot with Progressive Disclosure RAG

Single tool: retrieve_context(section_key)
──────────────────────────────────────────
Each call advances a per-section retrieval_state counter stored in the
Chainlit session.  The counter controls which layer is returned:

  retrieval_state[key] == 0  →  first call  →  Layer 1 (bullet summary)
  retrieval_state[key] == 1  →  second call →  Layer 2 (full PDF text)
  retrieval_state[key] >= 2  →  subsequent  →  "already at maximum depth"

Layer 0 (one-sentence index) is always embedded in the system prompt and
never requires a tool call.

Layer 2 text is extracted live from the split PDF for the section so that
the JSON chunks don't need to store the full text at all.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import chainlit as cl
from anthropic import AsyncAnthropic
from pypdf import PdfReader

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    MAX_TOKENS_PER_CONVERSATION,
    MAX_TOKENS_PER_REQUEST,
    WELCOME_MESSAGE,
    ERROR_MESSAGES,
    LOG_LEVEL,
    LOG_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/chatbot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ── Path configuration ────────────────────────────────────────────────────────
CHUNKS_DIR   = Path(os.getenv("CHUNKS_DIR",   "./chunks"))
SECTIONS_DIR = Path(os.getenv("SECTIONS_DIR", "./sections"))  # split PDFs live here


# ── Chunk loading ─────────────────────────────────────────────────────────────
def _load_chunks() -> dict[str, dict]:
    """Load all section JSON chunks from disk. Returns {key: chunk_dict}."""
    chunks = {}
    if not CHUNKS_DIR.exists():
        logger.warning(f"Chunks directory not found: {CHUNKS_DIR}")
        return chunks
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        key = data["layer0_index"]["key"]
        chunks[key] = data
        logger.debug(f"Loaded chunk: {key}")
    return chunks


CHUNKS: dict[str, dict] = _load_chunks()


# ── Layer 2 — live PDF text extraction ───────────────────────────────────────
def _extract_pdf_text(key: str) -> str:
    """
    Extract full text from the section's split PDF.
    Falls back to the JSON layer2_full_text field if the PDF is not found.
    """
    pdf_path = SECTIONS_DIR / f"{key}.pdf"
    if pdf_path.exists():
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            logger.info(f"Extracted {len(text)} chars from {pdf_path.name}")
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed for {key}: {e}")

    # Fallback: pre-extracted text stored in JSON (may be empty if not generated)
    fallback = CHUNKS.get(key, {}).get("layer2_full_text", {}).get("text", "")
    if fallback:
        logger.info(f"Using JSON fallback text for {key} ({len(fallback)} chars)")
        return fallback

    return "(Full text unavailable: split PDF not found and no JSON fallback.)"


# ── Layer 0 — always-in-context index ────────────────────────────────────────
def _build_index_context() -> str:
    if not CHUNKS:
        return "(No sections loaded — run split_pdf.py first.)"
    lines = ["## Iron Arrow History — Section Index\n"]
    for key, chunk in CHUNKS.items():
        idx = chunk["layer0_index"]
        lines.append(
            f"**{idx['section_num']}. {idx['name']}** (`{key}`)\n"
            f"   {idx['one_sentence_summary']}\n"
            f"   Subsections: {', '.join(idx['subsections'])}\n"
        )
    return "\n".join(lines)


INDEX_CONTEXT = _build_index_context()

# ── Single tool definition ────────────────────────────────────────────────────
SECTION_KEYS = list(CHUNKS.keys()) or ["(none)"]

tools = [
    {
        "name": "retrieve_context",
        "description": (
            "Retrieve progressively deeper context for a section of the Iron Arrow history. "
            "Each call for the same section_key goes one layer deeper:\n"
            "  • 1st call → thematic overview paragraph + 5 narrative bullets (Layer 1)\n"
            "  • 2nd call → subsection-by-subsection fact bullets with names, dates, "
            "and key details (Layer 2)\n"
            "  • 3rd call → full raw text extracted from the section PDF (Layer 3)\n"
            "  • Further calls → nothing new; already at maximum depth\n\n"
            "Call protocol:\n"
            "  1. Always retrieve Layer 1 first for any section that looks relevant.\n"
            "  2. Call again (Layer 2) only if the overview was insufficient — e.g. the user "
            "needs a specific name, date, or fact not covered in the bullets.\n"
            "  3. Call a third time (Layer 3) only if Layer 2 still didn't have enough detail "
            "— e.g. the user needs an exact quote or very granular passage-level information.\n"
            "  4. Never skip layers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section_key": {
                    "type": "string",
                    "description": "The section key, e.g. 'section_01_beginnings'",
                    "enum": SECTION_KEYS,
                }
            },
            "required": ["section_key"],
        },
    }
]


# ── Tool execution ────────────────────────────────────────────────────────────
def execute_retrieve_context(section_key: str, retrieval_state: dict[str, int]) -> tuple[str, str]:
    """
    Advance retrieval_state[section_key] by 1, then return the appropriate layer.

    Layer mapping:
        1st call  →  layer1_summary   (paragraph + narrative bullets)
        2nd call  →  layer2_details   (subsection-by-subsection fact bullets)
        3rd call  →  layer3_full_text (raw PDF text extracted live)
        4th call+ →  already at max depth

    Returns:
        (result_text, ui_label)
    """
    chunk = CHUNKS.get(section_key)
    if chunk is None:
        return (
            f"Error: unknown section_key '{section_key}'. "
            f"Valid keys: {', '.join(CHUNKS.keys())}",
            f"Unknown key: {section_key}",
        )

    # Increment state (initialised to 0 if key not yet seen)
    next_level = retrieval_state.get(section_key, 0) + 1
    retrieval_state[section_key] = next_level

    idx          = chunk["layer0_index"]
    section_name = idx["name"]

    # ── Layer 1: thematic paragraph + narrative bullets ───────────────────────
    if next_level == 1:
        l1        = chunk["layer1_summary"]
        paragraph = l1.get("paragraph", "")
        bullets   = l1.get("bullets", [])
        bullet_block = "\n".join(f"* {b}" for b in bullets)
        result = (
            f"### {section_name} — Overview (Layer 1)\n\n"
            f"{paragraph}\n\n"
            f"{bullet_block}"
        )
        logger.info(f"[Layer 1] Served summary for '{section_key}'")
        return result, f"Loading overview: {section_name}…"

    # ── Layer 2: subsection-by-subsection fact bullets ────────────────────────
    if next_level == 2:
        details_text = chunk["layer2_details"].get("text", "")
        result = (
            f"### {section_name} — Detailed Notes (Layer 2)\n\n"
            f"{details_text}"
        )
        logger.info(f"[Layer 2] Served details for '{section_key}' ({len(details_text)} chars)")
        return result, f"Loading detailed notes: {section_name}…"

    # ── Layer 3: raw PDF full text ────────────────────────────────────────────
    if next_level == 3:
        full_text = _extract_pdf_text(section_key)
        result    = (
            f"### {section_name} — Full Text (Layer 3)\n"
            f"(Pages {idx['start_page']}–{idx['end_page']} of original PDF)\n\n"
            f"{full_text}"
        )
        logger.info(f"[Layer 3] Served full text for '{section_key}' ({len(full_text)} chars)")
        return result, f"Loading full text: {section_name}…"

    # ── Already at maximum depth ──────────────────────────────────────────────
    logger.info(f"[Layer 3+] Max depth already reached for '{section_key}' (level {next_level})")
    return (
        f"('{section_name}' is already at maximum retrieval depth — "
        f"all three context layers have been provided.)",
        f"Already at max depth: {section_name}",
    )


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are the Iron Arrow History Assistant, an expert on the history of \
Iron Arrow Honor Society at the University of Miami.

You answer questions using a **progressive disclosure retrieval system** with a single tool: \
`retrieve_context(section_key)`.

**How retrieval works — 4 layers:**
- Layer 0 (index): The section index below is always in context. Use it to identify relevant sections.
- Layer 1 (overview): 1st call per section → thematic paragraph + 5 narrative bullets. \
Sufficient for general "what is this about?" questions.
- Layer 2 (details): 2nd call per section → subsection-by-subsection fact bullets with names, \
dates, and key events. Use when a specific fact, person, or event needs more precision.
- Layer 3 (full text): 3rd call per section → complete raw text from the PDF. Use only when \
layers 1 and 2 were still insufficient — e.g. the user needs an exact quote or passage.

**Retrieval protocol — always follow this order:**
1. Identify relevant sections from the index.
2. Call `retrieve_context` once per relevant section (Layer 1 overview).
3. If the overview is insufficient for the specific question, call again for Layer 2 details.
4. Only call a third time for Layer 3 if Layer 2 still doesn't resolve the question.
5. Never skip a layer. Never call more than 3 times for the same section per turn.
6. You may retrieve multiple sections in the same turn if the question spans sections.

**Tone:** Warm, knowledgeable, and concise. Cite sections by name when referencing them.

---

{INDEX_CONTEXT}
"""


# ── Chainlit handlers ─────────────────────────────────────────────────────────
@cl.on_chat_start
async def start():
    cl.user_session.set("messages",        [])
    cl.user_session.set("total_tokens",    0)
    cl.user_session.set("retrieval_state", {})   # {section_key: int} — starts empty (all 0)
    logger.info("New chat session started")
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"USER: {message.content}")

    messages        = cl.user_session.get("messages")
    total_tokens    = cl.user_session.get("total_tokens")
    retrieval_state = cl.user_session.get("retrieval_state")

    if total_tokens >= MAX_TOKENS_PER_CONVERSATION:
        await cl.Message(content=ERROR_MESSAGES["token_limit"]).send()
        return

    messages.append({"role": "user", "content": message.content})

    try:
        final_text        = ""
        total_tokens_used = 0
        api_calls         = 0

        while True:
            api_calls += 1
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS_PER_REQUEST,
                temperature=CLAUDE_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )
            total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

            # ── Parse response ────────────────────────────────────────────
            text_parts = []
            tool_uses  = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            text_content = "".join(text_parts)

            if not tool_uses:
                final_text = text_content
                messages.append({"role": "assistant", "content": final_text})
                break

            logger.info(f"Claude requested {len(tool_uses)} tool call(s) on API call #{api_calls}")

            # ── Build assistant message ───────────────────────────────────
            assistant_content = []
            if text_content:
                assistant_content.append({"type": "text", "text": text_content})
            for tu in tool_uses:
                assistant_content.append({
                    "type":  "tool_use",
                    "id":    tu.id,
                    "name":  tu.name,
                    "input": tu.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # ── Execute tool calls ────────────────────────────────────────
            tool_results = []
            for tu in tool_uses:
                section_key = tu.input.get("section_key", "")
                result_text, ui_label = execute_retrieve_context(section_key, retrieval_state)

                await cl.Message(content=f"🔍 {ui_label}").send()

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu.id,
                    "content":     result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        # ── Persist session ───────────────────────────────────────────────
        total_tokens += total_tokens_used
        cl.user_session.set("messages",        messages)
        cl.user_session.set("total_tokens",    total_tokens)
        cl.user_session.set("retrieval_state", retrieval_state)

        logger.info(
            f"Done — {api_calls} API call(s), {total_tokens_used:,} tokens this turn, "
            f"{total_tokens:,} total | retrieval_state: {retrieval_state}"
        )

        if final_text:
            await cl.Message(
                content=(
                    f"{final_text}\n\n"
                    f"---\n📊 Tokens: {total_tokens_used:,} this turn | "
                    f"{total_tokens:,}/{MAX_TOKENS_PER_CONVERSATION:,} total"
                )
            ).send()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        error_msg = ERROR_MESSAGES["generic"].format(error=str(e))
        if "rate_limit" in str(e).lower(): error_msg = ERROR_MESSAGES["rate_limit"]
        elif "api_key"  in str(e).lower(): error_msg = ERROR_MESSAGES["api_key"]
        elif "timeout"  in str(e).lower(): error_msg = ERROR_MESSAGES["timeout"]
        await cl.Message(content=error_msg).send()
