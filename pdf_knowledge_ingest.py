"""
PDF → clean text → structured chunks → embeddings → ChromaDB.

Designed for narrative-heavy WHO PDFs (dialogues, figures): extract with pdfminer,
strip low-signal lines, chunk to ~100–300 words, tag with lightweight heuristics,
then ingest via StressRAGSystem.ingest_structured_chunks().
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from pdfminer.high_level import extract_text

# --- Cleaning -----------------------------------------------------------------

_PAGE_RE = re.compile(r"\bPage\s+\d+\b", re.IGNORECASE)
_FIG_RE = re.compile(r"(?m)^\s*(Figure|Fig\.?)\s+\d+[^\n]*$", re.IGNORECASE)
_ILLUS_RE = re.compile(r"(?m)^\s*Illustration[^\n]*$", re.IGNORECASE)
_CHAR_LINE_RE = re.compile(
    r"(?m)^\s*(Character\s+[A-Za-z0-9]+|[A-Za-z]+\s+Character)\s*:\s*.*$"
)
_DIALOGUE_NAME_RE = re.compile(
    r"(?m)^\s{0,4}[A-Z][a-z]{1,25}\s+and\s+[A-Z][a-z]{1,25}\s*:\s*.*$"
)
_NUM_ONLY_LINE = re.compile(r"(?m)^\s*\d{1,4}\s*$")


def clean_extracted_text(raw: str) -> str:
    text = raw.replace("\x00", " ")
    text = _PAGE_RE.sub(" ", text)
    text = _FIG_RE.sub(" ", text)
    text = _ILLUS_RE.sub(" ", text)
    text = _CHAR_LINE_RE.sub(" ", text)
    text = _DIALOGUE_NAME_RE.sub(" ", text)
    text = _NUM_ONLY_LINE.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"[ \t\r\f]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# --- Technique / condition heuristics -----------------------------------------

_TECHNIQUE_KEYWORDS: Sequence[tuple[str, str]] = (
    (r"\b(breath|breathing|respirat)\w*", "breathing exercises"),
    (r"\b(ground|anchor|present moment|54321)\w*", "grounding"),
    (r"\b(mindful|meditat|MBSR)\w*", "mindfulness"),
    (r"\b(defus|thoughts as|observe your thought)\w*", "cognitive defusion"),
    (r"\b(values|valued action|what matters)\w*", "values-based action"),
    (r"\b(self[- ]?compassion|kind to yourself)\w*", "self-compassion"),
    (r"\b(sleep|rest|insomnia)\w*", "sleep hygiene"),
    (r"\b(exercise|movement|walk|physical activ)\w*", "physical activity"),
    (r"\b(social|connect|support|talk to)\w*", "social connection"),
    (r"\b(problem[- ]solving|cope|copings skill)\w*", "problem-solving / coping"),
    (r"\b(routine|schedule|plan your day)\w*", "routine and planning"),
    (r"\b(relax|muscle tension|body scan)\w*", "relaxation"),
    (r"\b(fear|worry|anxiety)\w*", "anxiety management"),
)

_CONDITION_KEYWORDS: Sequence[tuple[str, str]] = (
    (r"\b(anxiety|worry|panic|fear)\w*", "anxiety"),
    (r"\b(stress|overwhelm|pressure)\w*", "high stress"),
    (r"\b(sleep|insomnia|tired|fatigue)\w*", "sleep difficulties"),
    (r"\b(sad|depress|low mood)\w*", "low mood"),
    (r"\b(anger|irritab|frustrat)\w*", "irritability"),
    (r"\b(isolat|lonely|alone)\w*", "isolation"),
    (r"\b(work|job|occupation)\w*", "work stress"),
)


def _infer_technique(text: str) -> str:
    low = text.lower()
    for pattern, label in _TECHNIQUE_KEYWORDS:
        if re.search(pattern, low, re.IGNORECASE):
            return label
    return "general stress coping"


def _infer_conditions(text: str, max_n: int = 6) -> List[str]:
    low = text.lower()
    found: List[str] = []
    for pattern, label in _CONDITION_KEYWORDS:
        if re.search(pattern, low, re.IGNORECASE) and label not in found:
            found.append(label)
        if len(found) >= max_n:
            break
    if not found:
        found = ["stress", "well-being"]
    return found


def _infer_stress_levels(text: str) -> List[str]:
    low = text.lower()
    if re.search(r"\b(severe|crisis|extreme|panic)\w*", low):
        return ["moderate", "high", "very_high"]
    if re.search(r"\b(mild|slight|little)\w*", low):
        return ["very_low", "low", "moderate"]
    return ["low", "moderate", "high", "very_high"]


def paragraphs_from_text(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 40]


def chunk_paragraphs(
    paragraphs: Iterable[str],
    min_words: int = 100,
    max_words: int = 280,
) -> List[str]:
    """Merge paragraphs until each chunk is in [min_words, max_words] when possible."""
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_words = 0

    for p in paragraphs:
        w = len(p.split())
        if not buf:
            buf.append(p)
            buf_words = w
            continue
        if buf_words + w <= max_words:
            buf.append(p)
            buf_words += w
            if buf_words >= min_words:
                flush()
            continue
        if buf_words < min_words:
            buf.append(p)
            buf_words += w
            flush()
        else:
            flush()
            buf = [p]
            buf_words = w

    if buf:
        if buf_words < min_words and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(buf)
        else:
            chunks.append("\n\n".join(buf).strip())

    return [c for c in chunks if len(c.split()) >= 40]


def build_structured_records(
    chunks: Sequence[str],
    source: str = "WHO",
    doc_slug: str = "doing_what_matters_stress",
    chunk_type: str = "technique",
) -> List[dict]:
    out: List[dict] = []
    for i, text in enumerate(chunks):
        technique = _infer_technique(text)
        conditions = _infer_conditions(text)
        stress_levels = _infer_stress_levels(text)
        out.append(
            {
                "text": text,
                "source": source,
                "type": chunk_type,
                "technique": technique,
                "conditions": conditions,
                "stress_level_mapping": stress_levels,
                "doc_slug": doc_slug,
                "chunk_index": i,
            }
        )
    return out


def pdf_to_structured_chunks(
    pdf_path: str | Path,
    source: str = "WHO",
    doc_slug: str | None = None,
    min_words: int = 100,
    max_words: int = 280,
) -> List[dict]:
    path = Path(pdf_path)
    raw = extract_text(str(path))
    cleaned = clean_extracted_text(raw)
    paras = paragraphs_from_text(cleaned)
    chunks = chunk_paragraphs(paras, min_words=min_words, max_words=max_words)
    slug = doc_slug or re.sub(r"[^\w]+", "_", path.stem.lower()).strip("_")
    return build_structured_records(chunks, source=source, doc_slug=slug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest WHO-style stress PDF into ChromaDB")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="knowledge/doingwhatmattersintimesofstress.pdf",
        help="Path to PDF",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="knowledge/who_stress_chunks.json",
        help="Write structured chunks to this JSON file",
    )
    parser.add_argument("--source", type=str, default="WHO")
    parser.add_argument("--no-chroma", action="store_true", help="Only build JSON, skip Chroma")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Do not delete existing ids with this ingest's --id-prefix before add (default: replace)",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="who",
        help="ChromaDB id prefix (e.g. who_mhgap for second WHO doc; avoids wiping other PDF chunks)",
    )
    args = parser.parse_args()

    records = pdf_to_structured_chunks(args.pdf, source=args.source)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} chunks to {out_path}")

    if args.no_chroma:
        return

    from rag_system import StressRAGSystem

    rag = StressRAGSystem()
    rag.ingest_structured_chunks(
        records, id_prefix=args.id_prefix, replace=not args.append
    )
    print("Ingested into ChromaDB collection", rag.collection_name)


if __name__ == "__main__":
    main()
