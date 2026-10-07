"""Rule/structure-based clause segmentation for PolicyLens.

This module intentionally does NOT classify clauses with AI. It converts the
page-level text produced by pdf_extractor into structured clause records while
preserving page and section context. A later DeBERTa model will assign the
PolicyLens taxonomy labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable


# Numbered headings such as "14. Waiting Period" or "15. Exclusions".
SECTION_RE = re.compile(r"^(?P<number>\d{1,2})\s*[.,;:]\s*(?P<title>[A-Z][A-Za-z0-9 &'(),\-/]{2,100})\s*:??$")

# Lettered benefit headings such as "A) Hospital Cash Benefit:".
LETTER_HEADING_RE = re.compile(
    r"^(?P<marker>[A-Z])\s*[.)]\s*(?P<title>[A-Za-z][A-Za-z0-9 &'(),\-/]{2,100})\s*:?$"
)

# Clause/list markers: i., ii), a), 1., 2), etc.
CLAUSE_MARKER_RE = re.compile(
    r"^(?P<marker>(?:[ivxlcdm]+|[a-z]|\d+))\s*(?P<sep>[.)])\s*(?P<text>[A-Z].+)$",
    re.IGNORECASE,
)

# Headings that end with a colon and are short enough to be structural labels.
SUBHEADING_RE = re.compile(
    r"^(?P<title>[A-Z][A-Za-z0-9 &'(),\-/]{2,90}):$"
)

# Some headings in extracted PDFs lose punctuation/spaces.

# Numbered structural subheadings that commonly occur inside a parent
# section (especially HDFC ERGO general terms). These must not be mistaken
# for numbered clause/list items.
KNOWN_POLICY_SECTION_TERMS = (
    "benefits offered under the plan", "payment of premiums",
    "mode and high hcb rebates", "automatic renewal date", "autom c renewal date",
    "options", "eligibility conditions and other restrictions",
    "eligibility condi ons and other restric ons", "other features",
    "free look period", "loan", "assignment", "benefit limits and conditions", "specific conditions applicable for persons with disability",
    "specific conditions applicable for persons with hiv - aids",
    "benefit limits and condi ons", "commencement and termination of benefit covers",
    "commencement and termina on of benefit covers", "termination of policy",
    "termina on of policy", "waiting period", "wai ng period", "exclusions", "taxes",
)

NUMBERED_SUBHEADING_TERMS = (
    "disclosure of information",
    "condition precedent to admission of liability",
    "claim settlement",
    "complete discharge",
    "multiple policies",
    "fraud",
    "cancellation",
    "migration",
    "portability",
    "renewal of policy",
    "premium payment in instalments",
    "moratorium period",
    "possibility of revision of terms",
    "free look period",
    "withdrawal of policy",
    "grievance redressal procedure",
    "nomination",
)

def parse_numbered_subheading(line: str) -> str | None:
    """Recognize known numbered headings nested inside a parent section."""
    line = normalize_line(line)
    m = re.match(r"^(?:\d{1,2})\s*[.)]\s*(?P<title>[A-Za-z][A-Za-z0-9 &'(),/\-]+?)\s*:??$", line)
    if not m:
        return None
    title = m.group("title").strip(" :")
    lower = re.sub(r"\s+", " ", title.lower())
    if any(term in lower for term in NUMBERED_SUBHEADING_TERMS):
        return title
    return None

KNOWN_SUBHEADINGS = {
    "general waiting period",
    "specific waiting period",
    "major surgical benefit",
    "hospital cash benefit",
    "day care procedure benefit",
    "other surgical benefit",
    "ambulance benefit",
    "premium waiver benefit",
    "surrender",
}

# Lines that are usually page furniture rather than policy clauses.
NOISE_PATTERNS = [
    re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.I),
    re.compile(r"^www\.[^ ]+$", re.I),
]


def normalize_line(line: str) -> str:
    """Normalize one extracted/OCR line without destroying legal wording."""
    from .text_cleaner import clean_text, normalize_unicode
    line = normalize_unicode(line)
    line = re.sub(r"[\x00-\x1f\x7f-\x9f\ufffd]", " ", line)
    line = clean_text(line)
    line = re.sub(r"^\d+\s+(\d{1,2}[.)])$", r"\1", line.strip())
    return line.strip()


def is_noise(line: str) -> bool:
    line = normalize_line(line)
    if not line:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def _looks_like_top_level_section_title(title: str) -> bool:
    """Reject numbered lists that look like ordinary content rather than headings."""
    title = normalize_line(title).strip(" :")
    if not title or len(title) > 110:
        return False
    words = title.split()
    if len(words) > 9:
        return False
    lower = title.lower()
    if any(term in lower for term in KNOWN_POLICY_SECTION_TERMS):
        return True

    # Common structural vocabulary in insurance policy wordings. This is only
    # a structural filter; it does not assign a PolicyLens category.
    structural_terms = (
        "preamble", "operating", "operative", "definitions", "definition",
        "coverage", "coverages", "benefit", "benefits", "exclusion",
        "exclusions", "waiting",
        "claim", "claims", "procedure", "procedures", "renewal",
        "termination", "eligibility", "option", "options", "premium",
        "features", "general", "territorial", "migration",
        "portability", "arbitration", "moratorium", "annexure", "annex",
        "grievance", "dispute", "policy", "schedule", "discount",
        "hospitalization", "hospitalisation", "table", "claim procedure", "terms", "choose", "work out",
        "other features", "restrictions", "restriction", "discharge",
    )
    lower = title.lower()
    # Long numbered sentences often contain structural words (e.g.
    # ``Sum Insured ... shall ...``) but are list items, not headings.
    sentence_cues = (" shall ", " will ", " if ", " provided ", " means ", " is ", " are ")
    if any(cue in f" {lower} " for cue in sentence_cues):
        return False
    if any(term in lower for term in structural_terms):
        return True

    return False


def parse_decimal_heading(line: str) -> tuple[str, str] | None:
    """Recognize headings such as 4.1 Inpatient Care as a nested heading."""
    line = normalize_line(line)
    m = re.match(r"^(?P<major>\d{1,2})\.(?P<minor>\d{1,2})\s+(?P<title>[A-Z][A-Za-z0-9 &'(),/\-]{2,110})\s*:??$", line)
    if not m:
        return None
    return m.group("major"), m.group("title").strip(" :")

def parse_section_heading(line: str) -> tuple[str, str] | None:
    line = normalize_line(line)
    # Handle explicit SECTION headings used by many policy wordings.
    m = re.match(r"^SECTION\s*(?P<number>\d{1,2})\s*[.,;:\-]?\s*(?P<title>[A-Za-z][A-Za-z0-9 &'(),\-/]{2,110})\s*:??$", line, re.I)
    if m:
        number = m.group("number")
        title = m.group("title").strip(" :-")
        return number, title

    # Some PDFs put a page number immediately before the actual section
    # marker, e.g. ``2 2. Operating Clause``. Remove only that pattern.
    line = re.sub(r"^\d+\s+(?=\d{1,2}\s*[.,;:])", "", line)

    # A real top-level heading should start with its section number. Avoid
    # searching inside a sentence because that turns numbered list items into
    # false sections.
    match = re.match(
        r"^(?P<number>\d{1,2})\s*[.,;:]\s*(?P<title>[A-Za-z][A-Za-z0-9 &'(),\-/]{2,110})\s*:??$",
        line,
    )
    if not match:
        return None

    number = match.group("number")
    title = match.group("title").strip(" :")
    if not _looks_like_top_level_section_title(title):
        return None
    return number, title


def parse_unumbered_structural_heading(line: str) -> str | None:
    """Recognize standalone uppercase policy headings when numbering is absent."""
    line = normalize_line(line)
    if not line or len(line) > 100 or re.search(r"[.;!?]$", line):
        return None
    if not re.fullmatch(r"[A-Z0-9 &'(),/\-]+", line):
        return None
    lower = line.lower()
    terms = (
        "preamble", "operative clause", "definitions", "coverage", "benefits",
        "cumulative bonus", "waiting period", "exclusions", "moratorium",
        "claim procedure", "general terms", "specific terms", "redressal",
        "grievance", "table of benefits", "renewal", "portability",
        "migration", "arbitration", "territorial jurisdiction", "eligibility",
        "policy tenure", "sum insured", "claim settlement", "fraud",
        "cancellation", "withdrawal of policy", "free look period",
    )
    if any(term in lower for term in terms):
        return line.title()
    return None

def is_progressive_section(number: str, current_section: str | None) -> bool:
    """Recognize top-level numbered sections without confusing numbered lists.

    Insurance documents often contain numbered list items (1., 2., ...).
    This brochure's main sections progress from 1 through 16, so progression
    is a strong structural signal. After section 16, legal appendix numbering
    is deliberately not treated as policy sections.
    """
    n = int(number)
    if n < 1 or n > 16:
        return False
    if current_section is None:
        return n == 1
    m = re.match(r"^(\d+)\.", current_section)
    if not m:
        return False
    return n > int(m.group(1))


def parse_letter_heading(line: str) -> tuple[str, str] | None:
    line = normalize_line(line)
    match = LETTER_HEADING_RE.match(line)
    if not match:
        return None
    raw_marker = match.group("marker")
    marker = raw_marker.upper()
    title = match.group("title").strip(" :")
    if len(title.split()) > 10:
        return None

    lower = title.lower()
    heading_words = (
        "coverage", "specific conditions", "conditions", "benefit",
        "accidental death", "permanent disablement", "operating clause",
        "definitions", "standard definitions", "specific definitions",
        "general terms", "terms and clauses", "standard general terms",
        "claim", "exclusions", "coverage", "hospitalization", "treatment",
    )
    # Uppercase lettered headings are structural when they look like named
    # sections/benefits. Lowercase a)/b) items remain ordinary clauses.
    if raw_marker.isupper() and any(w in lower for w in heading_words):
        return marker, title
    return None

def coalesce_standalone_section_numbers(lines: list[str]) -> list[str]:
    """Join PDF lines such as ``5`` + ``Waiting Period`` into ``5. Waiting Period``."""
    out=[]
    i=0
    while i < len(lines):
        line=normalize_line(lines[i])
        if re.fullmatch(r"\d{1,2}", line) and i+1 < len(lines):
            nxt=normalize_line(lines[i+1])
            if (nxt and len(nxt) <= 110 and not re.match(r"^(?:[ivxlcdm]+|[a-z]|\d+)[.)]\s", nxt, re.I)
                    and not re.search(r"[.!?]$", nxt)
                    and _looks_like_top_level_section_title(nxt)):
                out.append(f"{line}. {nxt}")
                i += 2
                continue
        if line:
            out.append(line)
        i += 1
    return out

def coalesce_standalone_markers(lines: list[str]) -> list[str]:
    """Merge PDF lines where a list marker is separated from its text.

    Some policy pages extract ``i`` and ``.`` as separate lines, or ``ii.``
    on one line and ``Treatment...`` on the next. This function reconstructs
    those markers before clause parsing.
    """
    result: list[str] = []
    pending_marker: str | None = None
    i = 0
    marker_only = re.compile(r"^(?:[ivxlcdm]+|[a-z]|\d+)[.)]?$" )
    while i < len(lines):
        line = normalize_line(lines[i])
        if not line:
            i += 1
            continue

        if pending_marker is not None:
            # A PDF page number can appear immediately before the real marker,
            # e.g. ``1`` followed by ``1.``. In that case the first token is
            # page furniture, not part of the heading marker.
            if pending_marker.isdigit() and re.fullmatch(r"(?:[ivxlcdm]+|[a-z]|\d+)[.)]?", line, re.I):
                pending_marker = line
                i += 1
                continue

            # A standalone punctuation line completes markers such as ``i`` + ``.``.
            if line in {".", ")"}:
                pending_marker += line
                i += 1
                continue
            result.append(pending_marker + " " + line)
            pending_marker = None
            i += 1
            continue

        # Page-number + section-marker patterns such as ``1 1.`` or ``2 2.``
        # occur in some PDFs. Keep only the actual section marker so it can be
        # joined with the following heading text.
        page_plus_marker = re.fullmatch(r"\d+\s+(\d{1,2}[.)])", line)
        if page_plus_marker:
            pending_marker = page_plus_marker.group(1)
            i += 1
            continue

        if marker_only.fullmatch(line):
            pending_marker = line
            i += 1
            continue

        result.append(line)
        i += 1

    if pending_marker is not None:
        result.append(pending_marker)
    return result


def split_embedded_markers(line: str) -> list[str]:
    """Split lines containing multiple adjacent legal-list markers.

    PDF extraction can place several numbered items on one visual line. For
    example: ``iii. Treatment... iv. Treatment...``. Keeping these together
    would reduce recall of individual clauses, so we split only when a marker
    is followed by whitespace and text.
    """
    line = normalize_line(line)
    matches = list(re.finditer(r"(?<![A-Za-z0-9])(?:[ivxlcdm]+|[a-z]|\d+)[.)](?=\s+|[A-Z])", line, re.I))
    if len(matches) <= 1:
        return [line]
    parts: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        part = line[start:end].strip()
        if part:
            parts.append(part)
    # Preserve any text before the first marker if it is substantive.
    prefix = line[:matches[0].start()].strip()
    if prefix and len(prefix) > 2:
        parts.insert(0, prefix)
    return parts


def parse_clause_marker(line: str) -> tuple[str, str] | None:
    line = normalize_line(line)
    match = CLAUSE_MARKER_RE.match(line)
    if not match:
        return None

    marker = f"{match.group('marker')}{match.group('sep')}"
    text = match.group("text").strip()

    # Numeric values such as "90 days" are not clause markers unless the
    # marker is followed by a meaningful phrase.
    if len(text) < 3:
        return None
    return marker, text


def parse_subheading(line: str) -> str | None:
    line = normalize_line(line)
    match = SUBHEADING_RE.match(line)
    if not match:
        return None
    title = match.group("title").strip()
    if title.lower() in KNOWN_SUBHEADINGS:
        return title
    # Conservative generic rule: short heading-like labels only.
    if len(title.split()) <= 7 and not re.search(r"[.;!?]", title):
        return title
    return None


def looks_like_continuation(line: str) -> bool:
    """Return True for a normal text line that should join the current clause."""
    line = normalize_line(line)
    if not line:
        return False
    if parse_section_heading(line) or parse_letter_heading(line):
        return False
    if parse_clause_marker(line) or parse_subheading(line):
        return False
    return True


def join_clause_lines(lines: Iterable[str]) -> str:
    """Join wrapped PDF/OCR lines into readable clause text."""
    cleaned = [normalize_line(x) for x in lines if normalize_line(x)]
    if not cleaned:
        return ""

    text = " ".join(cleaned)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class ClauseRecord:
    clause_id: str
    policy_id: str
    page_start: int
    page_end: int
    section: str | None
    subsection: str | None
    marker: str | None
    clause_text: str
    source_lines: list[str]
    category: str | None = None


class ClauseBuilder:
    def __init__(self, policy_id: str = "POL001") -> None:
        self.policy_id = policy_id
        self.records: list[ClauseRecord] = []
        self.current_section: str | None = None
        self.current_subsection: str | None = None
        self.current_marker: str | None = None
        self.current_lines: list[str] = []
        self.current_page_start: int | None = None
        self.current_page_end: int | None = None
        self._counter = 0

    def flush(self) -> None:
        text = join_clause_lines(self.current_lines)
        if text:
            self._counter += 1
            self.records.append(
                ClauseRecord(
                    clause_id=f"{self.policy_id}_C{self._counter:04d}",
                    policy_id=self.policy_id,
                    page_start=self.current_page_start or self.current_page_end or 0,
                    page_end=self.current_page_end or self.current_page_start or 0,
                    section=self.current_section,
                    subsection=self.current_subsection,
                    marker=self.current_marker,
                    clause_text=text,
                    source_lines=list(self.current_lines),
                )
            )

        self.current_lines = []
        self.current_marker = None
        self.current_page_start = None
        self.current_page_end = None

    def start_clause(self, page: int, marker: str | None, text: str) -> None:
        self.flush()
        self.current_marker = marker
        self.current_page_start = page
        self.current_page_end = page
        if text:
            self.current_lines.append(text)

    def append(self, page: int, line: str) -> None:
        if self.current_page_start is None:
            # A paragraph before the first explicit marker is still useful.
            self.current_page_start = page
            self.current_page_end = page
        else:
            self.current_page_end = page
        self.current_lines.append(line)

    def update_page(self, page: int) -> None:
        if self.current_lines:
            self.current_page_end = page


def extract_clauses(
    extracted: dict[str, Any],
    policy_id: str = "POL001",
    min_clause_chars: int = 35,
) -> list[dict[str, Any]]:
    """Extract structured clauses from the page-level extraction JSON.

    The parser is deliberately deterministic so that the clause boundaries can
    be inspected and corrected before creating the supervised DeBERTa dataset.
    """
    builder = ClauseBuilder(policy_id=policy_id)

    pages = extracted.get("pages", [])
    started_sections = False
    stop_after_policy = False
    for page_obj in pages:
        if stop_after_policy:
            break
        page_no = int(page_obj["page"])
        text = page_obj.get("text") or ""
        lines = coalesce_standalone_markers(coalesce_standalone_section_numbers(text.splitlines()))

        for raw_line in lines:
            raw_line = normalize_line(raw_line)
            if is_noise(raw_line):
                continue

            # The brochure switches from policy terms to statutory/legal
            # notices after the Taxes section. Those notices are not part of
            # the policy-clause taxonomy used by PolicyLens.
            if re.search(r"SECTION\s*45\s*OF\s*THE\s*INSURANCE\s*ACT", raw_line, re.I):
                stop_after_policy = True
                break
            if re.search(r"PROHIBITION\s+OF\s+REBATES", raw_line, re.I):
                stop_after_policy = True
                break

            for line in split_embedded_markers(raw_line):
                numbered_subheading = parse_numbered_subheading(line)
                current_major = None
                if builder.current_section:
                    mcur = re.match(r"^(\d+)\.", builder.current_section)
                    if mcur:
                        current_major = int(mcur.group(1))
                if numbered_subheading and current_major is not None and current_major >= 9:
                    builder.flush()
                    builder.current_subsection = numbered_subheading
                    continue

                section = parse_section_heading(line)
                if section and is_progressive_section(section[0], builder.current_section):
                    builder.flush()
                    number, title = section
                    builder.current_section = f"{number}. {title}"
                    builder.current_subsection = None
                    started_sections = True
                    continue

                structural_heading = parse_unumbered_structural_heading(line)
                if structural_heading:
                    builder.flush()
                    builder.current_subsection = structural_heading
                    # Once a numbered policy section has begun, an unnumbered
                    # all-caps heading is treated as a subsection. For documents
                    # whose headings are mostly unnumbered, it becomes the
                    # current structural context instead.
                    if builder.current_section is None:
                        builder.current_section = structural_heading
                        started_sections = True
                    continue

                # Ignore brochure introduction/marketing material before the first
                # numbered policy section.
                if not started_sections:
                    continue

                decimal_heading = parse_decimal_heading(line)
                if decimal_heading:
                    major, title = decimal_heading
                    current_major = None
                    if builder.current_section:
                        mcur = re.match(r"^(\d+)\.", builder.current_section)
                        if mcur:
                            current_major = int(mcur.group(1))
                    if current_major is None or int(major) != current_major:
                        builder.flush()
                        builder.current_section = f"{major}. Coverage" if major == "4" else f"{major}. {title}"
                        builder.current_subsection = title
                        started_sections = True
                        continue
                    builder.flush()
                    builder.current_subsection = title
                    continue

                letter = parse_letter_heading(line)
                if letter:
                    # A benefit heading is a structural subsection, not a clause.
                    builder.flush()
                    marker, title = letter
                    builder.current_subsection = f"{marker}) {title}"
                    continue

                marker = parse_clause_marker(line)
                if marker:
                    marker_text, clause_text = marker
                    builder.start_clause(page_no, marker_text, clause_text)
                    continue

                subheading = parse_subheading(line)
                if subheading:
                    builder.flush()
                    builder.current_subsection = subheading
                    continue

                if looks_like_continuation(line):
                    builder.append(page_no, line)

        builder.update_page(page_no)

    builder.flush()

    # Some product brochures are designed as marketing/summary documents and
    # do not contain numbered legal sections. In that case the strict legal
    # parser above can legitimately produce zero clauses. Fall back to a
    # heading-aware paragraph segmentation so such documents can still enter
    # the annotation workflow. This is still deterministic; it does not assign
    # any PolicyLens category.
    if not builder.records:
        return _extract_brochure_clauses(extracted, policy_id, min_clause_chars)

    # Remove tiny fragments created by decorative/table text while retaining
    # short but meaningful clauses such as "No surrender value...".
    filtered: list[ClauseRecord] = []
    for record in builder.records:
        if len(record.clause_text) < min_clause_chars and record.marker is None:
            continue
        # Decorative/summary bullet lists are not policy clauses.
        if re.match(r"^(?:[•*+]|[-–])", record.clause_text):
            continue
        filtered.append(record)

    # Re-number after filtering so IDs remain contiguous and deterministic.
    output: list[dict[str, Any]] = []
    for idx, record in enumerate(filtered, start=1):
        record.clause_id = f"{policy_id}_C{idx:04d}"
        output.append(asdict(record))
    return output


def _is_brochure_heading(line: str) -> bool:
    """Heuristic heading detector for non-numbered product brochures."""
    line = normalize_line(line)
    if not line or len(line) > 90:
        return False
    if re.search(r"[.;!?]$", line):
        return False
    # Typical legal/benefit headings and short title-case labels.
    words = line.split()
    if len(words) <= 8 and any(ch.isalpha() for ch in line):
        lower = line.lower()
        heading_terms = (
            "coverage", "treatment", "benefit", "expenses", "ambulance",
            "hospitalization", "hospitalisation", "co-payment", "waiting",
            "exclusion", "discount", "restoration", "bonus", "vaccination",
            "cash", "surgery", "wellness", "entry age", "sum insured",
            "policy tenure", "claim", "delivery", "new born", "newborn",
            "domiciliary", "organ donor", "air ambulance", "pre hospitalization",
            "post hospitalization", "out-patient", "outpatient", "modern treatments",
            "ayush", "road ambulance", "limits", "eligibility"
        )
        if any(term in lower for term in heading_terms):
            return True
        # Title-case short labels such as "Automatic Restoration of SI".
        titleish = sum(1 for w in words if w[:1].isupper())
        if titleish >= max(1, len(words) - 1):
            return True
    return False


def _extract_brochure_clauses(
    extracted: dict[str, Any],
    policy_id: str,
    min_clause_chars: int,
) -> list[dict[str, Any]]:
    """Fallback segmentation for brochures without numbered legal sections."""
    records: list[ClauseRecord] = []
    counter = 0

    for page_obj in extracted.get("pages", []):
        page_no = int(page_obj["page"])
        lines = [normalize_line(x) for x in (page_obj.get("text") or "").splitlines()]
        lines = [x for x in lines if x and not is_noise(x)]
        if not lines:
            continue

        current: list[str] = []
        current_heading: str | None = None
        current_start = page_no

        def flush() -> None:
            nonlocal current, current_heading, current_start, counter
            text = join_clause_lines(current)
            if len(text) >= min_clause_chars:
                counter += 1
                records.append(ClauseRecord(
                    clause_id=f"{policy_id}_C{counter:04d}",
                    policy_id=policy_id,
                    page_start=current_start,
                    page_end=page_no,
                    section=None,
                    subsection=current_heading,
                    marker=None,
                    clause_text=text,
                    source_lines=list(current),
                ))
            current = []
            current_heading = None
            current_start = page_no

        for line in lines:
            # Avoid splitting obvious continuation lines such as list items.
            if _is_brochure_heading(line) and len(current) >= 1:
                flush()
                current_heading = line
                current.append(line)
                current_start = page_no
            else:
                if not current:
                    current_start = page_no
                current.append(line)

        flush()

    return [asdict(r) for r in records]


def extract_clauses_from_json(
    json_path: str,
    policy_id: str = "POL001",
    min_clause_chars: int = 35,
) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return extract_clauses(data, policy_id=policy_id, min_clause_chars=min_clause_chars)


def _native_layout_lines(page: Any) -> list[dict[str, Any]]:
    """Extract visually ordered text lines from a PyMuPDF page."""
    lines: list[dict[str, Any]] = []
    data = page.get_text("dict", sort=True)
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = normalize_line(text)
            if not text:
                continue
            bbox = line.get("bbox", [0, 0, 0, 0])
            lines.append({"text": text, "y": float(bbox[1]), "x": float(bbox[0])})
    return sorted(lines, key=lambda item: (item["y"], item["x"]))


def _ocr_layout_lines(page: Any, dpi: int = 120) -> list[dict[str, Any]]:
    """OCR a page and reconstruct text lines using Tesseract coordinates."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return []

    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        lang="eng",
        output_type=pytesseract.Output.DICT,
    )

    grouped: dict[tuple[int, int, int], list[tuple[float, str]]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = normalize_line(data["text"][i])
        if not text:
            continue
        conf_raw = data.get("conf", ["-1"] * n)[i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1
        if conf >= 0 and conf < 20:
            continue
        key = (
            int(data.get("block_num", [0] * n)[i]),
            int(data.get("par_num", [0] * n)[i]),
            int(data.get("line_num", [0] * n)[i]),
        )
        grouped.setdefault(key, []).append((float(data["left"][i]), text))

    result: list[dict[str, Any]] = []
    for (block_no, par_no, line_no), words in grouped.items():
        words.sort(key=lambda item: item[0])
        text = " ".join(word for _, word in words)
        # Approximate y from the first word; exact coordinates are only used
        # for ordering, not for citations.
        first_left = words[0][0]
        # Retrieve y from the first matching OCR token.
        y = 0.0
        for i in range(n):
            if (
                int(data.get("block_num", [0] * n)[i]) == block_no
                and int(data.get("par_num", [0] * n)[i]) == par_no
                and int(data.get("line_num", [0] * n)[i]) == line_no
            ):
                y = float(data["top"][i])
                break
        result.append({"text": normalize_line(text), "y": y, "x": first_left})

    return sorted(result, key=lambda item: (item["y"], item["x"]))


def extract_clauses_from_pdf(
    pdf_path: str,
    policy_id: str = "POL001",
    min_clause_chars: int = 35,
    use_ocr: bool = False,
) -> list[dict[str, Any]]:
    """Extract clauses directly from a PDF using visual text order.

    Native PyMuPDF blocks are used by default because they preserve the layout
    of headings and clauses much better than a flattened text stream. OCR can
    be enabled for pages where native text quality is poor.
    """
    import fitz
    from .text_cleaner import extraction_quality

    document = fitz.open(pdf_path)
    pseudo_extracted: dict[str, Any] = {"pages": []}

    for index, page in enumerate(document):
        native = _native_layout_lines(page)
        native_text = "\n".join(item["text"] for item in native)
        lines = native

        if use_ocr and extraction_quality(native_text) < 0.94:
            ocr = _ocr_layout_lines(page)
            if len(ocr) >= 5:
                lines = ocr

        pseudo_extracted["pages"].append(
            {
                "page": index + 1,
                "text": "\n".join(item["text"] for item in lines),
            }
        )

    return extract_clauses(
        pseudo_extracted,
        policy_id=policy_id,
        min_clause_chars=min_clause_chars,
    )
