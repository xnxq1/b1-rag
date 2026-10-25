import re

from rag.domain.entities import Heading

_MAX_HEADING_LEN = 120
_MAX_HEADING_WORDS = 14
_MAX_NUMBERED_MIXED_CASE_WORDS = 6
_MIN_CAPS_LETTERS = 4

_NUMBERED = re.compile(r"^(?:\d{1,3}\.)*\d{1,3}\.?\s+[A-ZА-ЯЁ«\"]")
_KEYWORD = re.compile(r"^(?:статья|раздел|глава|часть|приложение)\s*(?:№\s*)?[\dIVXLC]+", re.IGNORECASE)
_LETTERS = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _is_continuation(next_line: str | None) -> bool:
    if not next_line:
        return False
    first_letter = _LETTERS.search(next_line)
    stripped = next_line.lstrip()
    starts_with_letter_or_bracket = bool(stripped) and (stripped[0].isalpha() or stripped[0] in '(«"')
    return first_letter is not None and starts_with_letter_or_bracket and first_letter.group().islower()


def is_heading(line: str, next_line: str | None = None) -> bool:
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_LEN or len(line.split()) > _MAX_HEADING_WORDS:
        return False
    if line[-1] in ".;,:" and not _KEYWORD.match(line):
        return False
    if _is_continuation(next_line):
        return False
    letters = _LETTERS.findall(line)
    all_caps = len(letters) >= _MIN_CAPS_LETTERS and all(ch.isupper() for ch in letters)
    if _KEYWORD.match(line):
        return True
    if _NUMBERED.match(line):
        return all_caps or len(line.split()) <= _MAX_NUMBERED_MIXED_CASE_WORDS
    return all_caps


def normalize_heading(line: str) -> str:
    return " ".join(line.split())


def detect_headings(text: str) -> list[Heading]:
    lines = text.splitlines(keepends=True)
    headings = []
    offset = 0
    for index, line in enumerate(lines):
        next_line = next((candidate for candidate in lines[index + 1 :] if candidate.strip()), None)
        if is_heading(line, next_line):
            leading = len(line) - len(line.lstrip())
            headings.append(Heading(char_start=offset + leading, title=normalize_heading(line)))
        offset += len(line)
    return headings
