"""Select the chunks that can answer a question, before any LLM is involved.

Retrieval runs in three steps:

1. **Scope** (deterministic): read which days and parts of the day the question refers
   to ("Thursday", "tomorrow evening", "tonight", "at 6am", "this weekend"...).
2. **Coverage check**: if every day the question asks about is outside the forecast
   (in the past, or beyond the 7 days), return no chunks and a clear note. The LLM is
   never asked about days it has no data for.
3. **Rank** with BM25 only when the scope still leaves too many chunks; results are
   returned in time order. When the question names a time near a day-part boundary
   (e.g. 6am), the neighbouring chunk is added so both sides are visible.

Guide chunks (e.g. what to wear) are ranked separately and added when they match.
"""

import re
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi

from weather_rag.chunks import PART_WINDOWS, Chunk, DayPart

MAX_FORECAST_CHUNKS = 8
MAX_GUIDE_CHUNKS = 2

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
PART_WORDS: dict[str, DayPart] = {
    "morning": DayPart.MORNING,
    "breakfast": DayPart.MORNING,
    "afternoon": DayPart.AFTERNOON,
    "lunchtime": DayPart.AFTERNOON,
    "evening": DayPart.EVENING,
    "after work": DayPart.EVENING,
    "night": DayPart.NIGHT,
    "overnight": DayPart.NIGHT,
}
STOPWORDS = {
    "a", "an", "and", "are", "be", "do", "for", "i", "in", "is", "it", "me", "my", "need",
    "of", "on", "or", "should", "the", "this", "to", "what", "when", "which", "will", "with",
}  # fmt: skip

_TIME_12H = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b")
_TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:am|pm))")
_MONTH_PATTERN = "|".join(MONTHS)
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_PATTERN})[a-z]*\b")
_MONTH_DAY = re.compile(rf"\b({_MONTH_PATTERN})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")
_THE_NTH = re.compile(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b")
_IN_N_DAYS = re.compile(r"\bin (\d{1,2}) days?\b")
_FAR_FUTURE = re.compile(r"\b(next month|in (?:a|two|three|\d+) weeks?|next year)\b")


class QueryScope(BaseModel):
    """What a question refers to, read deterministically from its wording."""

    model_config = ConfigDict(frozen=True)

    days: list[date] = []
    parts: list[DayPart] = []
    hours: list[int] = []
    far_future: bool = False


class RetrievalResult(BaseModel):
    """Chunks to ground an answer in, plus a note when part of the question can't be."""

    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk]
    answerable: bool
    note: str | None = None


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without common filler words, used for BM25."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]


def _to_24h(hour: int, minute: int, suffix: str) -> int | None:
    if not (1 <= hour <= 12 and 0 <= minute < 60):
        return None
    return hour % 12 + (12 if suffix == "pm" else 0)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def read_scope(question: str, today: date) -> QueryScope:
    """Work out which days, day parts and hours a question asks about."""
    text = question.lower()
    days: list[date] = []
    parts: list[DayPart] = []
    hours: list[int] = []

    if "day after tomorrow" in text:
        days.append(today + timedelta(days=2))
        text = text.replace("day after tomorrow", " ")
    if "tomorrow" in text:
        days.append(today + timedelta(days=1))
    if "yesterday" in text:
        days.append(today - timedelta(days=1))
    if "today" in text or "this morning" in text or "this afternoon" in text:
        days.append(today)
    if "tonight" in text or "this evening" in text:
        days.append(today)
        parts.append(DayPart.NIGHT if "tonight" in text else DayPart.EVENING)
    if "weekend" in text:
        for offset in range(7):
            candidate = today + timedelta(days=offset)
            if candidate.weekday() >= 5:
                days.append(candidate)

    for index, name in enumerate(WEEKDAYS):
        if re.search(rf"\blast {name}\b", text):
            days.append(today - timedelta(days=(today.weekday() - index) % 7 or 7))
        elif re.search(rf"\bnext {name}\b", text):
            days.append(today + timedelta(days=(index - today.weekday()) % 7 or 7))
        elif re.search(rf"\b{name}('s|s)?\b", text):
            days.append(today + timedelta(days=(index - today.weekday()) % 7))

    for match in _DAY_MONTH.finditer(text):
        month = MONTHS.index(match.group(2)) + 1
        if found := _safe_date(today.year, month, int(match.group(1))):
            days.append(found)
    for match in _MONTH_DAY.finditer(text):
        month = MONTHS.index(match.group(1)) + 1
        if found := _safe_date(today.year, month, int(match.group(2))):
            days.append(found)
    for match in _THE_NTH.finditer(text):
        if found := _safe_date(today.year, today.month, int(match.group(1))):
            days.append(found)
    for match in _IN_N_DAYS.finditer(text):
        days.append(today + timedelta(days=int(match.group(1))))

    for word, part in PART_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            parts.append(part)

    for match in _TIME_12H.finditer(text):
        if (
            hour := _to_24h(int(match.group(1)), int(match.group(2) or 0), match.group(3))
        ) is not None:
            hours.append(hour)
    for match in _TIME_24H.finditer(text):
        hours.append(int(match.group(1)))
    if "noon" in text or "midday" in text:
        hours.append(12)

    return QueryScope(
        days=sorted(set(days)),
        parts=list(dict.fromkeys(parts)),
        hours=sorted(set(hours)),
        far_future=bool(_FAR_FUTURE.search(text)),
    )


def _part_for_hour(hour: int) -> DayPart:
    for part, (start, length) in PART_WINDOWS.items():
        if start <= hour < start + length or start <= hour + 24 < start + length:
            return part
    return DayPart.NIGHT


def _boundary_side(hour: int, part: DayPart) -> int:
    """Return -1 if ``hour`` is the first hour of ``part``, +1 if it is the last, else 0."""
    start, length = PART_WINDOWS[part]
    if hour == start % 24:
        return -1
    if hour == (start + length - 1) % 24:
        return 1
    return 0


class Retriever:
    """Holds the indexed chunks for one forecast and answers retrieval queries."""

    def __init__(self, forecast_chunks: list[Chunk], guide_chunks: list[Chunk]) -> None:
        self.forecast_chunks = forecast_chunks
        self.guide_chunks = guide_chunks
        self._position = {chunk.id: index for index, chunk in enumerate(forecast_chunks)}
        self._forecast_index = (
            BM25Okapi([tokenize(c.text) for c in forecast_chunks]) if forecast_chunks else None
        )
        self._guide_index = (
            BM25Okapi([tokenize(c.text) for c in guide_chunks]) if guide_chunks else None
        )
        days = [chunk.day for chunk in forecast_chunks if chunk.day is not None]
        self.first_day = min(days) if days else None
        self.last_day = max(days) if days else None

    def coverage(self) -> str:
        if self.first_day is None or self.last_day is None:
            return "No forecast data is available."
        return f"The forecast covers {self.first_day:%a %d %b} to {self.last_day:%a %d %b}."

    def retrieve(self, question: str) -> RetrievalResult:
        """Return the chunks relevant to ``question``, or a note if it can't be answered."""
        if self.first_day is None or self.last_day is None:
            return RetrievalResult(chunks=[], answerable=False, note=self.coverage())

        scope = read_scope(question, today=self.first_day)
        in_range = [d for d in scope.days if self.first_day <= d <= self.last_day]
        missing = [d for d in scope.days if d not in in_range]

        if (scope.days and not in_range) or (scope.far_future and not in_range):
            what = ", ".join(f"{d:%a %d %b}" for d in missing) or "that period"
            return RetrievalResult(
                chunks=[],
                answerable=False,
                note=f"There is no forecast data for {what}. {self.coverage()}",
            )

        candidates = self._select(scope, in_range)
        if len(candidates) > MAX_FORECAST_CHUNKS:
            candidates = self._rank(question, candidates)[:MAX_FORECAST_CHUNKS]
        candidates = self._add_neighbours(candidates, scope)
        candidates.sort(key=lambda c: self._position[c.id])

        note = None
        if missing:
            listed = ", ".join(f"{d:%a %d %b}" for d in missing)
            note = f"No forecast data for {listed}. {self.coverage()}"
        return RetrievalResult(
            chunks=candidates + self._matching_guides(question), answerable=True, note=note
        )

    def _select(self, scope: QueryScope, days: list[date]) -> list[Chunk]:
        parts = set(scope.parts) | {_part_for_hour(h) for h in scope.hours}
        wanted_ids: set[str] | None = None
        if days:
            wanted_ids = set()
            for day in days:
                for part in parts or set(DayPart):
                    wanted_ids.add(f"{day.isoformat()}-{part.value}")
                # Small hours (00-05) belong to the previous day's night chunk.
                if any(h < 6 for h in scope.hours):
                    wanted_ids.add(f"{(day - timedelta(days=1)).isoformat()}-night")
        return [
            chunk
            for chunk in self.forecast_chunks
            if (wanted_ids is None or chunk.id in wanted_ids)
            and (wanted_ids is not None or not parts or chunk.part in parts)
        ]

    def _rank(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._forecast_index is None:
            return chunks
        scores = self._forecast_index.get_scores(tokenize(question))

        # BM25 gives words found in every chunk (e.g. "like" from "feels like") a slightly
        # negative weight. Treat anything <= 0 as "no match" so those chunks tie and fall
        # back to time order: an uninformative question gets the soonest periods.
        def key(chunk: Chunk) -> tuple[float, int]:
            position = self._position[chunk.id]
            return (-max(float(scores[position]), 0.0), position)

        return sorted(chunks, key=key)

    def _add_neighbours(self, chunks: list[Chunk], scope: QueryScope) -> list[Chunk]:
        """Add the chunk on the other side of a boundary when the question's hour sits on one.

        E.g. 6am is the first hour of the morning, so the previous night is added too.
        """
        selected = {chunk.id for chunk in chunks}
        result = list(chunks)
        for chunk in chunks:
            if chunk.part is None:
                continue
            for hour in scope.hours:
                if _part_for_hour(hour) is not chunk.part:
                    continue
                side = _boundary_side(hour, chunk.part)
                neighbour = self._position[chunk.id] + side
                if side and 0 <= neighbour < len(self.forecast_chunks):
                    candidate = self.forecast_chunks[neighbour]
                    if candidate.id not in selected:
                        selected.add(candidate.id)
                        result.append(candidate)
        return result

    def _matching_guides(self, question: str) -> list[Chunk]:
        if self._guide_index is None:
            return []
        scores = self._guide_index.get_scores(tokenize(question))
        ranked = sorted(
            zip(scores, self.guide_chunks, strict=True), key=lambda pair: -float(pair[0])
        )
        return [chunk for score, chunk in ranked[:MAX_GUIDE_CHUNKS] if float(score) > 0]
