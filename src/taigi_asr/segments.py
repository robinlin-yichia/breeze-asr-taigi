"""Timestamped transcript segment — the canonical data structure passed between engines,
formatters, and the UI layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimestampedSegment:
    """A transcript line with start/end seconds.

    Immutable so formatters can hash/compare safely.
    """

    start_time: float
    end_time: float
    text: str

    @staticmethod
    def format_time(seconds: float | None, srt_format: bool = False) -> str:
        """Format seconds as HH:MM:SS (wall clock) or HH:MM:SS,mmm (SRT).

        None / negative are clamped to zero to keep formatters defensive against
        upstream decoder quirks (Whisper sometimes emits None timestamps).

        Carries millisecond overflow into minutes/hours so values like 59.9996
        don't render as an invalid ``HH:MM:60,000``.
        """
        if seconds is None or seconds < 0:
            seconds = 0.0

        if srt_format:
            # Round to the nearest millisecond, then decompose — avoids the
            # float-formatting "60.000" overflow when input is epsilon below 60.
            total_ms = int(round(seconds * 1000))
            ms = total_ms % 1000
            total_s = total_ms // 1000
            s = total_s % 60
            total_m = total_s // 60
            m = total_m % 60
            h = total_m // 60
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        # Wall clock: truncate seconds (matches prior behaviour).
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def to_timestamp_line(self) -> str:
        start = self.format_time(self.start_time)
        end = self.format_time(self.end_time)
        return f"[{start} - {end}] {self.text}"

    def to_srt_block(self, index: int) -> str:
        start = self.format_time(self.start_time, srt_format=True)
        end = self.format_time(self.end_time, srt_format=True)
        return f"{index}\n{start} --> {end}\n{self.text}\n"

    def to_vtt_block(self) -> str:
        start = self.format_time(self.start_time, srt_format=True).replace(",", ".")
        end = self.format_time(self.end_time, srt_format=True).replace(",", ".")
        return f"{start} --> {end}\n{self.text}\n"

    def to_json_dict(self) -> dict[str, float | str]:
        return {"start": self.start_time, "end": self.end_time, "text": self.text}
