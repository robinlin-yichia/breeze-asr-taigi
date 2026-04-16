"""Tests for taigi_asr.segments.TimestampedSegment.

TDD Red phase: these tests define the contract before any implementation exists.
"""

from __future__ import annotations

import json

import pytest

from taigi_asr.segments import TimestampedSegment


class TestFormatTime:
    """format_time must emit HH:MM:SS (wall clock) or HH:MM:SS,mmm (SRT) formats."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00"),
            (1.4, "00:00:01"),
            (59.9, "00:00:59"),
            (60.0, "00:01:00"),
            (3599.5, "00:59:59"),
            (3600.0, "01:00:00"),
            (3723.25, "01:02:03"),
        ],
    )
    def test_wall_clock(self, seconds: float, expected: str) -> None:
        seg = TimestampedSegment(start_time=seconds, end_time=seconds + 1, text="x")
        assert seg.format_time(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.234, "00:00:01,234"),
            (59.999, "00:00:59,999"),
            (60.5, "00:01:00,500"),
            (3723.001, "01:02:03,001"),
        ],
    )
    def test_srt_format(self, seconds: float, expected: str) -> None:
        seg = TimestampedSegment(start_time=0, end_time=1, text="x")
        assert seg.format_time(seconds, srt_format=True) == expected

    def test_negative_or_none_is_clamped_to_zero(self) -> None:
        seg = TimestampedSegment(start_time=0, end_time=1, text="x")
        assert seg.format_time(-5.0) == "00:00:00"
        assert seg.format_time(None) == "00:00:00"  # type: ignore[arg-type]

    def test_srt_rounding_does_not_overflow_seconds(self) -> None:
        """59.9996 must never render as HH:MM:60,000 (invalid SRT).

        Regression for Copilot review #7: earlier `f"{secs:06.3f}"` formatting
        rounded values epsilon below 60 up to ``60.000``.
        """
        seg = TimestampedSegment(0, 1, "x")
        # epsilon below 60 seconds -> should roll into the minute
        assert seg.format_time(59.9996, srt_format=True) == "00:01:00,000"
        # epsilon below an hour -> should roll into the hour
        assert seg.format_time(3599.9996, srt_format=True) == "01:00:00,000"
        # exact boundary still clean
        assert seg.format_time(59.999, srt_format=True) == "00:00:59,999"
        assert seg.format_time(60.0, srt_format=True) == "00:01:00,000"


class TestTimestampLine:
    def test_basic(self) -> None:
        seg = TimestampedSegment(start_time=2.5, end_time=7.25, text="小時候跌倒")
        assert seg.to_timestamp_line() == "[00:00:02 - 00:00:07] 小時候跌倒"

    def test_multi_hour(self) -> None:
        seg = TimestampedSegment(start_time=3723.0, end_time=3730.0, text="hi")
        assert seg.to_timestamp_line() == "[01:02:03 - 01:02:10] hi"


class TestSrtBlock:
    def test_basic(self) -> None:
        seg = TimestampedSegment(start_time=1.5, end_time=3.75, text="長大傻傻")
        expected = "1\n00:00:01,500 --> 00:00:03,750\n長大傻傻\n"
        assert seg.to_srt_block(1) == expected

    def test_index_increments(self) -> None:
        seg = TimestampedSegment(start_time=10.0, end_time=12.0, text="test")
        assert seg.to_srt_block(42).startswith("42\n")


class TestVttBlock:
    def test_basic_uses_dot_as_decimal_separator(self) -> None:
        seg = TimestampedSegment(start_time=1.5, end_time=3.75, text="玲瓏")
        # WebVTT uses '.' not ',' before milliseconds
        expected = "00:00:01.500 --> 00:00:03.750\n玲瓏\n"
        assert seg.to_vtt_block() == expected


class TestJsonDict:
    def test_shape(self) -> None:
        seg = TimestampedSegment(start_time=0.123, end_time=1.456, text="喊玲瓏")
        d = seg.to_json_dict()
        assert d == {"start": 0.123, "end": 1.456, "text": "喊玲瓏"}
        # must be JSON-serializable
        assert json.loads(json.dumps(d))["text"] == "喊玲瓏"


class TestDataclassEquality:
    def test_two_segments_with_same_fields_are_equal(self) -> None:
        a = TimestampedSegment(start_time=1.0, end_time=2.0, text="x")
        b = TimestampedSegment(start_time=1.0, end_time=2.0, text="x")
        assert a == b
