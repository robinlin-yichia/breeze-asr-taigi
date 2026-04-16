"""Tests for taigi_asr.formatters (to_txt / to_srt / to_vtt / to_json)."""

from __future__ import annotations

import json

import pytest

from taigi_asr.formatters import to_json, to_srt, to_txt, to_vtt
from taigi_asr.segments import TimestampedSegment


@pytest.fixture
def sample_segments() -> list[TimestampedSegment]:
    return [
        TimestampedSegment(start_time=0.0, end_time=3.0, text="小時候跌倒醫沒好"),
        TimestampedSegment(start_time=3.0, end_time=6.5, text="長大傻傻在這裡喊玲瓏"),
    ]


class TestToTxt:
    def test_one_line_per_segment_with_timestamps(self, sample_segments) -> None:
        expected = (
            "[00:00:00 - 00:00:03] 小時候跌倒醫沒好\n[00:00:03 - 00:00:06] 長大傻傻在這裡喊玲瓏\n"
        )
        assert to_txt(sample_segments) == expected

    def test_empty_list_returns_empty_string(self) -> None:
        assert to_txt([]) == ""


class TestToSrt:
    def test_indexed_blocks_separated_by_blank_line(self, sample_segments) -> None:
        result = to_srt(sample_segments)
        expected = (
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "小時候跌倒醫沒好\n"
            "\n"
            "2\n"
            "00:00:03,000 --> 00:00:06,500\n"
            "長大傻傻在這裡喊玲瓏\n"
            "\n"
        )
        assert result == expected

    def test_file_ends_with_blank_line(self, sample_segments) -> None:
        # SRT parsers are fussy — last cue must be followed by a blank line.
        assert to_srt(sample_segments).endswith("\n\n")

    def test_indexing_starts_at_1(self, sample_segments) -> None:
        result = to_srt(sample_segments)
        assert result.startswith("1\n")
        assert "\n2\n" in result

    def test_empty_list_is_empty_string(self) -> None:
        assert to_srt([]) == ""


class TestToVtt:
    def test_webvtt_header_then_cues(self, sample_segments) -> None:
        result = to_vtt(sample_segments)
        assert result.startswith("WEBVTT\n\n")
        # VTT uses '.' before ms (not ',')
        assert "00:00:00.000 --> 00:00:03.000" in result
        assert "00:00:03.000 --> 00:00:06.500" in result
        assert "小時候跌倒醫沒好" in result
        assert "長大傻傻在這裡喊玲瓏" in result

    def test_empty_still_emits_header(self) -> None:
        # WebVTT spec: header must be followed by a blank line even when
        # there are no cues. Some players reject the single-newline form.
        assert to_vtt([]) == "WEBVTT\n\n"


class TestToJson:
    def test_roundtrip(self, sample_segments) -> None:
        raw = to_json(sample_segments)
        data = json.loads(raw)
        assert "segments" in data
        assert len(data["segments"]) == 2
        assert data["segments"][0] == {
            "start": 0.0,
            "end": 3.0,
            "text": "小時候跌倒醫沒好",
        }
        assert data["segments"][1]["text"] == "長大傻傻在這裡喊玲瓏"

    def test_includes_metadata_when_provided(self, sample_segments) -> None:
        raw = to_json(sample_segments, meta={"engine": "faster_whisper", "language": "zh"})
        data = json.loads(raw)
        assert data["meta"] == {"engine": "faster_whisper", "language": "zh"}

    def test_utf8_not_escaped(self, sample_segments) -> None:
        raw = to_json(sample_segments)
        # Chinese characters should be raw, not \uXXXX
        assert "小時候" in raw
