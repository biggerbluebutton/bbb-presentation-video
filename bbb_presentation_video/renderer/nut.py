# SPDX-FileCopyrightText: 2024 BigBlueButton Inc. and by respective authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal NUT container muxer for piping VFR rawvideo to ffmpeg.

NUT is ffmpeg's native pipe-friendly container format. This implements
the bare minimum needed to write a single rawvideo stream with variable
frame rate timestamps, suitable for piping to ffmpeg for encoding.

Only the PIPE variant is implemented (no syncpoints, no seeking).
"""

import struct
from typing import BinaryIO


# NUT startcodes (64-bit, big-endian)
_MAIN_STARTCODE = 0x4E4D7A561F5F04AD
_STREAM_STARTCODE = 0x4E5311405BF2F9DB

# Frame flags
_FLAG_KEY = 1 << 0
_FLAG_CODED_PTS = 1 << 3
_FLAG_SIZE_MSB = 1 << 5
_FLAG_CHECKSUM = 1 << 6
_FLAG_INVALID = 1 << 13

# The frame code byte we use for all frames
_FRAME_CODE = 0x01


def _build_crc32_table() -> list[int]:
    """Build CRC-32 IEEE lookup table (non-reflected polynomial 0x04C11DB7).

    This matches ffmpeg's ff_crc04C11DB7_update / AV_CRC_32_IEEE.
    """
    table = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return table


_CRC_TABLE = _build_crc32_table()


def _crc32(data: bytes | bytearray) -> int:
    """Compute CRC-32 matching ffmpeg's ff_crc04C11DB7_update with init=0."""
    crc = 0
    for b in data:
        crc = (_CRC_TABLE[((crc >> 24) ^ b) & 0xFF] ^ (crc << 8)) & 0xFFFFFFFF
    return crc


def _encode_v(value: int) -> bytes:
    """Encode an unsigned integer in NUT variable-length format.

    MSB-first encoding: each byte has bit 7 as continuation flag,
    bits 0-6 as data. Decoding: value = 128 * value + data.
    """
    if value < 0:
        raise ValueError(f"Cannot encode negative value: {value}")
    if value < 128:
        return bytes([value])
    # Build bytes from LSB to MSB, then reverse
    parts = []
    parts.append(value & 0x7F)  # Last byte: no continuation
    value >>= 7
    while value > 0:
        parts.append(0x80 | (value & 0x7F))  # Continuation flag set
        value >>= 7
    parts.reverse()
    return bytes(parts)


def _encode_s(value: int) -> bytes:
    """Encode a signed integer in NUT zigzag format."""
    if value <= 0:
        return _encode_v(-2 * value)
    else:
        return _encode_v(2 * value - 1)


def _encode_vb(data: bytes) -> bytes:
    """Encode a variable-length binary (length-prefixed bytes)."""
    return _encode_v(len(data)) + data


def _build_packet(startcode: int, content: bytes) -> bytes:
    """Build a complete NUT packet with startcode, forward_ptr, and CRC.

    Matches ffmpeg's put_packet(): CRC covers content bytes only,
    not startcode or forward_ptr.
    """
    forward_ptr = len(content) + 4  # content + checksum
    startcode_bytes = struct.pack(">Q", startcode)
    forward_ptr_bytes = _encode_v(forward_ptr)
    # CRC covers content only (matching ffmpeg's nutenc.c put_packet)
    checksum = _crc32(content)
    return startcode_bytes + forward_ptr_bytes + content + struct.pack("<I", checksum)


class NutMuxer:
    """Minimal NUT pipe muxer for a single rawvideo stream with VFR.

    Writes NUT headers on construction, then call write_frame() for
    each unique frame with its PTS. All frames are keyframes.

    Args:
        stream: Writable binary stream (e.g., ffmpeg's stdin).
        width: Video width in pixels.
        height: Video height in pixels.
        time_base_num: Time base numerator (e.g., 1 for 1/24 time base).
        time_base_den: Time base denominator (e.g., 24 for 1/24 time base).
        fourcc: Codec fourcc tag (default: BGR0 for cairo RGB24 surfaces).
    """

    def __init__(
        self,
        stream: BinaryIO,
        width: int,
        height: int,
        time_base_num: int,
        time_base_den: int,
        fourcc: bytes = b"BGR\x00",
    ):
        self.stream = stream
        self.width = width
        self.height = height
        self._frame_size = width * height * 4  # BGR0: 4 bytes per pixel

        # Write file ID
        stream.write(b"nut/multimedia container\0")

        # Write main header
        stream.write(self._build_main_header(time_base_num, time_base_den))

        # Write stream header
        stream.write(
            self._build_stream_header(width, height, fourcc)
        )

    def _build_main_header(
        self, time_base_num: int, time_base_den: int
    ) -> bytes:
        content = b""
        content += _encode_v(3)  # version
        content += _encode_v(1)  # stream_count
        content += _encode_v(65536)  # max_distance
        content += _encode_v(1)  # time_base_count

        # Time base: num/den
        content += _encode_v(time_base_num)
        content += _encode_v(time_base_den)

        # Frame code table: 3 groups covering all 256 codes
        # Group 1 (code 0): invalid, count=1
        content += _encode_v(_FLAG_INVALID)  # tmp_flag
        content += _encode_v(0)  # tmp_fields (no additional fields)
        # count = tmp_mul(1) - tmp_size(0) = 1

        # Group 2 (code 1): keyframe with coded PTS, size MSB, checksum
        flags = _FLAG_KEY | _FLAG_CODED_PTS | _FLAG_SIZE_MSB | _FLAG_CHECKSUM
        content += _encode_v(flags)  # tmp_flag
        content += _encode_v(6)  # tmp_fields (read up to count)
        content += _encode_s(1)  # tmp_pts (pts_delta = 1)
        content += _encode_v(1)  # tmp_mul (data_size_mul = 1)
        content += _encode_v(0)  # tmp_stream (stream_id = 0)
        content += _encode_v(0)  # tmp_size (data_size_lsb = 0)
        content += _encode_v(0)  # tmp_res (reserved_count = 0)
        content += _encode_v(1)  # count = 1

        # Group 3 (codes 2-255): all invalid, count=254
        content += _encode_v(_FLAG_INVALID)  # tmp_flag
        content += _encode_v(6)  # tmp_fields
        content += _encode_s(0)  # tmp_pts
        content += _encode_v(1)  # tmp_mul
        content += _encode_v(0)  # tmp_stream
        content += _encode_v(0)  # tmp_size
        content += _encode_v(0)  # tmp_res
        content += _encode_v(254)  # count

        # Elision headers: none
        content += _encode_v(0)  # header_count_minus1

        # Main flags: no broadcast mode
        content += _encode_v(0)  # main_flags

        return _build_packet(_MAIN_STARTCODE, content)

    def _build_stream_header(
        self, width: int, height: int, fourcc: bytes
    ) -> bytes:
        content = b""
        content += _encode_v(0)  # stream_id
        content += _encode_v(0)  # stream_class (0 = video)
        content += _encode_vb(fourcc)  # codec fourcc
        content += _encode_v(0)  # time_base_id
        content += _encode_v(0)  # msb_pts_shift (always full PTS)
        content += _encode_v(65535)  # max_pts_distance
        content += _encode_v(0)  # decode_delay (rawvideo: no reordering)
        content += _encode_v(0)  # stream_flags
        content += _encode_vb(b"")  # codec_specific_data (none for rawvideo)

        # Video-specific fields
        content += _encode_v(width)
        content += _encode_v(height)
        content += _encode_v(0)  # sample_width (unknown aspect)
        content += _encode_v(0)  # sample_height
        content += _encode_v(0)  # colorspace_type (unknown)

        return _build_packet(_STREAM_STARTCODE, content)

    def write_frame(self, pts: int, data: bytes | bytearray) -> None:
        """Write a single video frame with the given PTS.

        Args:
            pts: Presentation timestamp in time_base units.
            data: Raw BGR0 frame data (width * height * 4 bytes).
        """
        if len(data) != self._frame_size:
            raise ValueError(
                f"Frame data size {len(data)} != expected {self._frame_size}"
            )

        # Build frame header
        header = bytes([_FRAME_CODE])
        header += _encode_v(pts)  # coded_pts (full PTS, since msb_pts_shift=0)
        header += _encode_v(len(data))  # data_size_msb (mul=1, lsb=0, so msb=size)

        # Frame checksum covers the header bytes
        checksum = _crc32(header)
        header += struct.pack("<I", checksum)

        # Write header + frame data
        self.stream.write(header)
        self.stream.write(data)
