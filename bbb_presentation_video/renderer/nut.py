# SPDX-FileCopyrightText: 2024 BigBlueButton Inc. and by respective authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal NUT muxer for piping VFR raw video frames to ffmpeg."""

import struct
from io import BytesIO
from typing import BinaryIO

# NUT startcodes (64-bit big-endian)
MAIN_STARTCODE = 0x4E4D7A561F5F04AD
STREAM_STARTCODE = 0x4E5311405BF2F9DB
SYNCPOINT_STARTCODE = 0x4E4BE4ADEECA4569

# Frame flags
FLAG_KEY = 1
FLAG_CODED_PTS = 8
FLAG_SIZE_MSB = 32
FLAG_CHECKSUM = 64
FLAG_INVALID = 8192

# Our frame code uses: keyframe + explicit PTS + variable size + checksum
FRAME_FLAGS = FLAG_KEY | FLAG_CODED_PTS | FLAG_SIZE_MSB | FLAG_CHECKSUM

# CRC-32 with polynomial 0x04C11DB7 (MPEG-2)
_CRC_TABLE = [0] * 256


def _init_crc_table() -> None:
    poly = 0x04C11DB7
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        _CRC_TABLE[i] = crc


_init_crc_table()


def _crc32(data: bytes, crc: int = 0) -> int:
    for b in data:
        crc = ((_CRC_TABLE[((crc >> 24) ^ b) & 0xFF] ^ (crc << 8))) & 0xFFFFFFFF
    return crc


def _encode_v(value: int) -> bytes:
    """Encode an unsigned integer using NUT variable-length coding."""
    if value < 128:
        return bytes([value])
    groups: list[int] = []
    tmp = value
    while tmp > 0:
        groups.append(tmp & 0x7F)
        tmp >>= 7
    groups.reverse()
    result = bytearray()
    for i, g in enumerate(groups):
        if i < len(groups) - 1:
            result.append(0x80 | g)
        else:
            result.append(g)
    return bytes(result)


def _encode_s(value: int) -> bytes:
    """Encode a signed integer using NUT signed variable-length coding.

    Encoding: put_v(2 * abs(val) - (val > 0))
    """
    if value > 0:
        return _encode_v(2 * value - 1)
    else:
        return _encode_v(-2 * value)


def _build_packet(startcode: int, payload: bytes) -> bytes:
    """Build a NUT packet.

    Format: startcode(8) + forward_ptr(v) [+ header_crc(4)] + payload + content_crc(4)
    Header CRC is only present when forward_ptr > 4096.
    Content CRC (trailing 4 bytes) is always present and included in forward_ptr.
    """
    buf = BytesIO()
    buf.write(struct.pack(">Q", startcode))

    # forward_ptr counts everything after itself: [header_crc] + payload + content_crc
    forward_ptr = len(payload) + 4  # +4 for trailing content CRC
    if forward_ptr > 4096:
        forward_ptr += 4  # +4 for header CRC
    buf.write(_encode_v(forward_ptr))

    if forward_ptr > 4096:
        header_data = buf.getvalue()
        header_crc = _crc32(header_data)
        buf.write(struct.pack(">I", header_crc))

    buf.write(payload)

    # Trailing content CRC (always present, CRC of payload only)
    content_crc = _crc32(payload)
    buf.write(struct.pack(">I", content_crc))

    return buf.getvalue()


class NutMuxer:
    """Minimal NUT muxer for a single raw video stream with VFR support.

    Writes NUT format that ffmpeg can read as input, containing raw BGR0
    frames with variable timestamps.
    """

    def __init__(
        self,
        output: BinaryIO,
        width: int,
        height: int,
        timebase_num: int,
        timebase_den: int,
    ) -> None:
        self.output = output
        self.width = width
        self.height = height
        self.timebase_num = timebase_num
        self.timebase_den = timebase_den
        self.frame_size = width * height * 4  # BGR0 = 4 bytes per pixel
        self.last_pts = 0
        self.frame_count = 0

    def _build_main_header(self) -> bytes:
        """Build the main header payload."""
        buf = BytesIO()

        # version = 3
        buf.write(_encode_v(3))
        # stream_count = 1
        buf.write(_encode_v(1))
        # max_distance = frame_size + 1024 (enough for one full frame + overhead)
        max_distance = self.frame_size + 1024
        buf.write(_encode_v(max_distance))
        # time_base_count = 1
        buf.write(_encode_v(1))
        # time_base[0]: num, den
        buf.write(_encode_v(self.timebase_num))
        buf.write(_encode_v(self.timebase_den))

        # Frame code table (256 entries)
        # Field order per NUT spec (matching ffmpeg nutdec.c):
        #   tmp_flags(v), tmp_fields(v),
        #   if fields>0: tmp_pts(s)
        #   if fields>1: tmp_mul(v)
        #   if fields>2: tmp_stream(v)
        #   if fields>3: tmp_size(v)
        #   if fields>4: tmp_res(v)
        #   if fields>5: count(v)

        # Entry 1: frame_code 0 = FLAG_INVALID, count=1
        buf.write(_encode_v(FLAG_INVALID))  # flags
        buf.write(_encode_v(6))  # tmp_fields = 6 (all fields present)
        buf.write(_encode_s(0))  # pts_delta
        buf.write(_encode_v(0))  # mul
        buf.write(_encode_v(0))  # stream
        buf.write(_encode_v(0))  # size (data_size_lsb)
        buf.write(_encode_v(0))  # reserved
        buf.write(_encode_v(1))  # count = 1

        # Entry 2: frame_code 1 = our video frame code
        # KEY + CODED_PTS + SIZE_MSB + CHECKSUM
        buf.write(_encode_v(FRAME_FLAGS))
        buf.write(_encode_v(6))  # tmp_fields = 6
        buf.write(_encode_s(1))  # pts_delta = 1
        buf.write(_encode_v(1))  # mul = 1
        buf.write(_encode_v(0))  # stream = 0
        buf.write(_encode_v(0))  # size = 0
        buf.write(_encode_v(0))  # reserved
        buf.write(_encode_v(1))  # count = 1

        # Entry 3: frame_codes 2-255 = FLAG_INVALID, count=253
        # count=253 because 0x4E='N' is auto-INVALID by decoder
        buf.write(_encode_v(FLAG_INVALID))
        buf.write(_encode_v(6))  # tmp_fields = 6
        buf.write(_encode_s(0))  # pts_delta
        buf.write(_encode_v(0))  # mul
        buf.write(_encode_v(0))  # stream
        buf.write(_encode_v(0))  # size
        buf.write(_encode_v(0))  # reserved
        buf.write(_encode_v(253))  # count = 253

        # header_count_minus1 = 0 (no elision headers)
        buf.write(_encode_v(0))

        # main_flags = 0 (no broadcast mode)
        buf.write(_encode_v(0))

        return buf.getvalue()

    def _build_stream_header(self) -> bytes:
        """Build the stream header payload for our video stream."""
        buf = BytesIO()

        # stream_id = 0
        buf.write(_encode_v(0))
        # stream_class = 0 (video)
        buf.write(_encode_v(0))
        # fourcc as vb (length-prefixed bytes): "BGR\0" for bgr0 pixel format
        fourcc = b"BGR\x00"
        buf.write(_encode_v(len(fourcc)))
        buf.write(fourcc)
        # time_base_id = 0
        buf.write(_encode_v(0))
        # msb_pts_shift = 7
        buf.write(_encode_v(7))
        # max_pts_distance = timebase_den * 10 (10 seconds max gap)
        buf.write(_encode_v(self.timebase_den * 10))
        # decode_delay = 0 (raw video, no reordering)
        buf.write(_encode_v(0))
        # stream_flags = 0
        buf.write(_encode_v(0))
        # codec_specific_data = empty (vb with length 0)
        buf.write(_encode_v(0))

        # Video-specific fields
        buf.write(_encode_v(self.width))
        buf.write(_encode_v(self.height))
        # sample_width = 0 (unknown aspect ratio)
        buf.write(_encode_v(0))
        # sample_height = 0
        buf.write(_encode_v(0))
        # colorspace_type = 0 (unknown)
        buf.write(_encode_v(0))

        return buf.getvalue()

    def _build_syncpoint(self, pts: int) -> bytes:
        """Build a syncpoint payload."""
        buf = BytesIO()

        # global_key_pts as type 't': tmp = pts * time_base_count + time_base_id
        # With 1 time base and id=0: tmp = pts
        buf.write(_encode_v(pts))

        # back_ptr_div16: points to previous syncpoint
        buf.write(_encode_v(0))

        return buf.getvalue()

    def write_headers(self) -> None:
        """Write the NUT file header (magic + main header + stream header)."""
        # File magic
        self.output.write(b"nut/multimedia container\x00")

        # Main header packet
        main_payload = self._build_main_header()
        self.output.write(_build_packet(MAIN_STARTCODE, main_payload))

        # Stream header packet
        stream_payload = self._build_stream_header()
        self.output.write(_build_packet(STREAM_STARTCODE, stream_payload))

        self.output.flush()

    def write_frame(self, data: bytes, pts: int) -> None:
        """Write a single video frame with the given PTS.

        Args:
            data: Raw BGR0 frame data (width * height * 4 bytes)
            pts: Presentation timestamp in timebase units
        """
        # Write a syncpoint before each frame
        syncpoint_payload = self._build_syncpoint(pts)
        self.output.write(
            _build_packet(SYNCPOINT_STARTCODE, syncpoint_payload)
        )

        # Frame header using frame_code 1
        # Flags: FLAG_KEY | FLAG_CODED_PTS | FLAG_SIZE_MSB | FLAG_CHECKSUM
        frame_header = bytearray()
        frame_header.append(1)  # frame_code = 1

        # coded_pts (FLAG_CODED_PTS is set)
        # NUT PTS decoding: if coded_pts >= (1 << msb_pts_shift):
        #   pts = coded_pts - (1 << msb_pts_shift)
        # So we encode as: coded_pts = pts + (1 << msb_pts_shift)
        coded_pts = pts + (1 << 7)  # msb_pts_shift = 7
        frame_header.extend(_encode_v(coded_pts))

        # data_size_msb (FLAG_SIZE_MSB is set)
        # data_size = mul * data_size_msb + data_size_lsb = 1 * msb + 0
        frame_header.extend(_encode_v(len(data)))

        # Compute CRC over frame_header + data (FLAG_CHECKSUM is set)
        frame_crc = _crc32(bytes(frame_header))
        frame_crc = _crc32(data, frame_crc)

        self.output.write(bytes(frame_header))
        self.output.write(data)
        # Write the frame checksum (u32 big-endian)
        self.output.write(struct.pack(">I", frame_crc))

        self.last_pts = pts
        self.frame_count += 1

        if self.frame_count % 10 == 0:
            self.output.flush()

    def close(self) -> None:
        """Flush any remaining data."""
        self.output.flush()
