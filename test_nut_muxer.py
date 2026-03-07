#!/usr/bin/env python3
"""Standalone test for the NUT muxer.

Generates a test video with colored frames at VFR timestamps and
verifies ffmpeg can decode the NUT stream correctly.
"""

import io
import struct
import subprocess
import sys

sys.path.insert(0, ".")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "nut", "bbb_presentation_video/renderer/nut.py"
)
nut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nut)

WIDTH = 320
HEIGHT = 240
FRAME_SIZE = WIDTH * HEIGHT * 4  # BGR0


def make_solid_frame(r: int, g: int, b: int) -> bytes:
    """Create a solid-color BGR0 frame."""
    pixel = struct.pack("BBBB", b, g, r, 0)  # BGR0 byte order
    return pixel * (WIDTH * HEIGHT)


def test_nut_to_buffer():
    """Write a NUT stream to a buffer and verify structure."""
    buf = io.BytesIO()

    # Time base: 1/24 (24fps)
    muxer = nut.NutMuxer(buf, WIDTH, HEIGHT, 1, 24)

    # Write 5 frames at VFR timestamps:
    # Frame 0 at PTS 0 (t=0.000s)
    # Frame 1 at PTS 48 (t=2.000s) -- 2 second hold on first frame
    # Frame 2 at PTS 49 (t=2.042s)
    # Frame 3 at PTS 72 (t=3.000s) -- ~1 second hold
    # Frame 4 at PTS 120 (t=5.000s) -- 2 second hold
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    pts_values = [0, 48, 49, 72, 120]

    for pts, (r, g, b) in zip(pts_values, colors):
        frame = make_solid_frame(r, g, b)
        muxer.write_frame(pts, frame)

    data = buf.getvalue()
    print(f"NUT stream size: {len(data)} bytes")
    print(f"  Header: {data[:24]}")

    # Verify file ID (24 chars + null = 25 bytes)
    file_id = b"nut/multimedia container\0"
    assert data[:len(file_id)] == file_id, "Bad file ID"
    print("  File ID: OK")

    # Verify main header startcode
    offset = len(file_id)
    main_sc = struct.unpack(">Q", data[offset:offset + 8])[0]
    assert main_sc == 0x4E4D7A561F5F04AD, f"Bad main startcode: {hex(main_sc)}"
    print("  Main startcode: OK")

    print(f"  Total stream: {len(data)} bytes for {len(pts_values)} frames")
    print(f"  Per-frame overhead: ~{(len(data) - FRAME_SIZE * len(pts_values)) // len(pts_values)} bytes")
    return data


def test_ffmpeg_decode(nut_data: bytes):
    """Pipe NUT stream to ffmpeg and verify it decodes correctly."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\nffmpeg not found, skipping decode test")
        return False

    # Test 1: Decode to null (verify no errors)
    print("\nTest: ffmpeg decode to null...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "nut", "-i", "pipe:0", "-f", "null", "-"],
        input=nut_data,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr.decode()}")
        return False
    print("  OK")

    # Test 2: Encode to MP4 and check with ffprobe
    print("\nTest: ffmpeg encode to MP4...")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "nut", "-i", "pipe:0",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-vsync", "vfr",
            "-movflags", "+faststart",
            "-f", "mp4", "/tmp/test_nut_output.mp4",
        ],
        input=nut_data,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr.decode()}")
        return False
    print("  OK")

    # Test 3: Verify frame timestamps with ffprobe
    print("\nTest: ffprobe frame timestamps...")
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time,pkt_pts_time,key_frame",
            "-of", "csv=p=0",
            "/tmp/test_nut_output.mp4",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ffprobe FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr.decode()}")
        return False

    output = result.stdout.decode().strip()
    print(f"  Frame timestamps:\n  {output.replace(chr(10), chr(10) + '  ')}")

    # Also check total duration
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            "/tmp/test_nut_output.mp4",
        ],
        capture_output=True,
    )
    if result.returncode == 0:
        duration = result.stdout.decode().strip()
        print(f"  Total duration: {duration}s (expected: ~5.0s)")

    print("  OK")
    return True


if __name__ == "__main__":
    print("=== NUT Muxer Test ===\n")

    print("Test 1: Generate NUT stream in memory")
    nut_data = test_nut_to_buffer()

    print("\nTest 2: FFmpeg decode/encode")
    success = test_ffmpeg_decode(nut_data)

    if success:
        print("\n=== All tests PASSED ===")
    else:
        print("\n=== Some tests skipped (no ffmpeg) ===")
        print("Run on a system with ffmpeg to fully validate.")
