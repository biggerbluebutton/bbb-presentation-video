# SPDX-FileCopyrightText: 2024 BigBlueButton Inc. and by respective authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import shutil
import struct
import tempfile
import threading
import time
from enum import Enum
from fractions import Fraction
from queue import Empty, Queue
from subprocess import PIPE, CalledProcessError, Popen
from typing import List, Optional, Tuple, cast

import cairo

from bbb_presentation_video import events
from bbb_presentation_video.events import EventsInfo, PerPodEvent, RecordEvent
from bbb_presentation_video.events.helpers import Color, Size
from bbb_presentation_video.renderer.cursor import CursorRenderer
from bbb_presentation_video.renderer.presentation import PresentationRenderer
from bbb_presentation_video.renderer.tldraw import TldrawRenderer
from bbb_presentation_video.renderer.whiteboard import ShapesRenderer

DRAWING_BG = Color.from_int(0xE2E8ED)


class Codec(Enum):
    H264 = "h264"
    H264_MP4 = "h264_mp4"
    VP9 = "vp9"
    RAW_VIDEO = "rawvideo"


class EncoderError(Exception):
    """Raised when the encoder thread encounters an error."""

    pass


class Encoder:
    queue: "Queue[Optional[bytearray]]"
    ret_queue: "Queue[bytearray]"

    def __init__(
        self, output: str, width: int, height: int, framerate: Fraction, codec: Codec
    ):
        self.output = output
        self.width = width
        self.height = height
        self.framerate = framerate
        self.codec = codec

        self.queue = Queue()
        self.ret_queue = Queue()
        for x in range(0, 3):
            self.ret_queue.put(bytearray(width * height * 4))

        # Track encoder thread errors so the main thread can detect them
        self.error: Optional[BaseException] = None
        self.ffmpeg_process: Optional[Popen] = None

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def put(self, data: bytearray) -> None:
        # Use a timeout loop so we can detect if the encoder thread has died.
        # Without this, a dead encoder thread causes ret_queue.get() to block
        # forever since buffers are never returned.
        while True:
            if self.error is not None:
                raise EncoderError(
                    f"Encoder thread failed: {self.error}"
                ) from self.error
            if not self.thread.is_alive():
                raise EncoderError(
                    "Encoder thread exited unexpectedly"
                    + (f": {self.error}" if self.error else "")
                )
            try:
                buf = self.ret_queue.get(timeout=2)
                break
            except Empty:
                continue
        buf[:] = data
        self.queue.put(buf)

    def join(self) -> None:
        # This is a sentinel value to tell the writing thread to exit
        self.queue.put(None)
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            print("WARNING: Encoder thread did not exit within timeout, killing ffmpeg")
            self._kill_ffmpeg()
            self.thread.join(timeout=10)
        if self.error is not None:
            raise EncoderError(
                f"Encoder thread failed: {self.error}"
            ) from self.error

    def cancel(self) -> None:
        """Cancel encoding: drain queues, signal thread to stop, kill ffmpeg."""
        # Send sentinel to unblock the encoder thread's queue.get()
        self.queue.put(None)
        self._kill_ffmpeg()
        self.thread.join(timeout=10)

    def _kill_ffmpeg(self) -> None:
        """Forcefully terminate the ffmpeg process if it's running."""
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.kill()
            except OSError:
                pass

    def output_raw(self) -> None:
        with open(self.output, "wb") as f:
            while True:
                buf = self.queue.get()
                if buf is None:
                    break
                f.write(buf)
                self.ret_queue.put(buf)

    def output_ffmpeg(self) -> None:
        if self.codec == Codec.H264:
            codec_opts = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast"]
        elif self.codec == Codec.H264_MP4:
            codec_opts = [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
            ]
        elif self.codec == Codec.VP9:
            codec_opts = [
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-lossless",
                "1",
                "-row-mt",
                "1",
            ]

        # Use mp4 container for h264_mp4 output, matroska for everything else
        if self.codec == Codec.H264_MP4:
            container_fmt = "mp4"
        else:
            container_fmt = "matroska"

        # Launch the video encoder
        # Note that the hardcoded 'bgr0' here is only applicable in
        # little-endian!
        ffmpeg_cmdline = [
            "ffmpeg",
            "-y",
            "-nostats",
            "-v",
            "warning",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr0",
            "-video_size",
            f"{self.width:d}x{self.height:d}",
            "-framerate",
            str(self.framerate),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"mpdecimate=max={int(round(self.framerate)):d}:hi=1:lo=1:frac=1",
            *codec_opts,
            "-threads",
            "2",
            "-g",
            str(round(self.framerate) * 10),
            "-f",
            container_fmt,
            self.output,
        ]

        ffmpeg = Popen(ffmpeg_cmdline, stdin=PIPE, stdout=PIPE, close_fds=True)
        self.ffmpeg_process = ffmpeg
        assert ffmpeg.stdout is not None and ffmpeg.stdin is not None
        ffmpeg.stdout.close()

        try:
            while True:
                buf = self.queue.get()
                if buf is None:
                    break

                ffmpeg.stdin.write(buf)

                self.ret_queue.put(buf)
        except BrokenPipeError:
            # ffmpeg process died; drain remaining items from queue so the
            # main thread's put() calls don't block indefinitely.
            while True:
                buf = self.queue.get()
                if buf is None:
                    break
                self.ret_queue.put(buf)
            raise
        finally:
            try:
                ffmpeg.stdin.close()
            except BrokenPipeError:
                pass

        ffmpeg.wait(timeout=120)

        if ffmpeg.returncode != 0:
            raise CalledProcessError(returncode=ffmpeg.returncode, cmd=ffmpeg_cmdline)

    def run(self) -> None:
        try:
            if self.codec == Codec.RAW_VIDEO:
                self.output_raw()
            else:
                self.output_ffmpeg()
        except Exception as e:
            self.error = e
            print(f"ERROR: Encoder thread failed: {e}")


def _write_bmp(filepath: str, data: bytearray, width: int, height: int) -> None:
    """Write raw BGRX pixel data as a BMP file.

    Cairo FORMAT_RGB24 stores pixels as 32-bit BGRX (little-endian), which maps
    directly to BMP's 32-bit BGR format with a top-down orientation.
    """
    pixel_data_size = width * height * 4
    file_size = 14 + 40 + pixel_data_size  # BMP header + DIB header + pixels

    with open(filepath, "wb") as f:
        # BMP file header (14 bytes)
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))  # File size
        f.write(struct.pack("<HH", 0, 0))  # Reserved
        f.write(struct.pack("<I", 14 + 40))  # Pixel data offset

        # DIB header - BITMAPINFOHEADER (40 bytes)
        f.write(struct.pack("<I", 40))  # Header size
        f.write(struct.pack("<i", width))  # Width
        f.write(struct.pack("<i", -height))  # Height (negative = top-down)
        f.write(struct.pack("<HH", 1, 32))  # Planes, bits per pixel
        f.write(struct.pack("<I", 0))  # Compression (BI_RGB)
        f.write(struct.pack("<I", pixel_data_size))  # Image size
        f.write(struct.pack("<ii", 2835, 2835))  # Resolution (72 DPI)
        f.write(struct.pack("<II", 0, 0))  # Colors used, important colors

        # Pixel data (BGRX from Cairo, direct write)
        f.write(data[:pixel_data_size])


class ConcatEncoder:
    """Encodes video by saving unique frames to temp files and using ffmpeg's
    concat demuxer. This avoids piping hundreds of GB of redundant raw video
    data to ffmpeg for mostly-static presentations."""

    def __init__(
        self, output: str, width: int, height: int, framerate: Fraction, codec: Codec
    ):
        self.output = output
        self.width = width
        self.height = height
        self.framerate = framerate
        self.codec = codec
        self.tmpdir = tempfile.mkdtemp(prefix="bbb-video-")
        self.segments: List[Tuple[str, int]] = []  # (filepath, frame_count)
        self.segment_index = 0
        self.current_segment_file: Optional[str] = None
        self.current_segment_frames = 0

    def put(self, data: bytearray) -> None:
        """Save a new unique frame. Finalizes the previous segment's duration."""
        # Finalize previous segment
        if self.current_segment_file is not None and self.current_segment_frames > 0:
            self.segments.append(
                (self.current_segment_file, self.current_segment_frames)
            )

        # Save new frame as BMP
        self.segment_index += 1
        filename = os.path.join(
            self.tmpdir, f"frame_{self.segment_index:06d}.bmp"
        )
        _write_bmp(filename, data, self.width, self.height)
        self.current_segment_file = filename
        self.current_segment_frames = 1

    def hold(self) -> None:
        """Extend the current segment by one frame (content unchanged)."""
        self.current_segment_frames += 1

    def join(self) -> None:
        """Finalize all segments and run ffmpeg with concat demuxer."""
        # Finalize the last segment
        if self.current_segment_file is not None and self.current_segment_frames > 0:
            self.segments.append(
                (self.current_segment_file, self.current_segment_frames)
            )
            self.current_segment_file = None

        if not self.segments:
            print("WARNING: No frames were captured, nothing to encode")
            self._cleanup()
            return

        total_unique = len(self.segments)
        total_frames = sum(count for _, count in self.segments)
        print(
            f"ConcatEncoder: {total_unique} unique frames out of "
            f"{total_frames} total ({100 * total_unique / max(total_frames, 1):.1f}%)"
        )

        try:
            self._encode()
        finally:
            self._cleanup()

    def cancel(self) -> None:
        """Clean up temp files on cancellation."""
        self._cleanup()

    def _encode(self) -> None:
        """Write concat file and run ffmpeg."""
        concat_file = os.path.join(self.tmpdir, "concat.txt")
        framerate_float = float(self.framerate)

        with open(concat_file, "w") as f:
            f.write("ffconcat version 1.0\n")
            for filepath, frame_count in self.segments:
                duration = frame_count / framerate_float
                f.write(f"file '{os.path.basename(filepath)}'\n")
                f.write(f"duration {duration:.6f}\n")
            # Repeat last file to avoid concat demuxer cutting it short
            if self.segments:
                f.write(f"file '{os.path.basename(self.segments[-1][0])}'\n")

        # Build codec options
        if self.codec == Codec.H264:
            codec_opts = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast"]
        elif self.codec == Codec.H264_MP4:
            codec_opts = [
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            ]
        elif self.codec == Codec.VP9:
            codec_opts = [
                "-c:v", "libvpx-vp9", "-deadline", "realtime",
                "-cpu-used", "8", "-lossless", "1", "-row-mt", "1",
            ]
        else:
            raise ValueError(f"Unsupported codec for ConcatEncoder: {self.codec}")

        if self.codec == Codec.H264_MP4:
            container_fmt = "mp4"
        else:
            container_fmt = "matroska"

        ffmpeg_cmdline = [
            "ffmpeg",
            "-y",
            "-nostats",
            "-v", "warning",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-pix_fmt", "yuv420p",
            "-r", str(self.framerate),
            *codec_opts,
            "-threads", "2",
            "-g", str(round(self.framerate) * 10),
            "-f", container_fmt,
            self.output,
        ]

        print(f"ConcatEncoder: running ffmpeg with concat demuxer...")
        ffmpeg = Popen(ffmpeg_cmdline, stdout=PIPE, close_fds=True)
        try:
            if ffmpeg.stdout is not None:
                ffmpeg.stdout.close()
            ffmpeg.wait(timeout=600)
        except Exception:
            try:
                ffmpeg.kill()
            except OSError:
                pass
            raise

        if ffmpeg.returncode != 0:
            raise CalledProcessError(
                returncode=ffmpeg.returncode, cmd=ffmpeg_cmdline
            )
        print("ConcatEncoder: encoding complete")

    def _cleanup(self) -> None:
        """Remove temporary directory and all frame files."""
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except Exception as e:
            print(f"WARNING: Failed to clean up temp directory {self.tmpdir}: {e}")


class Renderer:
    events: EventsInfo
    start_time: Fraction = Fraction(0)
    length: Fraction
    input: str
    output: str
    width: int
    height: int
    framerate: Fraction
    codec: Codec
    pod_id: str
    ignore_record_status: bool

    frame: int
    framestep: Fraction
    pts: Fraction
    recording: bool

    def __init__(
        self,
        events: EventsInfo,
        input: str,
        output: str,
        width: int,
        height: int,
        framerate: Fraction,
        codec: Codec,
        start_time: Fraction | None,
        end_time: Fraction | None,
        pod_id: str,
        ignore_record_status: bool,
    ):
        self.events = events
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.framerate = framerate
        self.codec = codec
        self.pod_id = pod_id
        self.ignore_record_status = ignore_record_status

        # Current video position state
        self.frame = 1
        self.framestep = 1 / framerate
        self.pts = Fraction(0)
        if self.ignore_record_status:
            print("\tRenderer: ignoring record status events")
            self.recording = True
        else:
            self.recording = False

        # Only the section of recording within the time range of start_time
        # through end_time will be included in the final video
        if events.length is None:
            raise ValueError("Recording length cannot be determined from events.xml")
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None and end_time < events.length:
            self.length = end_time
        else:
            self.length = events.length

        # Cairo rendering context
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.width, self.height)
        self.ctx = cairo.Context(self.surface)

        # Set up font rendering options
        font_options = cairo.FontOptions()
        font_options.set_antialias(cairo.ANTIALIAS_GRAY)
        font_options.set_hint_style(cairo.HINT_STYLE_NONE)
        self.ctx.set_font_options(font_options)

    def update_record(self, event: RecordEvent) -> bool:
        if self.ignore_record_status:
            return False

        if self.recording != event["status"]:
            self.recording = event["status"]
            print(f"\tRenderer: recording: {self.recording}")
            return True
        return False

    def render(self) -> None:
        cursor = CursorRenderer(
            self.ctx,
            Size(self.width, self.height),
            tldraw_whiteboard=self.events.tldraw_whiteboard,
        )
        presentation = PresentationRenderer(
            self.ctx,
            self.input,
            Size(self.width, self.height),
            self.events.hide_logo,
            tldraw_whiteboard=self.events.tldraw_whiteboard,
            bbb_version=self.events.bbb_version,
        )
        shapes = ShapesRenderer(self.ctx, presentation.transform)
        tldraw = TldrawRenderer(
            self.ctx, presentation.transform, self.events.bbb_version
        )

        # Use ConcatEncoder for encoded output (dramatically reduces ffmpeg work
        # by only encoding unique frames), fall back to pipe-based Encoder for
        # raw video output which needs every frame.
        use_concat = self.codec != Codec.RAW_VIDEO
        if use_concat:
            encoder = ConcatEncoder(
                self.output, self.width, self.height, self.framerate, self.codec
            )
        else:
            encoder = Encoder(
                self.output, self.width, self.height, self.framerate, self.codec
            )

        try:
            presentation_changed = True
            shapes_changed = False
            cursor_changed = False
            recording_changed = False
            while self.pts < self.length:
                event_ts = Fraction(0)
                while True:
                    if len(self.events.events) == 0:
                        break

                    event = self.events.events[0]
                    event_ts = event["timestamp"]
                    if event_ts > self.pts:
                        break

                    self.events.events.popleft()

                    name = event["name"]
                    print(f"{float(event['timestamp']):012.6f} {event['name']}")

                    # Skip events that are for a different pod
                    if name in ["pan_zoom", "presentation", "slide", "presenter"]:
                        pod_event = cast(PerPodEvent, event)
                        if pod_event["pod_id"] != self.pod_id:
                            print(f"\tskipping event for pod {pod_event['pod_id']}")
                            continue

                    tldraw.update(event)

                    if name == "cursor":
                        cursor.update_cursor(cast(events.CursorEvent, event))
                    elif name == "cursor_v2":
                        cursor.update_cursor_v2(
                            cast(events.WhiteboardCursorEvent, event)
                        )
                    elif name == "pan_zoom":
                        presentation.update_pan_zoom(cast(events.PanZoomEvent, event))
                    elif name == "presentation":
                        presentation_event = cast(events.PresentationEvent, event)
                        presentation.update_presentation(presentation_event)
                        shapes.update_presentation(presentation_event)
                        cursor.update_presentation(presentation_event)
                    elif name == "slide":
                        slide_event = cast(events.SlideEvent, event)
                        presentation.update_slide(slide_event)
                        shapes.update_slide(slide_event)
                        cursor.update_slide(slide_event)
                    elif name == "shape":
                        shape_event = cast(events.ShapeEvent, event)
                        shapes.update_shape(shape_event)
                        cursor.update_shape(shape_event)
                    elif name == "undo":
                        shapes.update_undo(cast(events.UndoEvent, event))
                    elif name == "clear":
                        shapes.update_clear(cast(events.ClearEvent, event))
                    elif name == "record":
                        if self.update_record(cast(events.RecordEvent, event)):
                            recording_changed = True
                    elif name == "presenter":
                        cursor.update_presenter(cast(events.PresenterEvent, event))
                    elif name == "join":
                        cursor.update_join(cast(events.JoinEvent, event))
                    elif name == "left":
                        cursor.update_left(cast(events.LeftEvent, event))
                    elif (
                        name == "tldraw.add_shape"
                        or name == "tldraw.delete_shape"
                        or name == "tldraw.camera"
                    ):
                        pass
                    else:
                        print("\tdon't know how to handle this event")

                if self.recording and self.pts >= self.start_time:
                    start_time = time.perf_counter_ns()

                    presentation_changed = presentation.finalize_frame()
                    shapes_changed = shapes.finalize_frame(presentation.transform)
                    tldraw_changed = tldraw.finalize_frame(presentation.transform)
                    cursor_changed = cursor.finalize_frame(presentation.transform)

                    if (
                        presentation_changed
                        or shapes_changed
                        or tldraw_changed
                        or cursor_changed
                    ):
                        # Composite the frame

                        # Base background color
                        ctx = self.ctx
                        ctx.save()
                        ctx.set_source_rgb(*DRAWING_BG)
                        ctx.paint()
                        ctx.restore()

                        # Presentation
                        presentation.render()

                        # Shapes
                        shapes.render()
                        tldraw.render()

                        # Cursor
                        cursor.render()

                        recording_changed = True

                    self.surface.flush()

                    if recording_changed:
                        end_time = time.perf_counter_ns()
                        print(
                            f"-- {float(self.pts):012.6f} frame {self.frame} ({(end_time - start_time) / 1000000:.3f}ms)"
                        )

                    if use_concat:
                        # ConcatEncoder: only save frames when content changed,
                        # otherwise just extend the current segment duration
                        concat_enc = cast(ConcatEncoder, encoder)
                        if recording_changed:
                            concat_enc.put(bytearray(self.surface.get_data()))
                        else:
                            concat_enc.hold()
                    else:
                        # Pipe-based Encoder: send every frame
                        encoder.put(bytearray(self.surface.get_data()))

                    presentation_changed = False
                    shapes_changed = False
                    cursor_changed = False
                    recording_changed = False

                self.frame += 1
                self.pts += self.framestep

            encoder.join()
        except EncoderError:
            # Encoder already failed; re-raise without trying to join again
            raise
        except BaseException:
            # Ensure ffmpeg process is killed and encoder thread is stopped
            # on any exception (rendering error, KeyboardInterrupt, etc.)
            encoder.cancel()
            raise
