# SPDX-FileCopyrightText: 2024 BigBlueButton Inc. and by respective authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
import time
from enum import Enum
from fractions import Fraction
from queue import Empty, Queue
from subprocess import PIPE, CalledProcessError, Popen
from typing import Optional, Union, cast

import cairo

from bbb_presentation_video import events
from bbb_presentation_video.events import EventsInfo, PerPodEvent, RecordEvent
from bbb_presentation_video.events.helpers import Color, Size
from bbb_presentation_video.renderer.cursor import CursorRenderer
from bbb_presentation_video.renderer.nut import NutMuxer
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


class VfrEncoder:
    """Encoder using NUT muxer for variable frame rate output.

    Only changed frames are piped to ffmpeg with correct VFR timestamps,
    eliminating duplicate frame processing entirely.
    """

    queue: "Queue[Optional[tuple[int, bytearray]]]"
    ret_queue: "Queue[bytearray]"

    def __init__(
        self, output: str, width: int, height: int, framerate: Fraction, codec: Codec
    ):
        self.output = output
        self.width = width
        self.height = height
        self.framerate = framerate
        self.codec = codec

        self.queue: Queue[Optional[tuple[int, bytearray]]] = Queue()
        self.ret_queue: Queue[bytearray] = Queue()
        for _ in range(3):
            self.ret_queue.put(bytearray(width * height * 4))

        self.error: Optional[BaseException] = None
        self.ffmpeg_process: Optional[Popen] = None

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def put(self, data: bytearray, pts: int) -> None:
        """Queue a frame for encoding at the given PTS."""
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
        self.queue.put((pts, buf))

    def join(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            print("WARNING: VfrEncoder thread did not exit within timeout, killing ffmpeg")
            self._kill_ffmpeg()
            self.thread.join(timeout=10)
        if self.error is not None:
            raise EncoderError(
                f"Encoder thread failed: {self.error}"
            ) from self.error

    def cancel(self) -> None:
        """Cancel encoding: drain queues, signal thread to stop, kill ffmpeg."""
        self.queue.put(None)
        self._kill_ffmpeg()
        self.thread.join(timeout=10)

    def _kill_ffmpeg(self) -> None:
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.kill()
            except OSError:
                pass

    def run(self) -> None:
        try:
            self._output_ffmpeg()
        except Exception as e:
            self.error = e
            print(f"ERROR: VfrEncoder thread failed: {e}")

    def _output_ffmpeg(self) -> None:
        if self.codec == Codec.H264_MP4:
            codec_opts = [
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "20",
            ]
            container_fmt = "mp4"
        elif self.codec == Codec.H264:
            codec_opts = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast"]
            container_fmt = "matroska"
        else:
            codec_opts = [
                "-c:v", "libvpx-vp9",
                "-deadline", "realtime",
                "-cpu-used", "8",
                "-lossless", "1",
                "-row-mt", "1",
            ]
            container_fmt = "matroska"

        ffmpeg_cmdline = [
            "ffmpeg",
            "-y",
            "-nostats",
            "-v", "warning",
            "-f", "nut",
            "-i", "pipe:0",
            "-pix_fmt", "yuv420p",
            *codec_opts,
            "-vsync", "vfr",
            "-threads", "2",
            "-movflags", "+faststart",
            "-f", container_fmt,
            self.output,
        ]

        ffmpeg = Popen(ffmpeg_cmdline, stdin=PIPE, stdout=PIPE, close_fds=True)
        self.ffmpeg_process = ffmpeg
        assert ffmpeg.stdout is not None and ffmpeg.stdin is not None
        ffmpeg.stdout.close()

        # Use framerate denominator/numerator as time base so PTS units
        # correspond exactly to frame ticks (1 PTS unit = 1 frame period)
        time_base_num = self.framerate.denominator
        time_base_den = self.framerate.numerator
        muxer = NutMuxer(
            ffmpeg.stdin, self.width, self.height,
            time_base_num, time_base_den,
        )

        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break

                pts, buf = item
                muxer.write_frame(pts, buf)
                self.ret_queue.put(buf)
        except BrokenPipeError:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                _, buf = item
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

        use_vfr = self.codec == Codec.H264_MP4
        encoder: Union[Encoder, VfrEncoder]
        if use_vfr:
            encoder = VfrEncoder(
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
            # PTS counter for VFR encoder (in frame tick units)
            pts_counter = 0
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

                    frame_changed = (
                        presentation_changed
                        or shapes_changed
                        or tldraw_changed
                        or cursor_changed
                    )

                    if frame_changed:
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

                    # Output frame(s) to encoder
                    if use_vfr:
                        # VFR: only pipe changed frames with their PTS
                        if frame_changed or recording_changed:
                            assert isinstance(encoder, VfrEncoder)
                            encoder.put(
                                bytearray(self.surface.get_data()), pts_counter
                            )
                    else:
                        # CFR: pipe every frame (mpdecimate handles dedup)
                        assert isinstance(encoder, Encoder)
                        encoder.put(bytearray(self.surface.get_data()))

                    presentation_changed = False
                    shapes_changed = False
                    cursor_changed = False
                    recording_changed = False
                    pts_counter += 1

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
