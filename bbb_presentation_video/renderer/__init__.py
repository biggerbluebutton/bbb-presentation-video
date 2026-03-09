# SPDX-FileCopyrightText: 2024 BigBlueButton Inc. and by respective authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
import time
from fractions import Fraction
from queue import Queue
from subprocess import PIPE, CalledProcessError, Popen
from typing import Optional, Tuple, cast

# Cursor-only frames are throttled to this interval to reduce encoder load
CURSOR_ONLY_INTERVAL = Fraction(1, 4)  # 4fps max for cursor-only changes

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

# Timebase for VFR output: 1ms precision
TIMEBASE_DEN = 1000


class Encoder:
    """Encodes VFR raw BGR0 frames to H.264/MP4 via NUT muxer piped to ffmpeg."""

    queue: "Queue[Optional[Tuple[bytearray, int]]]"
    ret_queue: "Queue[bytearray]"

    def __init__(self, output: str, width: int, height: int, framerate: Fraction = Fraction(24), ffmpeg_threads: int = 0) -> None:
        self.output = output
        self.width = width
        self.height = height
        self.framerate = framerate
        self.ffmpeg_threads = ffmpeg_threads

        self.queue: Queue[Optional[Tuple[bytearray, int]]] = Queue()
        self.ret_queue: Queue[bytearray] = Queue()
        for _ in range(3):
            self.ret_queue.put(bytearray(width * height * 4))

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def put(self, data: bytearray, pts_ms: int) -> None:
        """Queue a frame for encoding.

        Args:
            data: Raw BGR0 frame data
            pts_ms: Presentation timestamp in milliseconds
        """
        buf = self.ret_queue.get()
        buf[:] = data
        self.queue.put((buf, pts_ms))

    def join(self) -> None:
        # Sentinel value to tell the writing thread to exit
        self.queue.put(None)
        self.thread.join()

    def run(self) -> None:
        """Pipe VFR NUT stream to ffmpeg for H.264/MP4 encoding."""
        ffmpeg_cmdline = [
            "ffmpeg",
            "-y",
            "-nostats",
            "-v",
            "warning",
            "-f",
            "nut",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-bf",
            "0",
            "-fps_mode",
            "cfr",
            "-r",
            str(float(self.framerate)),
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-threads",
            str(self.ffmpeg_threads),
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            self.output,
        ]

        ffmpeg = Popen(ffmpeg_cmdline, stdin=PIPE, stdout=PIPE, close_fds=True)
        assert ffmpeg.stdout is not None and ffmpeg.stdin is not None
        ffmpeg.stdout.close()

        muxer = NutMuxer(ffmpeg.stdin, self.width, self.height, 1, TIMEBASE_DEN)
        muxer.write_headers()

        while True:
            item = self.queue.get()
            if item is None:
                break

            buf, pts_ms = item
            muxer.write_frame(bytes(buf), pts_ms)
            self.ret_queue.put(buf)

        muxer.close()
        ffmpeg.stdin.close()
        ffmpeg.wait()

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
        start_time: Fraction | None,
        end_time: Fraction | None,
        pod_id: str,
        ignore_record_status: bool,
        ffmpeg_threads: int = 0,
    ):
        self.events = events
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.framerate = framerate
        self.pod_id = pod_id
        self.ignore_record_status = ignore_record_status
        self.ffmpeg_threads = ffmpeg_threads

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

        encoder = Encoder(self.output, self.width, self.height, self.framerate, self.ffmpeg_threads)

        presentation_changed = True
        shapes_changed = False
        cursor_changed = False
        recording_changed = False
        first_frame = True
        # Cached background (presentation + shapes + tldraw, without cursor)
        bg_data: Optional[bytearray] = None
        # Throttle cursor-only frame output
        last_cursor_only_pts = Fraction(-1)

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
                    cursor.update_cursor_v2(cast(events.WhiteboardCursorEvent, event))
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

                bg_changed = (
                    presentation_changed or shapes_changed or tldraw_changed
                )

                if bg_changed or cursor_changed:
                    ctx = self.ctx

                    if bg_changed:
                        # Repaint background layers to main surface
                        ctx.save()
                        ctx.set_source_rgb(*DRAWING_BG)
                        ctx.paint()
                        ctx.restore()

                        presentation.render()
                        shapes.render()
                        tldraw.render()

                        # Snapshot the background for cursor-only redraws
                        self.surface.flush()
                        bg_data = bytearray(self.surface.get_data())
                    elif bg_data is not None:
                        # Restore background from snapshot (cursor-only change)
                        buf = memoryview(self.surface.get_data())
                        buf[:] = bg_data
                        self.surface.mark_dirty()

                    # Add cursor on top
                    cursor.render()

                    recording_changed = True

                if recording_changed or first_frame:
                    # Throttle cursor-only frames to reduce encoder load
                    cursor_only = cursor_changed and not bg_changed
                    if (
                        cursor_only
                        and not first_frame
                        and (self.pts - last_cursor_only_pts)
                        < CURSOR_ONLY_INTERVAL
                    ):
                        # Skip this cursor-only frame (too soon)
                        recording_changed = False
                    else:
                        self.surface.flush()

                        end_time = time.perf_counter_ns()
                        print(
                            f"-- {float(self.pts):012.6f} frame {self.frame} ({(end_time - start_time) / 1000000:.3f}ms)"
                        )

                        # Output frame with VFR timestamp
                        pts_ms = int((self.pts - self.start_time) * TIMEBASE_DEN)
                        encoder.put(bytearray(self.surface.get_data()), pts_ms)
                        first_frame = False

                        if cursor_only:
                            last_cursor_only_pts = self.pts

                presentation_changed = False
                shapes_changed = False
                cursor_changed = False
                recording_changed = False

            self.frame += 1
            self.pts += self.framestep

        # Output a final frame at the end timestamp to set correct video duration
        end_pts_ms = int((self.length - self.start_time) * TIMEBASE_DEN)
        encoder.put(bytearray(self.surface.get_data()), end_pts_ms)

        encoder.join()
