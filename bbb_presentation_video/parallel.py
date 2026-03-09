# SPDX-FileCopyrightText: 2024 BigBlueButton Inc. and by respective authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import multiprocessing
import os
import shutil
import tempfile
from fractions import Fraction
from subprocess import CalledProcessError, run
from typing import Optional


def render_chunk(
    input_dir: str,
    output_path: str,
    width: int,
    height: int,
    framerate: Fraction,
    start_time: Fraction,
    end_time: Fraction,
    pod_id: str,
    ignore_record_status: bool,
    ffmpeg_threads: int,
    chunk_index: int,
) -> str:
    """Render a single time chunk of the video in a worker process.

    Each worker independently parses events, replays state up to start_time,
    then renders only frames within [start_time, end_time].
    """
    print(f"[Chunk {chunk_index}] Rendering {float(start_time):.3f}s - {float(end_time):.3f}s to {output_path}")

    from bbb_presentation_video.events import parse_events
    from bbb_presentation_video.renderer import Renderer

    events = parse_events(input_dir)

    renderer = Renderer(
        events,
        input_dir,
        output_path,
        width,
        height,
        framerate,
        start_time,
        end_time,
        pod_id,
        ignore_record_status,
        ffmpeg_threads=ffmpeg_threads,
    )
    renderer.render()

    print(f"[Chunk {chunk_index}] Done")
    return output_path


def concat_chunks(chunk_files: list, output_path: str) -> None:
    """Concatenate chunk MP4 files into a single output using ffmpeg concat demuxer."""
    temp_dir = os.path.dirname(chunk_files[0])
    list_path = os.path.join(temp_dir, "concat_list.txt")

    with open(list_path, "w") as f:
        for chunk_file in chunk_files:
            f.write(f"file '{chunk_file}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-nostats",
        "-v",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]

    print(f"Concatenating {len(chunk_files)} chunks into {output_path}...")
    result = run(cmd, check=False)
    if result.returncode != 0:
        raise CalledProcessError(returncode=result.returncode, cmd=cmd)
    print("Concatenation complete.")


def render_parallel(
    input_dir: str,
    output: str,
    width: int,
    height: int,
    framerate: Fraction,
    start_time: Optional[Fraction],
    end_time: Fraction,
    pod_id: str,
    ignore_record_status: bool,
    num_workers: int,
    ffmpeg_threads: int,
) -> None:
    """Render video in parallel by splitting into time-based chunks.

    Divides the video timeline into num_workers equal chunks, renders each
    in a separate process, then concatenates the results.
    """
    effective_start = start_time if start_time is not None else Fraction(0)
    effective_end = end_time
    total_duration = effective_end - effective_start

    if total_duration <= 0:
        raise ValueError(f"Invalid time range: {float(effective_start):.3f}s - {float(effective_end):.3f}s")

    # Cap worker count: at least 30 seconds per chunk.
    # Formula: max_workers = int(duration / 30) + 1
    max_useful_workers = int(total_duration / 30) + 1
    if num_workers > max_useful_workers:
        print(
            f"Video is {float(total_duration):.1f}s — capping workers "
            f"from {num_workers} to {max_useful_workers} "
            f"(~30s per chunk)"
        )
        num_workers = max_useful_workers

    # Compute chunk boundaries
    chunk_duration = total_duration / num_workers
    chunks = []
    for i in range(num_workers):
        cs = effective_start + chunk_duration * i
        ce = effective_start + chunk_duration * (i + 1)
        if i == num_workers - 1:
            ce = effective_end  # Avoid fractional drift on last chunk
        chunks.append((cs, ce))

    print(f"Parallel rendering: {num_workers} workers, {ffmpeg_threads} ffmpeg thread(s) each")
    print(f"Chunk duration: {float(chunk_duration):.3f}s")

    temp_dir = tempfile.mkdtemp(prefix="bpv_parallel_")
    try:
        # Build argument list for each worker
        chunk_files = []
        args_list = []
        for i, (cs, ce) in enumerate(chunks):
            chunk_output = os.path.join(temp_dir, f"chunk_{i:04d}.mp4")
            chunk_files.append(chunk_output)
            args_list.append((
                input_dir,
                chunk_output,
                width,
                height,
                framerate,
                cs,
                ce,
                pod_id,
                ignore_record_status,
                ffmpeg_threads,
                i,
            ))

        # Render all chunks in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(render_chunk, args_list)

        # Verify all chunks were created
        missing = [f for f in chunk_files if not os.path.exists(f)]
        if missing:
            raise RuntimeError(f"Missing chunk files: {missing}")

        # Concatenate chunks into final output
        concat_chunks(chunk_files, output)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
