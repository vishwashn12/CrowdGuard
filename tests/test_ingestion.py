"""Frame buffer policies and the dual-rate producer."""

from __future__ import annotations

import time
from queue import Empty

import cv2
import numpy as np
import pytest

from conftest import ROOT  # noqa: F401  (ensures sys.path setup)

from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.ingestion.producer import FrameProducer
from crowdguard.ingestion.video_source import VideoFileSource, build_source
from crowdguard.utils.exceptions import FrameBufferError, VideoSourceError


@pytest.fixture
def sample_video(tmp_path):
    """A short synthetic clip: 40 frames, 20 fps, 64x48."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")

    for index in range(40):
        frame = np.full((48, 64, 3), index * 5 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("could not write a test video")
    return path


def make_packet(frame_id: int = 1) -> FramePacket:
    return FramePacket(
        camera_id="cam",
        frame_id=frame_id,
        source_frame_index=frame_id,
        video_timestamp=frame_id / 25.0,
        wall_clock_timestamp=time.time(),
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        source_fps=25.0,
    )


class TestFrameBuffer:
    def test_round_trip(self):
        buffer = FrameBuffer(capacity=4)
        assert buffer.put(make_packet(1))
        assert buffer.get(block=False).frame_id == 1

    def test_fifo_order(self):
        buffer = FrameBuffer(capacity=8)
        for frame_id in range(1, 5):
            buffer.put(make_packet(frame_id))
        assert [buffer.get(block=False).frame_id for _ in range(4)] == [1, 2, 3, 4]

    def test_oldest_policy_keeps_the_newest_frame(self):
        """Right for density: a stale count is worth nothing."""
        buffer = FrameBuffer(capacity=2, drop_policy="oldest")
        for frame_id in (1, 2, 3):
            assert buffer.put(make_packet(frame_id))

        assert [buffer.get(block=False).frame_id for _ in range(2)] == [2, 3]
        assert buffer.frames_dropped == 1

    def test_newest_policy_preserves_the_existing_queue(self):
        """Right for anomaly: trajectory continuity beats recency."""
        buffer = FrameBuffer(capacity=2, drop_policy="newest")
        assert buffer.put(make_packet(1))
        assert buffer.put(make_packet(2))
        assert buffer.put(make_packet(3)) is False  # refused

        assert [buffer.get(block=False).frame_id for _ in range(2)] == [1, 2]
        assert buffer.frames_dropped == 1

    def test_get_on_empty_non_blocking_raises(self):
        with pytest.raises(Empty):
            FrameBuffer(capacity=2).get(block=False)

    def test_get_times_out(self):
        started = time.monotonic()
        with pytest.raises(Empty):
            FrameBuffer(capacity=2).get(block=True, timeout=0.05)
        assert time.monotonic() - started >= 0.04

    def test_closed_buffer_refuses_writes(self):
        buffer = FrameBuffer(capacity=2)
        buffer.close()
        with pytest.raises(FrameBufferError):
            buffer.put(make_packet())

    def test_closed_buffer_drains_then_raises(self):
        buffer = FrameBuffer(capacity=2)
        buffer.put(make_packet(1))
        buffer.close()

        assert buffer.get(block=False).frame_id == 1
        with pytest.raises(Empty):
            buffer.get(block=False)

    def test_rejects_non_packets(self):
        with pytest.raises(TypeError):
            FrameBuffer(capacity=2).put("not a packet")  # type: ignore[arg-type]

    def test_try_get_returns_none_instead_of_raising(self):
        assert FrameBuffer(capacity=2).try_get() is None

    def test_invalid_configuration_rejected(self):
        with pytest.raises(ValueError):
            FrameBuffer(capacity=0)
        with pytest.raises(ValueError):
            FrameBuffer(capacity=2, drop_policy="nonsense")

    def test_stats_report_counters(self):
        buffer = FrameBuffer(capacity=2, drop_policy="oldest", name="test")
        for frame_id in (1, 2, 3):
            buffer.put(make_packet(frame_id))
        buffer.get(block=False)

        stats = buffer.stats()
        assert stats["offered"] == 3
        assert stats["dropped"] == 1
        assert stats["consumed"] == 1
        assert stats["name"] == "test"


class TestVideoSource:
    def test_reads_metadata(self, sample_video):
        source = VideoFileSource(sample_video)
        assert source.source_fps == pytest.approx(20.0, abs=1.0)
        assert source.resolution == (64, 48)
        assert source.frame_count > 0
        assert not source.is_live

    def test_reads_frames_in_order(self, sample_video):
        with VideoFileSource(sample_video) as source:
            first, index0 = source.read_next()
            second, index1 = source.read_next()

        assert first is not None and second is not None
        assert (index0, index1) == (0, 1)

    def test_signals_end_of_stream(self, sample_video):
        with VideoFileSource(sample_video) as source:
            count = 0
            while True:
                frame, _ = source.read_next()
                if frame is None:
                    break
                count += 1
                if count > 200:
                    pytest.fail("source never reported EOF")
        assert count > 0

    def test_rewind_restarts_the_file(self, sample_video):
        with VideoFileSource(sample_video) as source:
            source.read_next()
            source.read_next()
            assert source.rewind()
            _, index = source.read_next()
        assert index == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(VideoSourceError, match="does not exist"):
            VideoFileSource(tmp_path / "nope.mp4").open()

    def test_builder_dispatches_on_type(self, sample_video):
        assert build_source("video", str(sample_video)).source_type == "video"
        assert build_source("rtsp", "rtsp://host/stream").source_type == "rtsp"
        assert build_source("webcam", "0").source_type == "webcam"

        with pytest.raises(VideoSourceError, match="Unsupported"):
            build_source("carrier-pigeon", "x")

    def test_rtsp_credentials_are_not_logged(self):
        source = build_source("rtsp", "rtsp://admin:secret@10.0.0.5:554/s1")
        described = source.metadata["target"]
        assert "secret" not in described
        assert "redacted" in described


class TestDualRateProducer:
    """The core of the dual-rate design: one read, two independently decimated paths."""

    def _run(self, video, density_fps=5.0, anomaly_stride=1, timeout=8.0):
        density = FrameBuffer(capacity=256, drop_policy="oldest", name="density")
        anomaly = FrameBuffer(capacity=256, drop_policy="newest", name="anomaly")

        producer = FrameProducer(
            camera_id="cam_test",
            source=VideoFileSource(video),
            density_buffer=density,
            anomaly_buffer=anomaly,
            density_fps=density_fps,
            anomaly_stride=anomaly_stride,
            real_time_mode=False,   # run as fast as possible in tests
            loop_source=False,
        )
        producer.start()

        deadline = time.monotonic() + timeout
        while not producer.complete and time.monotonic() < deadline:
            time.sleep(0.02)
        producer.stop()

        assert producer.complete, "producer did not finish within the timeout"
        return producer, density, anomaly

    def test_anomaly_path_gets_every_frame_at_stride_one(self, sample_video):
        producer, _, anomaly = self._run(sample_video, anomaly_stride=1)
        assert producer.frames_read > 0
        assert producer.frames_to_anomaly == producer.frames_read

    def test_density_path_is_decimated(self, sample_video):
        producer, _, _ = self._run(sample_video, density_fps=5.0, anomaly_stride=1)
        # 20 fps source at 5 fps target means stride 4.
        assert producer.density_stride == 4
        assert producer.frames_to_density < producer.frames_to_anomaly

    def test_anomaly_stride_decimates_the_pose_path(self, sample_video):
        producer, _, _ = self._run(sample_video, anomaly_stride=2)
        assert producer.frames_to_anomaly == pytest.approx(producer.frames_read / 2, abs=1)

    def test_both_paths_share_one_frame_id_clock(self, sample_video):
        """The synchroniser compares the two paths, so their clocks must agree."""
        _, density, anomaly = self._run(sample_video, anomaly_stride=1)

        density_ids = set()
        while True:
            packet = density.try_get()
            if packet is None:
                break
            density_ids.add(packet.frame_id)

        anomaly_ids = set()
        while True:
            packet = anomaly.try_get()
            if packet is None:
                break
            anomaly_ids.add(packet.frame_id)

        assert density_ids
        assert anomaly_ids
        # Every density frame was also seen by the pose path at stride 1, so the
        # ids must be a subset rather than two unrelated numbering schemes.
        assert density_ids <= anomaly_ids

    def test_packets_carry_the_strides_in_use(self, sample_video):
        _, _, anomaly = self._run(sample_video, density_fps=5.0, anomaly_stride=2)
        packet = anomaly.try_get()
        assert packet is not None
        assert packet.anomaly_stride == 2
        assert packet.density_stride == 4
        assert packet.anomaly_fps == pytest.approx(10.0, abs=0.5)

    def test_frame_ids_increase_monotonically(self, sample_video):
        _, _, anomaly = self._run(sample_video, anomaly_stride=1)
        ids = []
        while True:
            packet = anomaly.try_get()
            if packet is None:
                break
            ids.append(packet.frame_id)
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))

    def test_stats_are_reported(self, sample_video):
        producer, _, _ = self._run(sample_video)
        stats = producer.stats()
        assert stats["camera_id"] == "cam_test"
        assert stats["frames_read"] > 0
        assert stats["complete"] is True

    def test_missing_source_does_not_crash_the_thread(self, tmp_path):
        producer = FrameProducer(
            camera_id="cam_bad",
            source=VideoFileSource(tmp_path / "missing.mp4"),
            density_buffer=FrameBuffer(capacity=4),
            anomaly_buffer=FrameBuffer(capacity=4),
            real_time_mode=False,
        )
        producer.start()

        deadline = time.monotonic() + 3.0
        while not producer.complete and time.monotonic() < deadline:
            time.sleep(0.02)

        assert producer.complete
        assert producer.frames_read == 0

    def test_real_time_pacing_tracks_the_source_rate(self, sample_video):
        """Regression: pacing must apply to every frame READ, not every frame emitted.

        Pacing only the emitted frames lets the loop sprint through the ones the
        stride skips, so a clip plays back stride-times too fast and the models
        are handed far more frames per second than the source contains. That
        shows up as permanent pose-branch saturation that looks like a GPU
        limit rather than a timing bug.
        """
        density = FrameBuffer(capacity=512, drop_policy="oldest")
        anomaly = FrameBuffer(capacity=512, drop_policy="newest")

        producer = FrameProducer(
            camera_id="cam_pace",
            source=VideoFileSource(sample_video),
            density_buffer=density,
            anomaly_buffer=anomaly,
            density_fps=5.0,
            anomaly_stride=6,        # large stride: the bug scaled with this
            real_time_mode=True,     # the whole point of the test
            loop_source=True,        # keep reading for the full window
        )
        producer.start()
        time.sleep(2.0)
        producer.stop()

        source_fps = producer.source_fps
        assert producer.read_fps > 0

        # Allow generous slack for scheduler jitter and decode cost, but catch
        # the stride-times-too-fast failure, which was 6x here.
        assert producer.read_fps < source_fps * 2.0, (
            f"read {producer.read_fps:.1f} fps from a {source_fps:.1f} fps source: "
            f"pacing is not applied to skipped frames"
        )

    def test_display_sink_gets_every_frame_on_schedule(self, sample_video):
        """Playback receives every frame, stamped t0 + k/fps, not only analysed ones."""
        received: list[tuple[int, float]] = []

        producer = FrameProducer(
            camera_id="cam_disp",
            source=VideoFileSource(sample_video),
            density_buffer=FrameBuffer(capacity=256),
            anomaly_buffer=FrameBuffer(capacity=256),
            anomaly_stride=4,
            real_time_mode=True,
            display_sink=lambda cam, fid, sched, img: received.append((fid, sched)),
        )
        t0 = time.monotonic() + 0.2
        producer.start(t0=t0)
        time.sleep(1.2)
        producer.stop()

        assert len(received) >= 10
        assert producer.frames_to_anomaly < len(received)  # display sees more
        interval = 1.0 / producer.source_fps
        assert received[0][1] == pytest.approx(t0, abs=1e-6)
        for k, (_, scheduled) in enumerate(received):
            assert scheduled == pytest.approx(t0 + k * interval, abs=1e-6)

    def test_producers_sharing_t0_start_in_lockstep(self, sample_video):
        """The Start button's guarantee: every gate's frame 0 falls due together."""
        first_seen: dict[str, float] = {}

        def sink(cam, fid, sched, img):
            first_seen.setdefault(cam, time.monotonic())

        producers = [
            FrameProducer(
                camera_id=f"cam_{i}",
                source=VideoFileSource(sample_video),
                density_buffer=FrameBuffer(capacity=256),
                anomaly_buffer=FrameBuffer(capacity=256),
                real_time_mode=True,
                display_sink=sink,
            )
            for i in range(3)
        ]
        t0 = time.monotonic() + 0.4
        for producer in producers:
            producer.start(t0=t0)
        time.sleep(0.8)
        for producer in producers:
            producer.stop()

        assert len(first_seen) == 3
        # Nobody released early, and all three within a few ms of each other.
        assert min(first_seen.values()) >= t0 - 0.005
        assert max(first_seen.values()) - min(first_seen.values()) < 0.03

    def test_producer_can_be_restarted(self, sample_video):
        """Stop then Start must replay from the beginning with a fresh clock."""
        producer = FrameProducer(
            camera_id="cam_restart",
            source=VideoFileSource(sample_video),
            density_buffer=FrameBuffer(capacity=256),
            anomaly_buffer=FrameBuffer(capacity=256),
            real_time_mode=False,
        )
        producer.start()
        time.sleep(0.5)
        producer.stop()
        first_ids_end = producer._frame_id
        assert producer.frames_read > 0

        producer.start()
        time.sleep(0.5)
        producer.stop()

        assert producer.frames_read > 0
        # frame_id stays monotonic across sessions, for the synchroniser.
        assert producer._frame_id > first_ids_end

    def test_refused_anomaly_frames_are_counted(self, sample_video):
        """Saturation must be visible, not silent."""
        density = FrameBuffer(capacity=256, drop_policy="oldest")
        anomaly = FrameBuffer(capacity=2, drop_policy="newest")  # deliberately tiny

        producer = FrameProducer(
            camera_id="cam_test",
            source=VideoFileSource(sample_video),
            density_buffer=density,
            anomaly_buffer=anomaly,
            anomaly_stride=1,
            real_time_mode=False,
        )
        producer.start()

        deadline = time.monotonic() + 8.0
        while not producer.complete and time.monotonic() < deadline:
            time.sleep(0.02)
        producer.stop()

        assert producer.anomaly_frames_rejected > 0
