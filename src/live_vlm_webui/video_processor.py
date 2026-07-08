# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Video Track Processor
Handles video frames, adds text overlays, and manages VLM processing
"""

import asyncio
import base64
import cv2
import numpy as np
from PIL import Image
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from typing import Optional
import logging
import time
import av

from .vlm_service import VLMService

# Enable swscaler warnings to track hardware acceleration status
# TODO: Implement hardware-accelerated color space conversion on Jetson using NVMM/VPI
av.logging.set_level(av.logging.WARNING)

logger = logging.getLogger(__name__)


class VideoProcessorTrack(VideoStreamTrack):
    """
    Video track that receives frames, sends them to VLM for analysis,
    and overlays responses on the video before sending back
    """

    # Class variable for frame processing interval (can be updated dynamically)
    process_every_n_frames = 30
    # Max allowed latency before dropping frames (in seconds, 0 = disabled)
    max_frame_latency = 0.0

    # VLM trigger mode: "interval" (fixed frame count, default) or "yolo"
    # (fire VLM only when YOLO's detected object classes change)
    trigger_mode = "interval"
    yolo_detector = None  # YoloTrigger instance, set when trigger_mode == "yolo"
    yolo_detect_every_n_frames = 5

    def __init__(self, track: VideoStreamTrack, vlm_service: VLMService, text_callback=None):
        super().__init__()
        self.track = track
        self.vlm_service = vlm_service
        self.text_callback = text_callback  # Callback to send text updates
        self.last_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.first_frame_pts = None  # Track first frame PTS to calculate relative time
        self.first_frame_time = None  # Wall clock time of first frame
        self.frame_time_base = None  # Time base for PTS conversion (e.g., 1/90000)
        self._yolo_current_classes = []  # Most recently observed YOLO detection classes
        self._yolo_detections = []  # Most recently observed YOLO boxes/masks, for overlay drawing
        self._last_seen_response = None  # Tracks response text to detect new VLM results

    @staticmethod
    def _make_thumbnail(img_bgr: np.ndarray, max_width: int = 160) -> Optional[str]:
        """Encode a small JPEG data URL thumbnail from a BGR frame, for the result history log."""
        try:
            height, width = img_bgr.shape[:2]
            scale = max_width / width
            resized = cv2.resize(
                img_bgr, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA
            )
            ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                return None
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to build thumbnail: {e}")
            return None

    async def recv(self):
        """
        Receive frame from input track, process it, and return with text overlay
        """
        try:
            # Get frame from incoming track
            frame = await self.track.recv()

            # Initialize timing on first frame
            if self.first_frame_pts is None and frame.pts is not None:
                self.first_frame_pts = frame.pts
                self.first_frame_time = time.time()
                # Store time_base for PTS conversion (e.g., 1/90000 for 90kHz clock)
                self.frame_time_base = float(frame.time_base)
                logger.info(
                    f"Latency tracking initialized: PTS={frame.pts}, time_base={frame.time_base} ({self.frame_time_base}s per tick)"
                )

            # Calculate actual frame age (latency) using PTS and time_base
            # Note: Some streams (like RTSP) may not have PTS set, so skip latency checks
            frame_latency = 0.0
            if frame.pts is not None and self.first_frame_pts is not None:
                # PTS is in time_base units, convert to seconds: pts * time_base
                frame_time_offset = (frame.pts - self.first_frame_pts) * self.frame_time_base
                expected_wall_time = self.first_frame_time + frame_time_offset
                current_time = time.time()
                frame_latency = current_time - expected_wall_time

            # Check for accumulated latency and drop old frames if needed (only if max_latency > 0)
            max_latency = self.__class__.max_frame_latency
            if max_latency > 0 and frame_latency > max_latency and frame.pts is not None:
                logger.warning(
                    f"Frame is {frame_latency:.2f}s behind, dropping frames (threshold: {max_latency}s)"
                )

                # Drop frames until we get a fresh one
                dropped_count = 0
                while frame_latency > max_latency:
                    self.dropped_frames += 1
                    dropped_count += 1

                    # Get next frame
                    frame = await self.track.recv()

                    # Recalculate latency for new frame (using time_base for correct conversion)
                    if frame.pts is not None and self.first_frame_pts is not None:
                        frame_time_offset = (
                            frame.pts - self.first_frame_pts
                        ) * self.frame_time_base
                        expected_wall_time = self.first_frame_time + frame_time_offset
                        frame_latency = time.time() - expected_wall_time
                    else:
                        # If PTS becomes unavailable, stop dropping frames
                        break

                    # Prevent infinite loop
                    if dropped_count > 100:
                        logger.error(
                            f"Dropped {dropped_count} frames, but still behind. Resetting timing."
                        )
                        if frame.pts is not None:
                            self.first_frame_pts = frame.pts
                            self.first_frame_time = time.time()
                            self.frame_time_base = float(frame.time_base)
                        break

                if dropped_count > 0:
                    logger.info(
                        f"Dropped {dropped_count} frames, now at {frame_latency:.2f}s latency"
                    )

            # Increment frame counter
            self.frame_count += 1

            # Only convert to numpy when needed (for VLM processing or first frame)
            # This avoids expensive CPU color conversion on every frame
            mode = self.__class__.trigger_mode
            detector = self.__class__.yolo_detector
            if mode == "yolo" and detector is not None:
                interval = max(1, self.__class__.yolo_detect_every_n_frames)
            else:
                interval = self.__class__.process_every_n_frames
            need_conversion = (self.frame_count % interval == 0) or (self.frame_count == 1)
            trigger_flash = False

            if need_conversion:
                t1 = time.time()
                # Convert to numpy array (expensive: YUV→BGR color conversion on CPU)
                img = frame.to_ndarray(format="bgr24")
                t2 = time.time()
                self.last_frame = img.copy()
                t3 = time.time()

                # Log timing every 100 frames to identify bottlenecks
                if self.frame_count % 100 == 0:
                    logger.info(
                        f"Frame conversion times: to_ndarray={1000*(t2-t1):.1f}ms, copy={1000*(t3-t2):.1f}ms"
                    )

                # Log first frame
                if self.frame_count == 1:
                    logger.info(f"First frame received: {img.shape}")

                # Decide whether this cycle should fire the VLM
                should_send_to_vlm = False
                if self.frame_count % interval == 0:
                    if mode == "yolo" and detector is not None:
                        if self.vlm_service.is_processing:
                            # Skip YOLO inference entirely while the VLM call is in
                            # flight - both compete for the same GPU, and letting YOLO
                            # keep firing fragments/stalls the (much more important)
                            # VLM inference, sometimes by tens of seconds
                            logger.debug(
                                f"Frame {self.frame_count}: VLM busy, skipping YOLO detection"
                            )
                        else:
                            # Run YOLO off the event loop thread - inference blocks
                            triggered, classes, detections = await asyncio.to_thread(
                                detector.detect, img
                            )
                            self._yolo_current_classes = sorted(classes)
                            self._yolo_detections = detections
                            if triggered:
                                should_send_to_vlm = True
                                trigger_flash = True
                                logger.info(
                                    f"Frame {self.frame_count}: YOLO trigger fired, "
                                    f"classes={sorted(classes) or ['none']}"
                                )
                    else:
                        should_send_to_vlm = True

                if should_send_to_vlm:
                    if self.vlm_service.is_processing:
                        # VLM is still busy with a previous request - skip the costly
                        # color conversion / JPEG encode, it would just be discarded
                        logger.debug(f"Frame {self.frame_count}: VLM busy, skipping encode")
                    else:
                        # Convert to PIL Image for VLM
                        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        # Snapshot a small thumbnail of this exact frame for the result history log
                        thumbnail_b64 = self._make_thumbnail(img)
                        # Fire and forget - don't wait for result
                        asyncio.create_task(
                            self.vlm_service.process_frame(pil_img, thumbnail_b64=thumbnail_b64)
                        )
                        if mode != "yolo":
                            logger.info(
                                f"Frame {self.frame_count}: Sending to VLM (interval={interval})"
                            )

            # Get current response (may be old if VLM is still processing)
            response, is_processing = self.vlm_service.get_current_response()

            # Get metrics
            metrics = self.vlm_service.get_metrics()

            # Detect a new result landing (independent of trigger mode) and attach
            # its paired thumbnail + timestamp for the result history log
            if response != self._last_seen_response:
                self._last_seen_response = response
                result_thumbnail = self.vlm_service.get_current_thumbnail()
                if result_thumbnail:
                    metrics["result_thumbnail"] = result_thumbnail
                    metrics["result_ts"] = self.vlm_service.get_current_result_ts()

            if mode == "yolo" and detector is not None:
                metrics["trigger"] = {
                    "mode": "yolo",
                    "classes": self._yolo_current_classes,
                    "detections": self._yolo_detections,
                    "just_triggered": trigger_flash,
                }

            # Send text update via callback (for WebSocket)
            if self.text_callback:
                self.text_callback(response, metrics)

            # Return original frame directly - zero-copy passthrough!
            # This avoids expensive BGR→YUV conversion
            return frame

        except MediaStreamError:
            # Track ended (user stopped, tab closed, etc.) — normal, not an error
            logger.debug("Video track ended")
            raise
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            raise

    def _add_text_overlay(self, img: np.ndarray, text: str, status: str = "") -> np.ndarray:
        """
        Add text overlay to image

        Args:
            img: Input image (BGR format)
            text: Text to overlay (VLM response)
            status: Optional status text

        Returns:
            Image with text overlay
        """
        img_copy = img.copy()
        height, width = img_copy.shape[:2]

        # Prepare text
        full_text = f"{text} {status}" if status else text

        # Text wrapping - split long captions
        max_chars_per_line = 60
        words = full_text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background
        padding = 10
        line_height = 30

        # Calculate total height needed
        total_text_height = len(lines) * line_height + 2 * padding

        # Create semi-transparent overlay at bottom
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, height - total_text_height), (width, height), bg_color, -1)

        # Blend overlay with original image
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

        # Add text lines
        y_position = height - total_text_height + padding + line_height
        for line in lines:
            cv2.putText(
                img_copy,
                line,
                (padding, y_position),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            y_position += line_height

        return img_copy
