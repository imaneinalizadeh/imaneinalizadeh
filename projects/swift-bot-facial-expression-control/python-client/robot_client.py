"""
robot_client.py

Ties together face detection, emotion classification, and gaze
tracking, maps the result to a movement command, and sends that
command as a newline-terminated string over a TCP socket to the Java
RobotServer.

Protocol (see docs/protocol.md): one command per line, one of
ADVANCE / RETREAT / HOLD, optionally followed by a gaze tag:
    "ADVANCE|centre\n"
The Java server parses on '|' and ignores the gaze tag if it doesn't
recognise it — this keeps the wire format forwards-compatible.

Usage:
    python3 robot_client.py --video path/to.mp4 --headless
    python3 robot_client.py                        # webcam, needs a live server
"""

import argparse
import socket
import time

import cv2

import config
from emotion_detector import EmotionDetector
from face_detector import FaceDetector
from gaze_tracker import GazeTracker


def emotion_to_command(label, confidence):
    if confidence < config.EMOTION_CONFIDENCE_THRESHOLD:
        return config.DEFAULT_COMMAND
    return config.EMOTION_COMMAND_MAP.get(label, config.DEFAULT_COMMAND)


class RobotClient:
    def __init__(self, host=config.SERVER_HOST, port=config.SERVER_PORT, dry_run=False):
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.sock = None

    def connect(self):
        if self.dry_run:
            return
        self.sock = socket.create_connection(
            (self.host, self.port), timeout=config.SOCKET_TIMEOUT_SECONDS
        )

    def send_command(self, command, gaze_tag="centre"):
        line = f"{command}|{gaze_tag}\n"
        if self.dry_run or self.sock is None:
            print(f"[dry-run] would send: {line.strip()}")
            return
        self.sock.sendall(line.encode("utf-8"))

    def close(self):
        if self.sock is not None:
            self.sock.close()


def run(video_source, headless, dry_run, max_frames=None):
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    gaze_tracker = GazeTracker()
    client = RobotClient(dry_run=dry_run)

    if not dry_run:
        client.connect()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    frame_idx = 0
    command_log = []
    last_command = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        face_box = face_detector.largest_face(frame_bgr)
        label, confidence = emotion_detector.detect(frame_bgr, face_box)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gaze = gaze_tracker.estimate_direction(frame_rgb) or "centre"

        command = emotion_to_command(label, confidence)

        # Only send when the command actually changes, to avoid
        # flooding the socket with an identical command every frame.
        if command != last_command:
            client.send_command(command, gaze)
            last_command = command

        command_log.append({
            "frame": frame_idx,
            "emotion": label,
            "confidence": round(confidence, 2),
            "gaze": gaze,
            "command": command,
        })

        if not headless:
            cv2.putText(frame_bgr, f"{label} ({confidence:.2f}) -> {command}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Swift Bot Control", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    gaze_tracker.close()
    client.close()
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    return command_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of sending over socket")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    source = args.video if isinstance(args.video, str) else 0
    log = run(source, args.headless, args.dry_run, args.max_frames)
    print(f"Processed {len(log)} frames.")


if __name__ == "__main__":
    main()
