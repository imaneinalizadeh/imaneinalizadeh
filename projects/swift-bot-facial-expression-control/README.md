# Adaptive Movement Control of Swift Bot via Facial Expression Recognition

A Python vision client (face detection → emotion classification → gaze
tracking) that drives a Java-based robot control server over a TCP
socket, so a Swift Bot moves forward/backward in response to the
operator's facial expression.

**Verified end-to-end in this repo**: the Java server was compiled
with `javac`, run as a real process, and driven both by raw socket
commands and by the actual Python client — see the command log in
[`docs/architecture.md`](docs/architecture.md) for what a real session
looks like.

## Pipeline

```
Webcam → Haar cascade face detection → emotion classifier → gaze tracker
                                              │
                                  emotion → movement command
                                              │
                                   TCP socket (see docs/protocol.md)
                                              │
                              Java RobotServer → CommandHandler → MotorController
```

Only **happiness** and **anger** map to movement (`ADVANCE` /
`RETREAT` respectively) per the original project spec — everything
else maps to `HOLD`, and any detection below a confidence threshold
also forces `HOLD` rather than acting on a shaky guess.

## Repository structure

```
swift-bot-facial-expression-control/
├── python-client/
│   ├── config.py              Camera/server config, emotion→command map
│   ├── face_detector.py       Haar cascade face detection
│   ├── emotion_detector.py    FER-library backend + heuristic fallback
│   ├── gaze_tracker.py        MediaPipe Face Mesh iris-based gaze direction
│   └── robot_client.py        Ties it together, sends commands over TCP
├── java-server/src/
│   ├── RobotServer.java       TCP server accept loop
│   ├── CommandHandler.java    Wire-protocol parsing and dispatch
│   └── MotorController.java   Hardware abstraction (stub — logs actions)
├── tests/
│   └── test_emotion_mapping.py
├── docs/
│   ├── architecture.md
│   └── protocol.md
└── requirements.txt
```

## Two emotion-detection backends

The original project used the `fer` library (TensorFlow-based). To
keep this repo runnable without a large ML dependency, `emotion_detector.py`
uses `fer` when installed and **transparently falls back** to a
Haar-cascade smile heuristic otherwise — happy/neutral only, clearly
documented as a fallback, not silently pretending to detect the full
emotion set. See [`docs/architecture.md`](docs/architecture.md).

## Running it

**Server** (needs a JDK):
```bash
cd java-server/src
javac *.java
java RobotServer 5050
```

**Client** (needs Python + the packages in `requirements.txt`):
```bash
cd python-client
pip install -r ../requirements.txt
python3 robot_client.py                       # webcam, live server required
python3 robot_client.py --video clip.mp4 --headless --dry-run   # test without a server
```

## Running the tests

```bash
python3 -m pytest tests/ -v
```

7 tests covering the emotion→command mapping logic, including the
confidence-threshold boundary and unmapped-emotion cases — these run
without a camera, MediaPipe model, or live server.

## Author

Iman Ein Alizadeh.
