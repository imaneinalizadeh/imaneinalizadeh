# Adaptive Movement Control of Swift Bot via Facial Expression Recognition

**BSc Final Year Project — Brunel University London**
Department of Computer Science · CS3072-CS3605 Final Year Project · Academic Year 2024-25

📄 [Read the full dissertation](./dissertation.pdf)

---

## Overview

An AI-driven human-robot interaction system that lets a Swift Bot robot respond autonomously to a user's facial expressions and gaze direction — replacing traditional controllers (joysticks, manual programming) with a more natural, accessible interface.

The system watches the user's face via webcam, classifies their emotional expression and gaze direction in real time, and streams that interpretation over the network to a robot controller that translates it into movement.

## Motivation

Traditional robot control interfaces can be inaccessible for users with physical disabilities, elderly users, or anyone unfamiliar with technical controls. This project explores whether facial expression and gaze alone are enough to drive intuitive, inclusive robot control — with applications in assistive robotics, education, and healthcare.

## How It Works

1. **Face & expression detection (Python client)** — a webcam feed is processed with OpenCV and Haar Cascade for face detection, and the `fer` library for facial emotion classification.
2. **Gaze tracking** — MediaPipe detects facial landmarks to determine eye position and gaze direction.
3. **Network transport** — the Python client streams detected emotion/gaze data over a TCP/UDP socket connection to a Java server.
4. **Movement mapping (Java server)** — the Java server receives the classified data and maps it to Swift Bot movement commands (e.g. specific emotions or gaze directions trigger specific motions).
5. **Evaluation** — the system was tested for detection accuracy, responsiveness, and usability (including a System Usability Scale survey and user feedback sessions) under varying lighting and demographic conditions.

## Tech Stack

| Layer | Technology |
|---|---|
| Facial expression recognition | Python, OpenCV, `fer` library |
| Face detection | Haar Cascade |
| Gaze tracking | MediaPipe |
| Robot control server | Java |
| Client–server transport | TCP / UDP sockets |
| Hardware platform | Swift Bot |

## Project Structure (as covered in the dissertation)

- **Background & Research** — prior work in FER, gaze tracking, human-robot interaction, and deep learning for emotion recognition
- **Methodology** — Agile development process, testing strategy (white/black/grey box), ethical considerations (privacy, informed consent, bias mitigation, accessibility)
- **Design** — system flowcharts, interaction flow, risk assessment, usability evaluation design
- **Implementation** — Python facial emotion detection pipeline, Java server/client architecture, Swift Bot movement action mapping
- **Testing & Evaluation** — unit and integration testing, edge case testing, real-world evaluation with user feedback
- **Conclusion** — objective fulfilment, chapter summary, future work

## Author

Iman Ein Alizadeh
