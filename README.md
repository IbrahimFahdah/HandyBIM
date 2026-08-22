# HandyBIM

Drag and drop an IFC/BIM model over your webcam feed and control it entirely with hand gestures — no mouse, no keyboard.

Built with [Three.js](https://threejs.org/), [That Open Components](https://github.com/ThatOpen/engine_components) (IFC loading via Fragments), and [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) for real-time hand and background segmentation.

## Demo

[▶ Watch the demo video](Demo.mp4)

Also posted on [LinkedIn](https://lnkd.in/p/d-XzYBht).

## Features

- **Drag & drop IFC loading** — drop a `.ifc` file straight onto the camera view.
- **Hand-gesture navigation** — pinch to rotate, two hands to zoom, open palm to pan, point to aim a cursor.
- **Live section cuts** — hold up two fingers to slice through the model; switch between horizontal/vertical planes with keys `1`/`2`/`3`.
- **Webcam background controls** — cycle between clear, blurred (person-aware segmentation), or hidden camera background.
- **Fist to reset, thumbs-up to pause** — quick gestures for common actions, mirrored by on-screen buttons and keyboard shortcuts (`Space`, `R`, `B`, `L`, `I`).

## Getting started

```bash
npm install
npm run dev
```

Open the local dev URL, allow camera access, then drag an `.ifc` file onto the window.

### Build

```bash
npm run build
npm run preview
```

## Controls

| Gesture | Action |
| --- | --- |
| Pinch + drag | Rotate model |
| Two hands | Zoom in/out |
| Open palm | Pan |
| Point | Move cursor |
| Pinch tap | Select |
| ✌️ Two fingers | Section cut |
| Thumbs up | Pause gestures |
| Fist | Reset view |

Keyboard equivalents: `Space` pause/resume, `R` reset, `B` cycle background, `L` toggle hand landmark overlay, `I` toggle the on-screen legend, `1`/`2`/`3` section orientation.
