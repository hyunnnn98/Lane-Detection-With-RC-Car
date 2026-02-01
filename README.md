# Lane Detection With RC Car

> Real-time lane detection system using OpenCV on RC Car

![demo](https://user-images.githubusercontent.com/52916934/147223614-796f9e78-8071-4bfd-a26a-a6a36dcbf66a.gif)

## Features
- Lane detection using Canny Edge & HoughLinesP
- Curved lane recognition with Sliding Window + Curve Fitting
- Bird's Eye View perspective transform

## Tech Stack
Python · OpenCV · Arduino

## Architecture
```
Camera → Python (OpenCV) → Arduino → Servo/ESC
   │          │                │
   │     Lane Detection        └── Steering Control
   │     Angle Calculation          Motor Control
   │          │
   └──────────┘
      Video Feed
```

## Hardware
| Component | Description |
|-----------|-------------|
| RC Car | Base platform with front-mounted camera |
| Arduino | Serial communication (9600 baud) |
| Servo Motor | Steering control (Pin 7, 60°~120°) |
| ESC | Motor speed control (Pin 9) |
| Camera | Calibrated camera with perspective warp |

## Pipeline
![pipeline](https://user-images.githubusercontent.com/52916934/147220970-418eb67a-6f4b-40e3-a997-52969b536316.png)

| Step | Description |
|------|-------------|
| 1. Original | Raw frame from RC car camera |
| 2. Grayscale | Convert to grayscale for processing |
| 3. Canny Edge | Detect edges using Canny algorithm |
| 4. ROI Mask | Apply region of interest mask |
| 5. HoughLinesP | Detect lane lines |
| 6. Sliding Window | Find lane pixels using sliding window |
| 7. Curve Fitting | Fit polynomial curve to lane |
| 8. Overlay | Project detected lane back to original |

## Getting Started
1. Ensure Python3 is installed
2. Clone the repository
3. Install dependencies: `pip install -r requirements.txt`
4. Add your video to `track/` folder
5. Customize ROI in `/utils_video.py` → `perspectiveWarp`

## Contributing
Issues and PRs are welcome.
