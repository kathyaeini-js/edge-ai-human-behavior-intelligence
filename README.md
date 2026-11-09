# Edge AI-Powered Multimodal Human Behavior Intelligence Platform

## 1. Project Overview

The Edge AI-Powered Multimodal Human Behavior Intelligence Platform is an AI-based system designed to analyze human activities and posture from video input.

The current implementation uses computer vision and pose estimation to detect people, track their body keypoints, and identify basic posture patterns.

The system processes video files, generates annotated video output, and stores detected behavior events in JSONL format. A FastAPI backend provides access to the generated events, while a React frontend provides the dashboard interface.

## 2. Objectives

- Detect humans in video input.
- Estimate human body keypoints using pose estimation.
- Analyze basic body posture using geometric calculations.
- Identify upright and possible bending or leaning postures.
- Generate an annotated output video.
- Store behavior analysis results in JSONL format.
- Display processed information through a web dashboard.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | YOLO |
| Pose Estimation | YOLO Pose |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Deep Learning | PyTorch |
| Backend API | FastAPI |
| API Server | Uvicorn |
| Frontend | React.js |
| Frontend Development Server | Vite |
| Data Storage | JSONL |

## 4. Project Structure

```text
edge_behavior_ai/
│
├── backend/
│   ├── behavior.py
│   ├── process_video.py
│   └── api.py
│
├── data/
│   ├── input.mp4
│   └── outputs/
│       ├── annotated.mp4
│       └── events.jsonl
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── requirements.txt
└── README.md
```

## 5. System Requirements

- Windows 10 or Windows 11
- Python 3.12
- Node.js and npm
- Visual Studio Code
- Internet connection for initial AI model downloads
- Minimum 8 GB RAM recommended

The current execution instructions use CPU processing. A CUDA-compatible GPU is not required.

## 6. Installation and Setup

### Step 1: Open the project

Open the project folder in Visual Studio Code.

Open the integrated PowerShell terminal.

Navigate to the project directory:

```powershell
cd C:\Users\kathy\Downloads\edge_behavior_ai_starter\edge_behavior_ai
```

### Step 2: Activate the virtual environment

If the virtual environment already exists in the parent directory:

```powershell
& ..\.venv\Scripts\Activate.ps1
```

If it does not exist, create it from the parent directory:

```powershell
cd ..
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd .\edge_behavior_ai
```

### Step 3: Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy fastapi uvicorn
```

Install the remaining project dependencies if required:

```powershell
python -m pip install -r requirements.txt
```

Verify the important packages:

```powershell
python -c "import cv2, numpy, torch; from ultralytics import YOLO; import fastapi, uvicorn; print('All imports successful')"
```

### Step 4: Verify the input video

Place the input video inside the `data` folder.

The expected path is:

```text
data/input.mp4
```

Check whether the video exists:

```powershell
Test-Path "data\input.mp4"
```

The expected result is:

```text
True
```

### Step 5: Process the input video

Run the following command from the main project directory:

```powershell
python -m backend.process_video --input data/input.mp4 --output data/outputs/annotated.mp4 --device cpu
```

During the first execution, Ultralytics may download the YOLO detection and pose-estimation model weights.

Wait until the video processing is completed.

The generated files should be:

```text
data/outputs/annotated.mp4
data/outputs/events.jsonl
```

The annotated video contains the detection and pose-estimation visualizations. The JSONL file contains the generated behavior events.

## 7. Start the Backend Server

Open a **new terminal** in Visual Studio Code.

Navigate to the main project directory:

```powershell
cd C:\Users\kathy\Downloads\edge_behavior_ai_starter\edge_behavior_ai
```

Activate the virtual environment:

```powershell
& ..\.venv\Scripts\Activate.ps1
```

Start the FastAPI server:

```powershell
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Backend API:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

Retrieve behavior events:

```text
http://127.0.0.1:8000/events
```

Keep this terminal running.

## 8. Start the Frontend

Open another new terminal.

Navigate to the frontend directory:

```powershell
cd C:\Users\kathy\Downloads\edge_behavior_ai_starter\edge_behavior_ai\frontend
```

Install frontend dependencies:

```powershell
npm install
```

Start the React development server:

```powershell
npm run dev
```

Open the URL displayed in the terminal. By default, it should be:

```text
http://localhost:5173/
```

Keep this terminal running.

## 9. Complete Execution Workflow

Execute the project in the following order:

1. Activate the Python virtual environment.
2. Verify that `data/input.mp4` exists.
3. Run the video-processing module.
4. Check that `annotated.mp4` and `events.jsonl` are generated.
5. Start the FastAPI backend.
6. Start the React frontend.
7. Open the dashboard in your browser.

## 10. Troubleshooting

### Problem 1: ModuleNotFoundError: No module named 'backend'

**Reason:** The Python script was executed directly instead of as a module.

**Solution:** Execute it from the main project directory:

```powershell
python -m backend.process_video --input data/input.mp4 --output data/outputs/annotated.mp4 --device cpu
```

### Problem 2: Input video not found

Check the input file:

```powershell
Test-Path "data\input.mp4"
```

Make sure the video is inside the `data` directory and its filename is `input.mp4`.

### Problem 3: Frontend directory not found

Make sure you are in the main project folder before navigating:

```powershell
cd .\frontend
```

### Problem 4: npm command not found

Check the Node.js and npm installations:

```powershell
node -v
npm -v
```

Install Node.js if it is unavailable.

### Problem 5: Backend API is not responding

Check whether the FastAPI server is running and open:

```text
http://127.0.0.1:8000/health
```

### Problem 6: No behavior events are displayed

Run the video-processing command first and verify that:

```text
data/outputs/events.jsonl
```

exists. Then restart the backend from the main project directory.

## 11. Expected Outputs

| Output | Description |
|---|---|
| annotated.mp4 | Video with human detection and pose annotations |
| events.jsonl | Generated behavior analysis events |
| FastAPI server | Provides access to the processed events |
| React dashboard | Frontend interface for displaying project information |

## 12. Future Enhancements

- Facial emotion recognition.
- Speech and voice characteristic analysis.
- Multi-person behavior analysis improvements.
- Real-time camera processing.
- Advanced behavioral classification.
- Explainable AI-based performance reports.
- Edge-device deployment and optimization.

## 13. Conclusion

The project demonstrates an edge-oriented computer vision workflow for human detection, pose estimation, and basic posture analysis. By integrating YOLO, OpenCV, FastAPI, and React, it provides a foundation for developing a more comprehensive multimodal human behavior intelligence platform.