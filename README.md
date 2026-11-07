# FormVision — AI Exercise Technique Analyzer

FormVision is a local Python application that processes squat videos with
MediaPipe Pose Landmarker and OpenCV and exports an annotated video with a
visible skeleton overlay. Pose inference happens locally; no paid or external
AI service is used.

## Requirements

- Windows 10/11, 64-bit
- Python 3.10–3.13 (the MediaPipe 0.10.35 Windows wheel was verified for
  Python 3.13)
- A local Pose Landmarker `.task` model file

## Windows setup

From PowerShell, open the `FormVision` folder and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, either enable local scripts for
your user or run the virtual-environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the official **Pose Landmarker Lite** model file
[`pose_landmarker_lite.task`](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task)
and place it at `models\pose_landmarker_lite.task`. The model is not bundled or
downloaded automatically. Alternatively, set `FORMVISION_MODEL_PATH` to the
absolute path of another compatible Pose Landmarker `.task` model.

## Run the application

```powershell
streamlit run app.py
```

Upload an MP4, MOV, AVI, MKV, or WEBM video in the browser. The app writes an
MP4 with the skeleton overlay to `outputs\` and offers it for download.
Supported **output** containers are MP4 and AVI. For a command-line call from
the `FormVision` folder:

```powershell
python -c "from src.pose_estimator import PoseEstimator; from src.video_processor import process_video; e=PoseEstimator(); print(process_video(r'sample_videos\squat.mp4', r'outputs\squat_annotated.mp4', e)); e.close()"
```

Input must be readable by the OpenCV build and have valid frame-rate and
dimension metadata. When pose detection fails on a frame, that frame is copied
without a skeleton; landmarks are never synthesized.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover landmark extraction (including missing confidence fields and
no-detection frames) and video-processing failures, timestamps, output
metadata, and resource cleanup. Inference is mocked; an end-to-end video test
requires the model file and a sample video.
