# Drowsiness Detection System

This project implements a drowsiness detection system using multiple face detection models and facial landmark analysis. It leverages computer vision techniques to detect drowsiness based on Eye Aspect Ratio (EAR) and mouth opening distance.

## Prerequisites

- Python 3.6+
- Google Colab/Jupyter Notebook (recommended for GPU support)
- Basic understanding of computer vision concepts

## Installation

```bash
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5.git
!pip install -r yolov5/requirements.txt

# Install dependencies
!pip install dlib imutils opencv-python-headless numpy retinaface_pytorch pydub

# Download facial landmark predictor
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Download Caffe model files (SSD Face Detector)
!wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
!wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

## Model Approaches

### 1. Caffe SSD Face Detector with dlib Landmarks
**Files Required:**
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- `shape_predictor_68_face_landmarks.dat`

**Steps:**
1. Face detection using Caffe-based SSD
2. Landmark detection with dlib's 68-point predictor
3. EAR calculation for both eyes
4. Drowsiness determination (EAR threshold: 0.3)

### 2. YOLOv5 Face Detector with Facial Landmarks
**Files Required:**
- `shape_predictor_68_face_landmarks.dat`
- YOLOv5s pretrained weights (automatically downloaded)

**Steps:**
1. Face detection using YOLOv5
2. Landmark detection with dlib
3. EAR calculation + mouth opening detection
4. Combined drowsiness analysis (EAR threshold: 0.279)

### 3. RetinaFace with dlib Landmarks
**Files Required:**
- `shape_predictor_68_face_landmarks.dat`
- RetinaFace ResNet50 weights (automatically downloaded)

**Steps:**
1. Face detection using RetinaFace
2. Landmark detection with dlib
3. EAR-based drowsiness detection (threshold: 0.25)

## Project Structure
```
├── yolov5/                   # YOLOv5 implementation
├── Video1.mp4                # Input video file
├── shape_predictor_68_face_landmarks.dat  # Landmark model
├── deploy.prototxt           # Caffe model configuration
├── res10_300x300_ssd_iter_140000_fp16.caffemodel  # Caffe weights
└── Drowsiness_Detection.ipynb  # Main implementation notebook
```

## Usage

1. **Video Input Preparation:**
   - Place your video file in the project directory
   - Update `video_path` in code to match your filename

2. **Running Detection:**
```python
# For Caffe-based detection
main()

# For YOLOv5-based detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
processed_image = detect_drowsiness_and_draw(image, model, predictor_path)

# For RetinaFace detection
detect_faces(video_path)
```

## Key Features

- Multi-model face detection support
- Real-time EAR calculation
- Mouth opening detection
- Visual feedback with bounding boxes and landmarks
- Configurable thresholds for different scenarios

## Results

- **DROWSY** state indicated by red text when EAR < threshold
- **NOT DROWSY** state shown in green when eyes are open
- Bounding boxes around detected faces
- White landmarks highlighting eyes and mouth regions

## Threshold Tuning

| Model              | Default EAR Threshold | Mouth Open Threshold |
|--------------------|-----------------------|----------------------|
| Caffe + dlib       | 0.30                  | N/A                  |
| YOLOv5 + dlib      | 0.279                 | 15 pixels            |
| RetinaFace + dlib  | 0.25                  | N/A                  |

Adjust these values in code based on your specific requirements.

## Acknowledgements

- YOLOv5 by Ultralytics
- dlib's facial landmark predictor
- OpenCV's Caffe models
- RetinaFace PyTorch implementation
