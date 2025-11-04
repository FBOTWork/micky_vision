# Micky Vision Package

## Overview
This is a group of ROS2 packages responsible for vision features of [FBOT@Work](https://fbotwork.vercel.app/) industrial robot (MICKY) in RoboCup@Work league.

## Pre Requisites

- ROS2 Humble
- Python 3.10+
- Ubuntu 22.04
- Dependencies listed in package.xml and requirements.txt

## Installation

### 1. Clone Repository
```bash
cd ~/work_ws/src
git clone https://github.com/FBOTWork/micky_vision.git
```

### 2. Install Dependencies
```bash
cd ~/work_ws
sudo rosdep init  # Skip if already initialized
rosdep update
rosdep install --from-paths src --ignore-src -r -y
pip install -r src/micky_vision/requirements.txt
```

### 3. Build Package
```bash
cd ~/work_ws
colcon build --packages-select micky_vision
source install/setup.bash
```

### Topics

#### 📥 **Subscribed Topics**
- `/camera/camera/color/image_raw` (sensor_msgs/Image) - Input RGB camera images

#### 📤 **Published Topics**
- `/micky_recognition/tools_recognition` (sensor_msgs/Image) - Output image with bounding boxes and class labels
- `/micky_recognition/tools_classes` (std_msgs/String) - Detected classes and confidence percentages in text format

#### ⚙️ **Services**
- `/micky_recognition/recognition_start` (std_srvs/Empty) - Start object recognition
- `/micky_recognition/recognition_stop` (std_srvs/Empty) - Stop object recognition


## Usage

### Quick Start (All-in-One)
```bash
# Launch camera + recognition + visualization
ros2 launch micky_recognition tools_recognition.launch.py
```

### Customized Launch Options

#### Without Camera (if already running separately)
```bash
ros2 launch micky_recognition tools_recognition.launch.py use_camera:=false
```

#### Without RViz (headless mode)
```bash
ros2 launch micky_recognition tools_recognition.launch.py use_rviz:=false
```

#### Manual Control (don't start recognition automatically)
```bash
ros2 launch micky_recognition tools_recognition.launch.py start_on_init:=false
```

### Service Control

#### Start Recognition
```bash
ros2 service call /micky_recognition/recognition_start std_srvs/srv/Empty
```

#### Stop Recognition
```bash
ros2 service call /micky_recognition/recognition_stop std_srvs/srv/Empty
```

### Monitor Output

#### View Detected Classes (Text)
```bash
ros2 topic echo /micky_recognition/tools_classes
```

#### View Detection Image (Visual)
```bash
ros2 run rqt_image_view rqt_image_view /micky_recognition/tools_recognition
```

#### Monitor Topic Rates
```bash
ros2 topic hz /micky_recognition/tools_recognition
ros2 topic hz /micky_recognition/tools_classes
```

## Configuration

### Parameters File: `config/yolov8_tools_recognition.yaml`

```yaml
tools_recognition:
  ros__parameters:
    # Algorithm settings
    threshold: 0.6              # Detection confidence threshold (0.0-1.0)
    model_file: yolov8n.pt      # YOLOv8 model file in weights/ directory
    start_on_init: true         # Auto-start recognition on node initialization
    
    # Topics configuration
    subscribers:
      image_rgb:
        topic: /camera/camera/color/image_raw
        qos_profile: 10
    publishers:
      image:
        topic: /micky_recognition/tools_recognition
        qos_profile: 10
      debug:
        topic: /micky_recognition/tools_classes
        qos_profile: 10
    
    # Services configuration
    services:
      recognition:
        start: /micky_recognition/recognition_start
        stop: /micky_recognition/recognition_stop
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_camera` | `true` | Launch USB camera automatically |
| `use_rviz` | `true` | Launch RViz visualization |
| `start_on_init` | `true` | Start recognition on initialization |
| `threshold` | `0.6` | Detection confidence threshold |
| `model_file` | `yolov8n.pt` | YOLOv8 model filename |
| `image_topic` | `/camera/camera/color/image_raw` | Input camera topic |
| `detection_topic` | `/micky_recognition/tools_recognition` | Output image topic |
| `debug_topic` | `/micky_recognition/tools_classes` | Output classes topic |

## Output Examples

### Text Output (`/micky_recognition/tools_classes`)
```
person: 85.3% | tv: 72.1% | laptop: 68.9%
```

### Visual Output (`/micky_recognition/tools_recognition`)
- Original camera image with green bounding boxes
- Class names and confidence percentages overlaid on detections

### Debug Commands

```bash
# Check available topics
ros2 topic list | grep micky_recognition

# Monitor topic types
ros2 topic list -t

# Check service availability
ros2 service list | grep recognition

# View node information
ros2 node info /tools_recognition

# Check parameter values
ros2 param list /tools_recognition
```