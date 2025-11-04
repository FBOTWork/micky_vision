#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
from ultralytics import YOLO
from std_msgs.msg import String
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

class ToolsRecognition(Node):
    def __init__(self) -> None:
        super().__init__('tools_recognition')

        self.model = None
        self.run = False
        self.bridge = CvBridge()
        
        self.declareParameters()
        self.readParameters()
        self.initRosComm()
        
        if self.start_on_init:
            self._startRecognition()

    def __del__(self):
        """Destructor to ensure clean shutdown"""
        try:
            if hasattr(self, 'run') and self.run:
                self._stopRecognition()
        except:
            pass

    def destroy_node(self):
        """Override destroy_node to ensure clean shutdown"""
        try:
            if hasattr(self, 'run') and self.run:
                self._stopRecognition()
        except:
            pass
        super().destroy_node()

    def initRosComm(self) -> None:
        """Initialize ROS2 communication"""
        # Publishers
        self.imagePublisher = self.create_publisher(
            Image, self.imagePublisherTopic, qos_profile=self.imageQosProfile)
        self.debugPublisher = self.create_publisher(
            String, self.debugPublisherTopic, qos_profile=self.debugQosProfile)
        
        # Services
        self.recognitionStartService = self.create_service(
            Empty, self.startRecognitionTopic, self.startRecognition)
        self.recognitionStopService = self.create_service(
            Empty, self.stopRecognitionTopic, self.stopRecognition)
        
        # Subscription
        self.imageSubscription = self.create_subscription(
            Image, self.imageSubscriberTopic, self.imageCallback, 
            qos_profile=self.imageSubscriberQosProfile)

    def loadModel(self) -> None:
        """Load YOLO model"""
        self.get_logger().info("=> Loading model")
        try:
            self.model = YOLO(self.modelFile)
            self.model.conf = self.threshold
            self.get_logger().info("=> Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None

    def unloadModel(self) -> None:
        """Unload YOLO model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        self.get_logger().info("=> Model unloaded")

    def _startRecognition(self):
        """Internal method to start recognition"""
        self.loadModel()
        if self.model is not None:
            self.run = True
            self.get_logger().info("Starting Object Recognition!!!")
        else:
            self.get_logger().error("Cannot start recognition - model loading failed")

    def _stopRecognition(self):
        """Internal method to stop recognition"""
        self.run = False
        self.unloadModel()
        self.get_logger().info("Stopping Object Recognition!!!")

    def startRecognition(self, req: Empty.Request, resp: Empty.Response):
        """Service callback to start recognition"""
        self._startRecognition()
        return resp

    def stopRecognition(self, req: Empty.Request, resp: Empty.Response):
        """Service callback to stop recognition"""
        self._stopRecognition()
        return resp

    def getClassName(self, classId: int) -> str:
        """Get human-readable class name from class ID using model's class names"""
        if self.model is not None and classId in self.model.names:
            return self.model.names[classId]
        return f"class_{classId}"

    def imageCallback(self, imageMsg: Image) -> None:
        """Main callback for processing images"""
        if not self.run:
            return
        
        if self.model is None:
            self.get_logger().error("Model is not loaded.")
            return

        if imageMsg is None:
            self.get_logger().error("Input image message is invalid.")
            return
        
        try:
            # Convert ROS image to OpenCV format
            cvImage = self.bridge.imgmsg_to_cv2(imageMsg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        try:
            # Run YOLO inference
            results = self.model(cvImage, verbose=False)
            if not results or len(results) == 0:
                return
                
            result = results[0]
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        # Process detections
        detectedClasses = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if box is None:
                    continue
                
                conf = float(box.conf)
                cls = int(box.cls)

                if conf < self.threshold:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class name and percentage
                className = self.getClassName(cls)
                percentage = conf * 100
                detectedClasses.append(f"{className}: {percentage:.1f}%")

                # Draw bounding box and label on image
                cv2.rectangle(cvImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cvImage, f"{className} {percentage:.1f}%", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish processed image with bounding boxes
        try:
            imageMsg_out = self.bridge.cv2_to_imgmsg(cvImage, "bgr8")
            imageMsg_out.header = imageMsg.header
            self.imagePublisher.publish(imageMsg_out)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

        # Publish detected classes as text
        classesMsg = String()
        if detectedClasses:
            classesMsg.data = " | ".join(detectedClasses)
        else:
            classesMsg.data = "No objects detected"
        self.debugPublisher.publish(classesMsg)

    def declareParameters(self) -> None:
        """Declare all ROS2 parameters"""
        # Publisher parameters
        self.declare_parameter("publishers.image.topic", "/micky_recognition/tools_recognition")
        self.declare_parameter("publishers.image.qos_profile", 10)
        self.declare_parameter("publishers.debug.topic", "/micky_recognition/tools_classes")
        self.declare_parameter("publishers.debug.qos_profile", 10)
        
        # Subscriber parameters
        self.declare_parameter("subscribers.image_rgb.topic", "camera/camera/color/image_raw")
        self.declare_parameter("subscribers.image_rgb.qos_profile", 10)
        
        # Service parameters
        self.declare_parameter("services.recognition.start", "/micky_recognition/recognition_start")
        self.declare_parameter("services.recognition.stop", "/micky_recognition/recognition_stop")
        
        # Algorithm parameters (values come from YAML config file)
        self.declare_parameter("threshold", 0.5)  # Default fallback - YAML will override
        self.declare_parameter("model_file", "yolov8n.pt")  # Default fallback - YAML will override  
        self.declare_parameter("start_on_init", True)  # Default fallback - YAML will override

    def readParameters(self) -> None:
        """Read all ROS2 parameters"""
        # Publisher parameters
        self.imagePublisherTopic = self.get_parameter("publishers.image.topic").value
        self.imageQosProfile = self.get_parameter("publishers.image.qos_profile").value
        self.debugPublisherTopic = self.get_parameter("publishers.debug.topic").value
        self.debugQosProfile = self.get_parameter("publishers.debug.qos_profile").value
        
        # Subscriber parameters
        self.imageSubscriberTopic = self.get_parameter("subscribers.image_rgb.topic").value
        self.imageSubscriberQosProfile = self.get_parameter("subscribers.image_rgb.qos_profile").value
        
        # Service parameters
        self.startRecognitionTopic = self.get_parameter("services.recognition.start").value
        self.stopRecognitionTopic = self.get_parameter("services.recognition.stop").value
        
        # Algorithm parameters
        self.threshold = self.get_parameter("threshold").value
        self.start_on_init = self.get_parameter("start_on_init").value
        
        # Model file path
        modelFileName = self.get_parameter("model_file").value
        try:
            packageShareDirectory = get_package_share_directory('micky_recognition')
            self.modelFile = f"{packageShareDirectory}/weights/{modelFileName}"
        except Exception:
            self.modelFile = f"weights/{modelFileName}"
            self.get_logger().warn(f"Using relative path for model: {self.modelFile}")
        
        # Log important parameters
        self.get_logger().info(f"Threshold: {self.threshold}")
        self.get_logger().info(f"Model file: {self.modelFile}")
        self.get_logger().info(f"Start on init: {self.start_on_init}")
        self.get_logger().info(f"Image topic: {self.imageSubscriberTopic}")
        self.get_logger().info(f"Output image topic: {self.imagePublisherTopic}")
        self.get_logger().info(f"Debug topic: {self.debugPublisherTopic}")

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = ToolsRecognition()
        print("Tools Recognition started. Press Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Tools Recognition shutdown requested (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        print("[INFO] Tools Recognition stopped cleanly.")

if __name__ == '__main__':
    main()