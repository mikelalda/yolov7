#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import sys
import pathlib

from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox
from models.experimental import attempt_load

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from detection_msgs.msg import DetectionArray, Detection
from cv_bridge import CvBridge

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter('weights', '"./models/yolov7.pt"')
        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('det_conf_thres', 0.8)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('device', "0")
        self.declare_parameter('img_size', 640)
        self.declare_parameter('inf_image', False)
        self.declare_parameter('inf_image_topic', '/object_detection/inf_image')
        self.declare_parameter('bbox_topic', '/object_detection/bboxes')
        self.declare_parameter('frequency', 40)

        self.weights = self.get_parameter('weights').value
        self.topic = self.get_parameter('topic').value
        self.det_conf_thres = self.get_parameter('det_conf_thres').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.device = self.get_parameter('device').value
        self.img_size = self.get_parameter('img_size').value
        self.inf_image = self.get_parameter('inf_image').value
        self.inf_image_topic = self.get_parameter('inf_left_image_topic').value
        self.bbox_topic = self.get_parameter('bbox_left_topic').value
        self.frequency = self.get_parameter('frequency').value

        self.get_logger().info("weights: %s, topic: %s, det_conf_thres: %s, conf_thres: %s, iou_thres: %s, device: %s, img_size: %s, inf_image: %s, inf_image_topic: %s, bbox_topic: %s, frequency: %s" %
                    (str(self.weights),
                        str(self.topic),
                        str(self.det_conf_thres),
                        str(self.conf_thres),
                        str(self.iou_thres),
                        str(self.device),
                        str(self.img_size),
                        str(self.inf_image),
                        str(self.inf_image_topic),
                        str(self.bbox_topic),
                        str(self.frequency)))


        # Camera info and frames
        self.rgb_image = None

        # Flags
        self.camera_RGB = False

        # Timer callback
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for Classes
        qos_pub = QoSProfile(depth=1) 
        qos_pub.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.infer_pub = self.create_publisher(Image, self.inf_image_topic, qos_profile=qos_pub)
        self.bb_pub = self.create_publisher(DetectionArray, self.bbox_topic, qos_profile=qos_pub)
 
        # OpenCv - ROS 2 Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        qos_sub = QoSProfile(depth=1) 
        qos_sub.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.sub = self.create_subscription(Image, self.topic, qos_profile=qos_sub, callback=self.img_callback)

        # Initialize YOLOv7
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def img_callback(self, data):
        """
        Subscription to the compressed RGB camera topic.

        """
        self.rgb_image = self.bridge.imgmsg_to_cv2(data)
        self.camera_RGB = True
        #self.YOLOv7_detect()

    def YOLOv7_detect(self):
        """ Preform object detection with YOLOv7"""
        im0 = self.rgb_image.copy()

        img = letterbox(self.rgb_image, self.img_size, self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()

        # Initialize array of bounding boxes
        det_array = DetectionArray()

        # Process detections   
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if conf > 0.8: # Limit confidence threshold to 80% for all classes
                        # Draw a boundary box around each object
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                        # Save bounding box to publish later
                        det = Detection()
                        det.class_name = label
                        det.class_id = cls
                        det.bbox.center.position.x = float(xyxy[0])-float(xyxy[2])
                        det.bbox.center.position.y = float(xyxy[1])-float(xyxy[3])
                        det.bbox.center.theta = 0.0
                        det.bbox.size_x = abs(det.bbox.center.position.x/2)
                        det.bbox.size_y = abs(det.bbox.center.position.y/2)
                        if self.is_seg:
                            # Segmentation
                            det.mask.size_x = abs(det.bbox.center.position.x/2)
                            det.mask.size_y = abs(det.bbox.center.position.y/2)
                            det.mask.data = xyxy[4:]
                        det_array.boxes.append(det)

                        
            #cv2.imshow("YOLOv7 Object detection result RGB", cv2.cvtColor(cv2.resize(im0, None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR)) 
            #cv2.waitKey()  

        # Publish bounding boxes
        self.bb_pub.publish(det_array)
        
        # Publish output ROS 2 image
        if self.inf_image == True:
            out_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(cv2.resize(im0, None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR))
            # Publish output local image test    
            #out_img = self.bridge.cv2_to_imgmsg(cv2.resize(im0, None, fx=1.5, fy=1.5)
            self.infer_pub.publish(out_img)

    def timer_callback(self):
        # If the ROS 2 topic is not avaiable uncomment the follwing lines to test with a local image
        #self.rgb_image = cv2.imread("/root/yolov7/inference/images/bus.jpg")
        #self.camera_RGB = True
        if self.camera_RGB == True:
            self.YOLOv7_detect()

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()