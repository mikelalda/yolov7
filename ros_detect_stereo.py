#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import sys
import pathlib

# add yolov7 submodule to path
FILE_ABS_DIR = pathlib.Path(__file__).absolute().parent
YOLOV7_ROOT = (FILE_ABS_DIR / 'yolov7').as_posix()
if YOLOV7_ROOT not in sys.path:
    print("Adding yolov7 to path: %s" % YOLOV7_ROOT)
    sys.path.append(YOLOV7_ROOT)

from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox
from models.experimental import attempt_load

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray
from cv_bridge import CvBridge

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter('weights', '"./models/yolov7.pt"')
        self.declare_parameter('topic_left', '/camera/left/image_raw')
        self.declare_parameter('topic_right', '/camera/right/image_raw')
        self.declare_parameter('det_conf_thres', 0.8)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('device', "0")
        self.declare_parameter('img_size', 640)
        self.declare_parameter('inf_image', False)
        self.declare_parameter('inf_left_image_topic', '/object_detection/left_inf_image')
        self.declare_parameter('bbox_left_topic', '/object_detection/left_bboxes')
        self.declare_parameter('inf_image_topic', '/object_detection/right_inf_image')
        self.declare_parameter('bbox_right_topic', '/object_detection/right_bboxes')
        self.declare_parameter('frequency', 40)

        self.weights = self.get_parameter('weights').value
        self.topic_left = self.get_parameter('topic_left').value
        self.topic_right = self.get_parameter('topic_right').value
        self.det_conf_thres = self.get_parameter('det_conf_thres').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.device = self.get_parameter('device').value
        self.img_size = self.get_parameter('img_size').value
        self.inf_image = self.get_parameter('inf_image').value
        self.inf_left_image_topic = self.get_parameter('inf_left_image_topic').value
        self.bbox_left_topic = self.get_parameter('bbox_left_topic').value
        self.inf_right_image_topic = self.get_parameter('inf_right_image_topic').value
        self.bbox_right_topic = self.get_parameter('bbox_right_topic').value
        self.frequency = self.get_parameter('frequency').value

        self.get_logger().info("weights: %s, topic_left: %s, topic_right: %s, det_conf_thres: %s, conf_thres: %s, iou_thres: %s, device: %s, img_size: %s, inf_image: %s, inf_left_image_topic: %s, bbox_left_topic: %s, inf_right_image_topic: %s, bbox_right_topic: %s, frequency: %s" %
                    (str(self.weights),
                        str(self.topic_left),
                        str(self.topic_right),
                        str(self.det_conf_thres),
                        str(self.conf_thres),
                        str(self.iou_thres),
                        str(self.device),
                        str(self.img_size),
                        str(self.inf_image),
                        str(self.inf_left_image_topic),
                        str(self.bbox_left_topic),
                        str(self.inf_right_image_topic),
                        str(self.bbox_right_topic),
                        str(self.frequency)))

        # Camera info and frames
        self.left_image = None
        self.right_image = None

        # Flags
        self.left_RGB = False
        self.right_RGB = False

        # Timer callback
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for Classes
        qos_pub = QoSProfile(depth=1) 
        qos_pub.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.infer_left_pub = self.create_publisher(Image, self.inf_left_image_topic, qos_profile=qos_pub)
        self.infer_right_pub = self.create_publisher(Image, self.inf_right_image_topic, qos_profile=qos_pub)
        self.bb_left_pub = self.create_publisher(BoundingBox2DArray, self.bbox_left_topic, qos_profile=qos_pub)
        self.bb_right_pub = self.create_publisher(BoundingBox2DArray, self.bbox_right_topic, qos_profile=qos_pub)
 
        # OpenCv - ROS 2 Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        qos_sub = QoSProfile(depth=1) 
        qos_sub.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.left_sub = self.create_subscription(Image, self.topic_left, qos_profile=qos_sub, callback=self.left_callback)
        self.right_sub = self.create_subscription(Image, self.topic_right, qos_profile=qos_sub, callback=self.right_callback)

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

    def left_callback(self, data):
        """
        Subscription to the compressed RGB camera topic.
        """
        self.left_image = self.bridge.imgmsg_to_cv2(data)
        self.left_RGB = True
    
    def right_callback(self, data):
        """
        Subscription to the compressed RGB camera topic.
        """
        self.right_image = self.bridge.imgmsg_to_cv2(data)
        self.right_RGB = True
        #self.YOLOv7_detect()

    def YOLOv7_detect(self):
        """ 
        Preform object detection with YOLOv7
        """
        im0s = [self.left_image.copy(), self.right_image.copy]
        
        self.left_RGB = True
        self.right_RGB = True

        img = letterbox(self.left_image, self.img_size, self.stride)[0]
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
        # t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        # t3 = time_synchronized()

        # Process detections
        bbox_array = []
        inf_images = []
        for i, det in enumerate(pred):  # detections per image
            # Initialize array of bounding boxes
            bb_array = BoundingBox2DArray()
            im0 = im0s[i].copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if conf > self.det_conf_thres: # Limit confidence threshold to 80% for all classes
                        # Draw a boundary box around each object
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                        # Save bounding box to publish later
                        bb = BoundingBox2D()
                        bb.center.position.x = float(xyxy[0])-float(xyxy[2])
                        bb.center.position.y = float(xyxy[1])-float(xyxy[3])
                        bb.center.theta = 0.0
                        bb.size_x = abs(bb.center.position.x/2)
                        bb.size_y = abs(bb.center.position.y/2)
                        bb_array.boxes.append(bb)
            bbox_array.append(bb_array)
            inf_images.append(im0)
            #cv2.imshow("YOLOv7 Object detection result RGB", cv2.cvtColor(cv2.resize(im0, None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR)) 
            #cv2.waitKey()  

        # t4 = time_synchronized()

        # Publish bounding boxes
        self.bb_left_pub.publish(bbox_array[0])
        self.bb_right_pub.publish(bbox_array[1])
        
        # Publish output ROS 2 image
        if self.inf_image:
            out_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(cv2.resize(inf_images[0], None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR))
            self.infer_left_pub.publish(out_img)
            out_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(cv2.resize(inf_images[1], None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR))
            self.infer_right_pub.publish(out_img)

    def timer_callback(self):
        if self.camera_left == True and self.camera_right == True:
            self.YOLOv7_detect()

def main(args=None):
    """
    Run the main function.
    """
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()