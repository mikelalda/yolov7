#!/usr/bin/env python3

import cv2
import torch
import numpy as np

from utils.general import check_img_size, cv2, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from utils.datasets import letterbox
from models.common import DetectMultiBackend
from utils.general import process_mask, scale_masks
from utils.plots import plot_masks, Annotator, colors

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from detection_msgs.msg import DetectionArray, Detection
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
        self.declare_parameter('inf_right_image_topic', '/object_detection/right_inf_image')
        self.declare_parameter('bbox_right_topic', '/object_detection/right_bboxes')
        self.declare_parameter('frequency', 40)
        self.declare_parameter('is_seg', False)
        self.declare_parameter('data', 'data.coco128.yaml')
        self.declare_parameter('line_thickness',3)

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
        self.is_seg = self.get_parameter('is_seg').value
        self.data = self.get_parameter('data').value
        self.line_thickness = self.get_parameter('line_thickness').value

        self.get_logger().info("weights: %s, topic_left: %s, topic_right: %s, det_conf_thres: %s, conf_thres: %s, iou_thres: %s, device: %s, img_size: %s, inf_image: %s, inf_left_image_topic: %s, bbox_left_topic: %s, inf_right_image_topic: %s, bbox_right_topic: %s, frequency: %s, is_seg: %s, data: %s, line_thickness: %s" %
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
                        str(self.frequency),
                        str(self.is_seg),
                        str(self.data),
                        str(self.line_thickness)))

        # Camera info and frames
        self.left_image = None
        self.right_image = None

        # Flags
        self.left_RGB = False
        self.right_RGB = False

        # Timer callback
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for Classes
        qos_pub = QoSProfile(depth=10) 
        qos_pub.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.infer_left_pub = self.create_publisher(Image, self.inf_left_image_topic, qos_profile=qos_pub)
        self.infer_right_pub = self.create_publisher(Image, self.inf_right_image_topic, qos_profile=qos_pub)
        self.bb_left_pub = self.create_publisher(DetectionArray, self.bbox_left_topic, qos_profile=qos_pub)
        self.bb_right_pub = self.create_publisher(DetectionArray, self.bbox_right_topic, qos_profile=qos_pub)
 
        # OpenCv - ROS 2 Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        qos_sub = QoSProfile(depth=1) 
        qos_sub.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.left_sub = self.create_subscription(Image, self.topic_left, qos_profile=qos_sub, callback=self.left_callback)
        self.right_sub = self.create_subscription(Image, self.topic_right, qos_profile=qos_sub, callback=self.right_callback)

        # Initialize YOLOv7
        try:
            print("torch_info: {current: ", torch.cuda.current_device(),
                "device 0: ",torch.cuda.device(0),
                "device count: ",torch.cuda.device_count(),
                "device name: ",torch.cuda.get_device_name(0),
                "is available",torch.cuda.is_available(),
                "}")
        except:
            print("No cuda available")
        set_logging()
        self.device = select_device(self.device)
        print("Selected device: ", self.device)
        self.half = False #self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        
        # Get names and colors
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        #if self.device.type != 'cpu':
           # self.model(torch.zeros(2, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
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
        im0s = [self.left_image.copy(), self.right_image.copy()]
        
        self.left_RGB = False
        self.right_RGB = False

        img_left = letterbox(self.left_image, self.imgsz, self.stride)[0]
        img_left = img_left[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img_right = letterbox(self.right_image, self.imgsz, self.stride)[0]
        img_right = img_right[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray([img_left,img_right])
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img_left = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=100, nm=32)

        # Process detections
        all_det_array = []
        inf_images = []
        s=''
        for i, det in enumerate(pred):  # detections per image
            # Initialize array of bounding boxes
            det_array = DetectionArray()
            im0 = im0s[i].copy()
            
            if len(det):
              # Write results
              for *xyxy, conf, cls in reversed(det):
                  if conf > self.det_conf_thres: # Limit confidence threshold to 80% for all classes
                      # Draw a boundary box around each object
                      label = f'{self.names[int(cls)]} {conf:.2f}'
                      # Save bounding box to publish later
                      detection = Detection()
                      detection.class_name = label
                      detection.class_id = cls
                      detection.bbox.center.position.x = float(xyxy[0])-float(xyxy[2])
                      detection.bbox.center.position.y = float(xyxy[1])-float(xyxy[3])
                      detection.bbox.center.theta = 0.0
                      detection.bbox.size_x = abs(detection.bbox.center.position.x/2)
                      detection.bbox.size_y = abs(detection.bbox.center.position.y/2)
                      if self.is_seg:
                          # Segmentation
                          detection.mask.size_x = abs(detection.bbox.center.position.x/2)
                          detection.mask.size_y = abs(detection.bbox.center.position.y/2)
                          detection.mask.data = xyxy[4:]
                      det_array.boxes.append(detection)

            det_array.header.stamp = self.get_clock().now().to_msg()
            all_det_array.append(det_array)
            inf_images.append(im0)
            #cv2.imshow("YOLOv7 Object detection result RGB", cv2.cvtColor(cv2.resize(im0, None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR)) 
            #cv2.waitKey()  

        
        # Publish bounding boxes
        self.bb_left_pub.publish(all_det_array[0])
        self.bb_right_pub.publish(all_det_array[1])
        
        # Publish output ROS 2 image
        if self.inf_image: #TODO: Add option to publish both images
            out_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(cv2.resize(inf_images[0], None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR))
            self.infer_left_pub.publish(out_img)
            out_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(cv2.resize(inf_images[1], None, fx=1.5, fy=1.5),cv2.COLOR_RGB2BGR))
            self.infer_right_pub.publish(out_img)

    def timer_callback(self):
        if self.left_RGB == True and self.right_RGB == True:
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
