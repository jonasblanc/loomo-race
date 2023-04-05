# VITA, EPFL

#import cv2
import socket
import sys
import numpy as np
import struct
import binascii

from PIL import Image
from detector import Detector
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('-c', '--checkpoint',
                    help=('directory to load checkpoint'))
parser.add_argument('-i','--ip-address',
                    help='IP Address of robot')
parser.add_argument('--instance-threshold', default=0.0, type=float,
                    help='Defines the threshold of the detection score')
parser.add_argument('-d', '--downscale', default=4, type=int,
                    help=('downscale of the received image'))
parser.add_argument('--square-edge', default=401, type=int,
                    help='square edge of input images')

args = parser.parse_args()

##### IP Address of server #########
host = args.ip_address #'128.179.150.43'  # The server's hostname or IP address
####################################
port = 8081        # The port used by the server

# image data
downscale = args.downscale
width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels

# create socket
print('# Creating socket')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
    print('Failed to create socket')
    sys.exit()

print('# Getting remote IP address')
try:
    remote_ip = socket.gethostbyname( host )
except socket.gaierror:
    print('Hostname could not be resolved. Exiting')
    sys.exit()

# Connect to remote server
print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
s.connect((remote_ip , port))

# Set up detector
arguments = ["--checkpoint",args.checkpoint,"--pif-fixed-scale", "1.0", "--instance-threshold",args.instance_threshold]
detector = Detector()

#Image Receiver
net_recvd_length = 0
recvd_image = b''

#Test Controller
direction = -1
cnt = 0

last_bbox = [0,0, 10, 10, 0.0]

# If true, when person of interest is lost, keep his last position as current one
WITH_INERTIA = True 

# 0.25 gauche => max
# 0.22 droite => min
MARGIN_DISTANCE_CAM_RIGHT = 0.23
MARGIN_DISTANCE_CAM_LEFT = 0.3

MARGIN_DONT_MOVE_WHEN_LOST = 0.35

while True:

    # Receive data
    reply = s.recv(sz_image)
    recvd_image += reply
    net_recvd_length += len(reply)

    if net_recvd_length == sz_image:

        pil_image = Image.frombytes('RGB', (width, height), recvd_image)
        # opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        # opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
        #
        # cv2.imshow('Test window',opencvImage)

        #cv2.waitKey(1)
        net_recvd_length = 0
        recvd_image = b''

        #######################
        # Detect
        #######################

        # Empty bbox if person of interest not detected

        frame = np.array(pil_image)

        bbox, bbox_label = detector.forward(frame, is_re_init_allowed=True)
        
        if bbox_label:
            print("BBOX: {}".format(bbox))
            print("BBOX_label: {}".format(bbox_label))
        else:
            print("False")

        
        width = frame.shape[1]

        if len(bbox) == 0:
                    
            # Robot follow last bbox (turn in circle probably)
            if WITH_INERTIA:
                values = last_bbox
            else:
                # 0.0 confidence => Robot does not move
                values = [0, 0, 10, 10, 0.0]
        else:
            # Update last_bbox
            # If tracking lost when in the image => don't move
            conf = 1.0
            min_x = int(MARGIN_DONT_MOVE_WHEN_LOST * width)
            max_x = int(width - MARGIN_DONT_MOVE_WHEN_LOST * width)
            if bbox[0] > min_x and bbox[0] < max_x:
                conf = 0.0
            last_bbox = [bbox[0], bbox[1], bbox[2], bbox[3], conf]
            print("Last_bbox", last_bbox)
            
            # According to the TA w, h = 10 is more robust for depth estimation
            values = [bbox[0], bbox[1], bbox[2], bbox[3], 1.0]
            print("Values", values)


        # Keep the detected person inside the distance camera
        width = frame.shape[1]
        min_distance_x = int(MARGIN_DISTANCE_CAM_LEFT * width)
        max_distance_x = int(width - MARGIN_DISTANCE_CAM_RIGHT * width)
        values[0] = max(min(values[0], max_distance_x), min_distance_x)
        values[2] = 10
        values[3] = 10

        #MARGIN_DISTANCE_CAM = 0.28
        #values = (int(frame.shape[1]/2), int(frame.shape[0]/2), int(width -   width * MARGIN_DISTANCE_CAM * 2), 10, 1.0)

        #print(values)

        packer = struct.Struct('f f f f f')
        packed_data = packer.pack(*values)

        # Send data
        send_info = s.send(packed_data)