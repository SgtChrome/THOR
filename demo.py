# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import pdb
import argparse, cv2, os
import numpy as np
import sys
from imutils.video import FPS
import json
from trackingmultiple import Controltool

from trackers.tracker import SiamFC_Tracker, SiamRPN_Tracker, SiamMask_Tracker
from benchmark.bench_utils.bbox_helper import cxy_wh_2_rect, xyxy_to_xywh

# constants
BRIGHTGREEN = [102, 255, 0]
RED = [0, 0, 255]
YELLOW = [0, 255, 255]
np.set_printoptions(precision=6, suppress=True)

OUTPUT_WIDTH = 740
OUTPUT_HEIGHT = 555
PADDING = 2

parser = argparse.ArgumentParser(description='Webcam Test')
parser.add_argument('-t', '--tracker', dest='tracker', required=True,
                    help='Name of the tracker [SiamFC, SiamRPN, SiamMask]')
parser.add_argument('--vanilla', action='store_true',
                    help='run the tracker without memory')
parser.add_argument('-v', '--viz', action='store_true',
                    help='whether visualize result')
parser.add_argument('--verbose', action='store_true',
                    help='print info about temp mem')
parser.add_argument('--lb_type', type=str, default='ensemble',
                    help='Specify the type of lower bound [dynamic, ensemble]')

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False

def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize, boxToDraw_xywh
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0, 2]] = np.sort(boxToDraw[[0, 2]])
    boxToDraw[[1, 3]] = np.sort(boxToDraw[[1, 3]])
    boxToDraw_xywh = xyxy_to_xywh(boxToDraw)

def bb_on_im(im, location, mask):
    location = [int(l) for l in location]  #

    if len(mask):
        im[:, :, 2] = mask * 255 + (1 - mask) * im[:, :, 2]

    # prediction
    cv2.rectangle(im, (location[0], location[1]),
                  (location[0] + location[2], location[1] + location[3]),
                  (0, 255, 255), 3)

    return im

def controlloop():
    #for videoname, signal in videofilelist:
    startframe, stopframe, video_name =  ct.initialize(0)
    positionlist = ct.get_start_position()
    outputlist = []
    # track every marked signal in image, for only one signal append [:1] to positionlist
    for index, position in enumerate(positionlist[:1], 1):
        startposition = (position.xmin, position.ymin, position.xmax-position.xmin, position.ymax-position.ymin)
        position_x_y_w_h = [position.xmin, position.ymin, position.xmax, position.ymax]
        outputlist.append(main(index, len(positionlist), startframe, stopframe, startposition, position_x_y_w_h, video_name))
    # write tracked signals as (frame1)s1xmin,s1ymin,s1xmax,s1ymax-s2xmin,s2ymin,s2xmax,s2ymax-...
    #                          (frame2)s1xmin,s1ymin,s1xmax,s1ymax-s2xmin,s2ymin,s2xmax,s2ymax-...
    with open("D:/Users/Frederic/DokumenteDokumente/INAVET/Versuch/boxes/versuch.txt", "w+") as boxesfile:
        outputstring = ''
        rows = len(outputlist[0])
        for i in range(0, rows):
            for positionlist in outputlist:
                outputstring += ','.join(positionlist[i]) + '-'
            outputstring += '\n'
        boxesfile.write(outputstring)

def show_webcam(tracker, ct, startframe, mirror=False, viz=False):
    initialize = True

    """ vs = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.setMouseCallback('Webcam', on_mouse, 0) """

    outputBoxToDraw = None
    bbox = None
    fps = None
    state = None
    mask = []

    # loop over video stream ims
    output = []
    f = startframe
    counter = 1
    #starttime = time.time()

    while f < startframe + 900:
        im = ct.get_frame_from_index(f)

        if mousedown:
            (x1, y1, x2, y2) = [int(l) for l in boxToDraw]
            cv2.rectangle(im, (x1, y1), (x2, y2),
                          BRIGHTGREEN, PADDING)

        elif mouseupdown:
            if initialize:
                init_pos = boxToDraw_xywh[[0, 1]]
                init_sz = boxToDraw_xywh[[2, 3]]
                state = tracker.setup(im, init_pos, init_sz)
                initialize = False
                #fps = FPS().start()
            else:
                state = tracker.track(im, state)
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                (cx, cy, w, h) = [int(l) for l in location]
                """
                bboxi = [cx, cy, cx + w, cy + h]
                bboxi = [str(x) for x in bboxi]
                output.append(bboxi) """

                fps.update()
                fps.stop()

                # Display the image
                info = [
                    ("Score:", f"{state['score']:.4f}"),
                    ("FPS:", f"{fps.fps():.2f}"),
                ]

                if not state['score'] > 0.8:
                    info.insert(0, ("Object lost since", ""))
                else:
                    if 'mask' in state.keys():
                        mask = state['mask'] > state['p'].seg_thr
                    im = bb_on_im(im, location, mask)

                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(im, text, (10, OUTPUT_HEIGHT - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Webcam", im)

        # check for escape key
        key = cv2.waitKey(1)
        if key==27 or key==1048603:
            break

    # release the pointer
    cv2.destroyAllWindows()

def load_cfg(args):
    json_path = f"configs/{args.tracker}/VOT2018_"
    if args.vanilla:
        json_path += "vanilla.json"
    else:
        json_path += f"THOR_{args.lb_type}.json"
    cfg = json.load(open(json_path))
    return cfg

class argsis:
    def __init__(self):
        self.viz = True
        self.verbose = True
        self.tracker = 'SiamFC'
        self.lb_type = 'ensemble'
        self.vanilla = False

if __name__ == '__main__':
    ct = Controltool()
    startframe, stopframe, video_name =  ct.initialize(0)

    # args = parser.parse_args()
    args = argsis()
    cfg = load_cfg(args)
    cfg['THOR']['viz'] = args.viz
    cfg['THOR']['verbose'] = args.verbose

    print("[INFO] Initializing the tracker.")
    if args.tracker == 'SiamFC':
        tracker = SiamFC_Tracker(cfg)
    elif args.tracker == 'SiamRPN':
        tracker = SiamRPN_Tracker(cfg)
    elif args.tracker == 'SiamMask':
        tracker = SiamMask_Tracker(cfg)
    elif args.tracker == 'SiamRPN_PP':
        tracker = SiamRPN_PP_Tracker(cfg)
    else:
        raise ValueError(f"Tracker {args.tracker} does not exist.")

    print("[INFO] Starting video stream.")
    show_webcam(tracker, ct, startframe, mirror=True, viz=args.viz)
