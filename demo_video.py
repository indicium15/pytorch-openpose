import ffmpeg
from src.hand import Hand
from src.body import Body
from src import util
from src import model
import csv
import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)


# openpose setup

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame_number, frame, body=True, hands=True):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        person_id = 1
    for person in subset:
        # Create a CSV file for each person
        with open(f'output/limb_coordinates_person_{person_id}_frame{frame_number}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header (optional)
            # Assuming 18 keypoints (OpenPose standard), adjust if different
            header = []
            for i in range(18):
                header.extend([f'Part_{i}_X', f'Part_{i}_Y', f'Part_{i}_Score'])
            writer.writerow(header)

            # Initialize a row for limb coordinates
            row = []
            for part_idx in person[:-2]:  # Exclude the last two elements
                if part_idx != -1:
                    # Add the limb coordinates and score
                    limb = candidate[int(part_idx)]
                    row.extend([limb[0], limb[1], limb[2]])  # x, y, score
                else:
                    # No keypoint detected for this limb, write empty values or placeholders
                    row.extend([None, None, None])
            writer.writerow(row)

        person_id += 1
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(
                peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(
                peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas


# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error

def get_cropping_dimensions(width, height):
    """ Return dimensions to crop the video to make width even while maintaining height """
    if width % 2 == 0:
        return 0, 0, width, height  # No cropping needed
    else:
        # Crop one pixel from the right
        return 0, 0, width - 1, height

# open specified video
parser = argparse.ArgumentParser(
    description="Process a video annotating poses detected.")
parser.add_argument('file', type=str, help='Video file location to process.')
parser.add_argument('--no_hands', action='store_true', help='No hand pose')
parser.add_argument('--no_body', action='store_true', help='No body pose')
args = parser.parse_args()
video_file = args.file
cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    cap.release()
    sys.exit(1)

input_height, input_width = frame.shape[:2]

# Calculate cropping dimensions
x, y, w, h = get_cropping_dimensions(input_width, input_height)

# get video file info
ffprobe_result = ffprobe(args.file)
info = json.loads(ffprobe_result.json)
videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
input_fps = videoinfo["avg_frame_rate"]
# input_fps = float(input_fps[0])/float(input_fps[1])
input_pix_fmt = videoinfo["pix_fmt"]
input_vcodec = videoinfo["codec_name"]

# define a writer object to write to a movidified file
postfix = info["format"]["format_name"].split(",")[0]
output_file = ".".join(video_file.split(".")[:-1])+".processed." + postfix


class Writer():
    def __init__(self, output_file, input_fps, input_framesize, input_pix_fmt,
                 input_vcodec):
        if os.path.exists(output_file):
            os.remove(output_file)
        width, height = input_framesize[1], input_framesize[0]
        self.width = width if width % 2 == 0 else width - 1
        self.height = height if height % 2 == 0 else height - 1
        self.ff_proc = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt="bgr24",
                   s='%dx%d' % (width, height),
                   r=input_fps)
            .filter('scale', width, height)  # Add the scale filter here
            .output(output_file, pix_fmt=input_pix_fmt, vcodec=input_vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()


writer = None
frame_number = 1
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    print(f"{y}:{y+h},{x}:{x+w}")
    frame = frame[y:y+h, x:x+w]
    posed_frame = process_frame(frame_number, frame, body=not args.no_body,
                                hands=False)
    frame_number+=1
    if writer is None:
        input_framesize = posed_frame.shape[:2]
        writer = Writer(output_file, input_fps, input_framesize, input_pix_fmt,
                        input_vcodec)

    cv2.imshow('frame', posed_frame)

    # write the frame
    writer(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.close()
cv2.destroyAllWindows()
