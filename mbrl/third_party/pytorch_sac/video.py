import os

import imageio

from mbrl.third_party.pytorch_sac import utils
from matplotlib import pyplot as plt
import cv2

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, "video") if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.height,
                width=self.width,
                camera_id=self.camera_id,
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            print("Start saving video")
            print(path)
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.frames[0].shape[:-1])

            for i in range(len(self.frames)):
                rgb_img = cv2.cvtColor(self.frames[i], cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()
            print("End Saving video")
            #imageio.mimsave(path, self.frames, fps=self.fps)