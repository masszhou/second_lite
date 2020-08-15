import numpy as np
import struct
import socket
import time


class LidarViewer:
    """
    message header
    1 = load points
    2 = clear points
    3 = reset view to fit all
    4 = set viewer property
    5 = get viewer property
    6 = print screen
    7 = wait for enter
    8 = load camera path animation
    9 = playback camera path animation
    10 = set per point attributes
    """
    def __init__(self, port: int):
        # start up viewer in separate process
        self._port_number = port

    def load_points(self, *args):
        positions = np.asarray(args[0], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(args[1], dtype=np.float32)
        self._load_positions(positions)
        self._load_colors(colors)

    def load_boxes(self, xyzwhl_array):
        pass

    def _load_positions(self, positions):
        # if no points, then done
        if positions.size == 0:
            return
        # construct message
        num_points = int(positions.size / 3)
        msg = struct.pack('b', 1) + struct.pack('i', num_points) + positions.tobytes()
        # send message to viewer
        self._send(msg)

    def _load_colors(self, colors):
        if colors.size == 0:
            return
        num_points = int(colors.size)
        msg = struct.pack('b', 10) + struct.pack('i', num_points) + colors.tostring()
        self._send(msg)

    def _send(self, msg):
        start = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', self._port_number))
        totalSent = 0
        while totalSent < len(msg):
            sent = s.send(msg)
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalSent = totalSent + sent
        s.close()
        print("{} [sec]".format(time.time() - start))