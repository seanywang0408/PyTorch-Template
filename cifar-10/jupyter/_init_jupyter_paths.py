""" setup jupyter path """
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(-1, path)


add_path(os.path.join(sys.path[0], '../..'))
print("add code root path (with `mylib`).")