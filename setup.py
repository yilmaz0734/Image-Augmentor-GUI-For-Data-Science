import PyQt5
import os
import imutils
import cv2
import numpy as np
from PIL import Image as im 
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QGuiApplication
import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os","PyQt5","imutils","cv2","numpy","PIL","sys"], "excludes": ["tkinter","pandas","matplotlib","bs4","scipy"],"include_files":["augmentation.ui","2582365.ico"]}

# GUI applications require a different base on Windows (the default is for
# a console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "Image Augmentor For Machine Learning",
        version = "1.0",
        description = "Provides up to 229 times more data!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("augmentation.py", base=base, icon="2582365.ico")])