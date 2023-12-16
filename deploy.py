from dotenv import load_dotenv
import os

load_dotenv()

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])

project = rf.workspace(os.environ['ROBOFLOW_WORKSPACE_NAME']).project("detect-players-dgxz0")

version = project.version(6)
version.deploy("yolov8", "runs/detect/train4")