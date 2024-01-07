from flask import Flask, render_template, Response, request
import json
import argparse
import os
import sys
from pathlib import Path
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.checks import cv2, print_args
from utils.general import update_options

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Initialize Flask API
app = Flask(__name__)

gender_age_model = load_model("models\Gender_Age_90%+5%.h5")
emotion_model = load_model("models\Emotion_CNN_82%.h5")
age_ranges = ["1-10", "11-18", "19-29", "30-45", "46-65", "65-100+"]
gender_ranges = ["male", "female"]
emotion_ranges = ["positive", "negative", "neutral"]
model = YOLO("models\yolov8n-face.pt")
model.fuse()


def predict(opt):
    """
    Perform object detection using the YOLO model and yield results.

    Parameters:
    - opt (Namespace): A namespace object that contains all the options for YOLO object detection,
        including source, model path, confidence thresholds, etc.

    Yields:
    - JSON: If opt.save_txt is True, yields a JSON string containing the detection results.
    - bytes: If opt.save_txt is False, yields JPEG-encoded image bytes with object detection results plotted.
    """
    img = cv2.imread(opt.source)
    results = model(opt.source)
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    # for result in results:
    #     if opt.save_txt:
    #         result_json = json.loads(result.tojson())
    #         yield json.dumps({"results": result_json})
    #     else:
    #         im0 = cv2.imencode(".jpg", result.plot())[1].tobytes()
    #         yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + im0 + b"\r\n")

    for face in xyxy:
        x1, y1, x2, y2 = face
        x1, y1, x2, y2 = int(x1 * 0.99), int(y1 * 0.98), int(x2 * 1.01), int(y2 * 1.01)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_gray = gray[y1:y2, x1:x2]
        img_gray = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_AREA)
        img_gray = np.expand_dims(img_gray, axis=0)
        img_gray = np.expand_dims(img_gray, axis=-1) / 255.0
        output = gender_age_model.predict(img_gray)
        gender = gender_ranges[round(output[0][0][0])]
        age = round(output[1][0][0])

        img_gray = gray[y1:y2, x1:x2]
        img_gray = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_AREA) / 255.0
        img_gray = np.expand_dims(img_gray, axis=0)
        img_gray = np.expand_dims(img_gray, axis=-1)
        emotion = emotion_model.predict(img_gray)
        class_id = np.argmax(emotion)
        emotion1 = emotion_ranges[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
        cv2.putText(
            img,
            "Age: " + str(round(output[1][0][0])),
            (x1, y2 + 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Gender: " + gender_ranges[round(output[0][0][0])],
            (x1, y2 + 60),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Emotion: " + emotion_ranges[class_id],
            (x1, y2 + 90),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    im0 = cv2.imencode(".jpg", img)[1].tobytes()
    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + im0 + b"\r\n")


@app.route("/")
def index():
    """
    Video streaming home page.
    """

    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def video_feed():
    if request.method == "POST":
        uploaded_file = request.files.get("myfile")
        save_txt = request.form.get(
            "save_txt", "F"
        )  # Default to 'F' if save_txt is not provided

        if uploaded_file:
            source = Path(__file__).parent / raw_data / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
            print(opt.source)
        else:
            opt.source, _ = update_options(request)

        opt.save_txt = True if save_txt == "T" else False

    elif request.method == "GET":
        opt.source, opt.save_txt = update_options(request)

    return Response(predict(opt), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data",
        "--raw-data",
        default=ROOT / "data/raw",
        help="save raw images to data/raw",
    )
    parser.add_argument("--port", default=5000, type=int, help="port deployment")
    opt, unknown = parser.parse_known_args()

    # print used arguments
    print_args(vars(opt))

    # Get por to deploy
    port = opt.port
    delattr(opt, "port")

    # Create path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, "raw_data")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Load model (Ensemble is not supported)
    # Run app
    gender_age_model.summary()
    emotion_model.summary()
    app.run(
        host="0.0.0.0", port=port, debug=False
    )  # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)
