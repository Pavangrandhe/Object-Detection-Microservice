from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)


AI_SERVICE_URL = "http://127.0.0.1:5001/detect"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        response = requests.post(AI_SERVICE_URL, files={'image': file})
        detections = response.json()
        return jsonify(detections)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)