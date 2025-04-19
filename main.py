from flask import Flask, render_template, jsonify
from gold_data import get_recent_gold, get_history_for_chart
from predict import forecast_next_days
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/history")
def history():
    return jsonify(get_history_for_chart())

@app.route("/api/predict/<int:days>")
def predict(days):
    recent_scaled, min_val, max_val = get_recent_gold()
    prices = recent_scaled[:, 1].astype(np.float32)

    preds = forecast_next_days(prices, steps=days)

    preds_real = [(float(p) * float(max_val - min_val)) + float(min_val) for p in preds]
    return jsonify(list(map(float, preds_real)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)


from gold_data import get_history_for_chart

@app.route("/api/history")
def history():
    return jsonify(get_history_for_chart())

