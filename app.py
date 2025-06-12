from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# 🔧 Define the custom AttentionSum layer
class AttentionSum(Layer):
    def call(self, inputs):
        return K.sum(inputs, axis=1)

app = Flask(__name__)

# ✅ Load tokenizer
tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))

# ✅ Load models with custom object scope
with CustomObjectScope({'AttentionSum': AttentionSum}):
    lstm_model = load_model("model/lstm_model.h5")

ann_model = load_model("model/ann_model.h5")

MAX_LEN = 200

@app.route("/", methods=["GET", "POST"])
def index():
    result_lstm = ""
    result_ann = ""

    if request.method == "POST":
        url = request.form["url"]
        seq = tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # 🔍 Predictions
        pred_lstm = lstm_model.predict(padded)[0][0]
        pred_ann = ann_model.predict(padded)[0][0]

        result_lstm = f"{round(pred_lstm*100, 2)}% - {'Phishing' if pred_lstm > 0.5 else 'Legitimate'}"
        result_ann = f"{round(pred_ann*100, 2)}% - {'Phishing' if pred_ann > 0.5 else 'Legitimate'}"

    return render_template("index.html", result_lstm=result_lstm, result_ann=result_ann)

if __name__ == "__main__":
    app.run(debug=True)
