from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

cat_features = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content','Item_Type']
num_features = ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {}
        for feature in cat_features + num_features:
            data[feature] = request.form.get(feature)
        for feature in num_features:
            data[feature] = float(data[feature])
        for feature in cat_features:
            data[feature] = str(data[feature])
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return render_template("index.html", prediction=round(prediction,2))
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
