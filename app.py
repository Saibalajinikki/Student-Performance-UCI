from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
loaded_ss = pickle.load(open('standard_scaler.pkl','rb'))
loaded_le = pickle.load(open('label_encoder.pkl','rb'))
loaded_lr = pickle.load(open('linear_regression_model.pkl','rb'))
loaded_ls = pickle.load(open('lasso_regression.pkl','rb'))
loaded_dt = pickle.load(open('decision_tree_regressor.pkl','rb'))
loaded_rd = pickle.load(open('random_forest_regressor.pkl','rb'))
loaded_sv = pickle.load(open('svr.pkl','rb'))
loaded_gb = pickle.load(open('gradient_boosting.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if data is not None and 'data' in data:
            data = data['data']
            app.logger.info('Raw text input from json: %s', data)

            input_data = pd.DataFrame(data)

            for col, encoder in loaded_le.items():
                if col in input_data.columns:
                    input_data[col] = le.transform(input_data[col].astype(str))

            ss_cols = [['age', 'absences', 'G1', 'G2', 'G3']]
            input_data[ss_cols] = loaded_ss.transform(input_data[ss_cols])

            predictions = {}

            models = {
                'Linear Regression': loaded_lr,
                'Lasso Regression': loaded_ls,
                'Decision Tree Regressor': loaded_dt,
                'Random Forest Regressor': loaded_rd,
                'SVR': loaded_sv,
                'Gradient Boosting Regressor': loaded_gb
                }
        
            for model_name, model in models.items():
                pred = model.predict(input_data)[0]
                org_pred = loaded_ss.inverse_transform([[pred]])[0][0]
                predictions[model_name] = f'The predicted Score using {model_name} is {org_pred:.2f}'

            
            return jsonify(predictions)
        else:
            return jsonify({'Error: Invalid JSON format. Expected {"data": {...}}'})
    except Exception as e:
        app.logger.info('Error during prediction: %s',str(e))
        return jsonify({'Error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
            


