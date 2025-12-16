import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = joblib.load('model/xgb_cfd_model.pkl')

# الأعمدة اللي الموديل متوقعها بالترتيب
input_columns = ['y_U1','y_U2','y_U3','y_U4','y_U5','y_U6','y_U7','y_U8',
                 'y_L1','y_L2','y_L3','y_L4','y_L5','y_L6','y_L7','y_L8',
                 'alpha','Mach']

st.title("Airfoil Aerodynamics Predictor")
st.write("Predict Lift Coefficient (Cl) and Moment Coefficient (Cm) from airfoil geometry and flow conditions.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with 18 inputs (same order as required)", type="csv")

if uploaded_file is not None:
    X_input = pd.read_csv(uploaded_file)

    # الأعمدة المطلوبة للرفع
    required_columns = ['y_U1','y_U2','y_U3','y_U4','y_U5','y_U6','y_U7','y_U8',
                        'y_L1','y_L2','y_L3','y_L4','y_L5','y_L6','y_L7','y_L8',
                        'alpha','Mach']

    if X_input.shape[1] != len(required_columns):
        st.error(f"CSV file must have exactly {len(required_columns)} columns.")
    else:
        X_input.columns = required_columns

        # Feature engineering: حساب الأعمدة الإضافية تلقائيًا
        y_U_cols = [f'y_U{i}' for i in range(1,9)]
        y_L_cols = [f'y_L{i}' for i in range(1,9)]

        y_U_vals = X_input[y_U_cols].values
        y_L_vals = X_input[y_L_cols].values

        camber = (y_U_vals + y_L_vals)/2
        thickness = y_U_vals - y_L_vals

        X_input['camber_mean'] = camber.mean(axis=1)
        X_input['max_camber'] = camber.max(axis=1)
        X_input['max_camber_idx'] = np.argmax(camber, axis=1)
        X_input['thickness_mean'] = thickness.mean(axis=1)
        X_input['max_thickness'] = thickness.max(axis=1)
        X_input['max_thickness_idx'] = np.argmax(thickness, axis=1)

        # Prediction
        if st.button("Predict"):
            feature_cols = input_columns + ['camber_mean', 'max_camber_idx', 'max_camber', 
                                'thickness_mean', 'max_thickness', 'max_thickness_idx']
            y_pred = model.predict(X_input[feature_cols])
            st.subheader("Predicted Aerodynamic Coefficients")
            for i, row in X_input.iterrows():
                st.write(f"Sample {i+1}:")
                st.write(f"Lift Coefficient (Cl): {y_pred[i][0]:.4f}")
                st.write(f"Moment Coefficient (Cm): {y_pred[i][1]:.4f}")

                # Plot airfoil geometry
                y_U = row[y_U_cols].values
                y_L = row[y_L_cols].values
                camber_line = (y_U + y_L)/2
                thickness_line = y_U - y_L
                x = np.linspace(0, 1, len(y_U))

                fig, ax = plt.subplots()
                ax.plot(x, y_U, 'b-o', label='Upper Surface')
                ax.plot(x, y_L, 'r-o', label='Lower Surface')
                ax.plot(x, camber_line, 'g--', label='Camber Line')
                ax.scatter(x[np.argmax(thickness_line)], np.max(thickness_line), color='purple', s=80, label='Max Thickness')
                ax.scatter(x[np.argmax(camber_line)], np.max(camber_line), color='orange', s=80, label='Max Camber')
                ax.set_xlabel("Chordwise Position")
                ax.set_ylabel("Coordinate")
                ax.set_title(f"Airfoil Geometry Visualization - Sample {i+1}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)