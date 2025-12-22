import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import io

# Load model
model = joblib.load('model/xgb_cfd_model.pkl')

# Input columns expected by the model
input_columns = ['y_U1','y_U2','y_U3','y_U4','y_U5','y_U6','y_U7','y_U8',
                 'y_L1','y_L2','y_L3','y_L4','y_L5','y_L6','y_L7','y_L8',
                 'alpha','Mach']

def validate_inputs(df, required_columns):
    """
    Validate input data for physical constraints and data quality
    Returns: (is_valid, error_messages)
    """
    errors = []
    
    # Check for missing values
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"❌ Missing values found in columns: {', '.join(missing_cols)}")
    
    # Check for non-numeric values
    non_numeric_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
    if non_numeric_cols:
        errors.append(f"❌ Non-numeric values found in columns: {', '.join(non_numeric_cols)}")
    
    # Physical constraint checks
    y_U_cols = [f'y_U{i}' for i in range(1,9)]
    y_L_cols = [f'y_L{i}' for i in range(1,9)]
    
    # Check if upper surface is above lower surface
    for idx, row in df.iterrows():
        y_U_vals = row[y_U_cols].values
        y_L_vals = row[y_L_cols].values
        
        if np.any(y_U_vals < y_L_vals):
            errors.append(f"❌ Row {idx+1}: Upper surface must be above lower surface (y_U > y_L)")
            break
    
    # Check angle of attack range (training range: -2° to 14.5°)
    if 'alpha' in df.columns:
        alpha_values = df['alpha']
        if alpha_values.min() < -2 or alpha_values.max() > 14.5:
            errors.append(f"⚠️ Warning: Angle of attack (alpha) outside training range [-2°, 14.5°]. Found: [{alpha_values.min():.2f}°, {alpha_values.max():.2f}°]. Predictions may be unreliable.")
    
    # Check Mach number range (subsonic: 0 to ~1)
    if 'Mach' in df.columns:
        mach_values = df['Mach']
        if mach_values.min() < 0:
            errors.append(f"❌ Mach number must be non-negative. Found: {mach_values.min():.4f}")
        if mach_values.max() > 1.0:
            errors.append(f"⚠️ Warning: Mach number exceeds subsonic range (>1.0). Found: {mach_values.max():.4f}. Model trained on subsonic data; predictions may be unreliable.")
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        errors.append(f"❌ Infinite values detected in the data")
    
    # Check thickness (upper - lower should be positive everywhere)
    for idx, row in df.iterrows():
        thickness = row[y_U_cols].values - row[y_L_cols].values
        if np.any(thickness <= 0):
            errors.append(f"❌ Row {idx+1}: Invalid airfoil geometry - upper surface must be above lower surface at all points")
            break
        if np.all(thickness < 0.001):
            errors.append(f"⚠️ Warning Row {idx+1}: Very thin airfoil (max thickness < 0.001). Results may be unreliable.")
            break
    
    is_valid = len([e for e in errors if e.startswith('❌')]) == 0
    return is_valid, errors


st.title("Airfoil Aerodynamics Predictor")
st.write("Predict Lift Coefficient (Cl) and Moment Coefficient (Cm) from airfoil geometry and flow conditions.")

# Display expected input format
with st.expander("ℹ️ Expected Input Format"):
    st.write("""
    **Required Columns (in order):**
    - `y_U1` to `y_U8`: Upper surface coordinates of the airfoil (8 chordwise points)
    - `y_L1` to `y_L8`: Lower surface coordinates of the airfoil (8 chordwise points)
    - `alpha`: Angle of attack in degrees (training range: -2° to 14.5°)
    - `Mach`: Mach number, dimensionless (subsonic range: 0 to ~1)
    
    **Physical Constraints:**
    - **Airfoil geometry**: For each chordwise point, the upper surface must be above the lower surface: y_U > y_L
    - **Mach number**: Must be non-negative
    - **Data integrity**: All values must be numeric with no missing data
    """)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with 18 inputs (same order as required)", type="csv")

if uploaded_file is not None:
    X_input = pd.read_csv(uploaded_file)
    
    # Required columns
    required_columns = ['y_U1','y_U2','y_U3','y_U4','y_U5','y_U6','y_U7','y_U8',
                        'y_L1','y_L2','y_L3','y_L4','y_L5','y_L6','y_L7','y_L8',
                        'alpha','Mach']

    if X_input.shape[1] != len(required_columns):
        st.error(f"❌ CSV file must have exactly {len(required_columns)} columns. Found: {X_input.shape[1]}")
    else:
        X_input.columns = required_columns
        
        # Validate inputs
        is_valid, validation_errors = validate_inputs(X_input, required_columns)
        
        # Display validation results
        if validation_errors:
            st.subheader("Validation Results")
            for error in validation_errors:
                if error.startswith('❌'):
                    st.error(error)
                else:
                    st.warning(error)
        
        if is_valid:
            st.success("✅ All validation checks passed!")
            
            # Feature engineering: calculate additional features
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
                    st.write(f"**Sample {i+1}:**")
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
                
                # Prepare output DataFrame for download
                output_df = X_input.copy()
                output_df['Cl_pred'] = y_pred[:, 0]
                output_df['Cm_pred'] = y_pred[:, 1]

                # Convert DataFrame to CSV in memory
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="📥 Download Prediction Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="airfoil_predictions.csv",
                    mime="text/csv"
                )
        else:
            st.error("❌ Please fix the validation errors before proceeding.")