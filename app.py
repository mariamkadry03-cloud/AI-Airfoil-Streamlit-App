# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import numpy as np

# # Load model
# model = joblib.load('model/xgb_cfd_model.pkl')

# # الأعمدة اللي الموديل متوقعها بالترتيب
# input_columns = ['y_U1','y_U2','y_U3','y_U4','y_U5','y_U6','y_U7','y_U8',
#                  'y_L1','y_L2','y_L3','y_L4','y_L5','y_L6','y_L7','y_L8',
#                  'alpha','Mach']

# def validate_inputs(df, required_columns):
#     """
#     Validate input data for physical constraints and data quality
#     Returns: (is_valid, error_messages)
#     """
#     errors = []
    
#     # Check for missing values
#     if df.isnull().any().any():
#         missing_cols = df.columns[df.isnull().any()].tolist()
#         errors.append(f"❌ Missing values found in columns: {', '.join(missing_cols)}")
    
#     # Check for non-numeric values
#     non_numeric_cols = []
#     for col in df.columns:
#         if not pd.api.types.is_numeric_dtype(df[col]):
#             non_numeric_cols.append(col)
#     if non_numeric_cols:
#         errors.append(f"❌ Non-numeric values found in columns: {', '.join(non_numeric_cols)}")
    
#     # Physical constraint checks
#     y_U_cols = [f'y_U{i}' for i in range(1,9)]
#     y_L_cols = [f'y_L{i}' for i in range(1,9)]
    
#     # Check if upper surface is above lower surface
#     for idx, row in df.iterrows():
#         y_U_vals = row[y_U_cols].values
#         y_L_vals = row[y_L_cols].values
        
#         if np.any(y_U_vals < y_L_vals):
#             errors.append(f"❌ Row {idx+1}: Upper surface must be above lower surface (y_U > y_L)")
#             break
    
#     # Check angle of attack range (typical: -20° to +20°)
#     if 'alpha' in df.columns:
#         alpha_values = df['alpha']
#         if alpha_values.min() < -30 or alpha_values.max() > 30:
#             errors.append(f"⚠️ Warning: Angle of attack (alpha) outside typical range [-30°, 30°]. Found: [{alpha_values.min():.2f}, {alpha_values.max():.2f}]")
    
#     # Check Mach number range (typical: 0 to 1 for subsonic, up to 5 for supersonic)
#     if 'Mach' in df.columns:
#         mach_values = df['Mach']
#         if mach_values.min() < 0:
#             errors.append(f"❌ Mach number cannot be negative. Found: {mach_values.min():.4f}")
#         if mach_values.max() > 5:
#             errors.append(f"⚠️ Warning: Mach number unusually high (>{mach_values.max():.2f}). Model may not be reliable.")
    
#     # Check for unrealistic airfoil coordinates (typically between -0.5 and 0.5)
#     all_y_cols = y_U_cols + y_L_cols
#     if all(col in df.columns for col in all_y_cols):
#         y_values = df[all_y_cols].values
#         if y_values.min() < -1 or y_values.max() > 1:
#             errors.append(f"⚠️ Warning: Airfoil coordinates outside typical range [-1, 1]. Found: [{y_values.min():.4f}, {y_values.max():.4f}]")
    
#     # Check for infinite values
#     if np.isinf(df.select_dtypes(include=[np.number]).values).any():
#         errors.append(f"❌ Infinite values detected in the data")
    
#     # Check thickness (upper - lower should be positive everywhere)
#     for idx, row in df.iterrows():
#         thickness = row[y_U_cols].values - row[y_L_cols].values
#         if np.any(thickness < 0):
#             errors.append(f"❌ Row {idx+1}: Negative thickness detected (upper surface below lower surface)")
#             break
#         if np.all(thickness < 0.001):
#             errors.append(f"⚠️ Warning Row {idx+1}: Very thin airfoil (max thickness < 0.001). Results may be unreliable.")
#             break
    
#     is_valid = len([e for e in errors if e.startswith('❌')]) == 0
#     return is_valid, errors


# st.title("Airfoil Aerodynamics Predictor")
# st.write("Predict Lift Coefficient (Cl) and Moment Coefficient (Cm) from airfoil geometry and flow conditions.")

# # Display expected input format
# with st.expander("ℹ️ Expected Input Format"):
#     st.write("""
#     **Required Columns (in order):**
#     - `y_U1` to `y_U8`: Upper surface coordinates (8 points)
#     - `y_L1` to `y_L8`: Lower surface coordinates (8 points)
#     - `alpha`: Angle of attack (degrees, typically -20° to +20°)
#     - `Mach`: Mach number (0 to 5, typically 0 to 1 for subsonic)
    
#     **Physical Constraints:**
#     - Upper surface must be above lower surface (y_U > y_L)
#     - Mach number must be non-negative
#     - All values must be numeric with no missing data
#     """)

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV file with 18 inputs (same order as required)", type="csv")

# if uploaded_file is not None:
#     try:
#         X_input = pd.read_csv(uploaded_file)
        
#         # الأعمدة المطلوبة للرفع
#         required_columns = ['y_U1','y_U2','y_U3','y_U4','y_U5','y_U6','y_U7','y_U8',
#                             'y_L1','y_L2','y_L3','y_L4','y_L5','y_L6','y_L7','y_L8',
#                             'alpha','Mach']

#         if X_input.shape[1] != len(required_columns):
#             st.error(f"❌ CSV file must have exactly {len(required_columns)} columns. Found: {X_input.shape[1]}")
#         else:
#             X_input.columns = required_columns
            
#             # Validate inputs
#             is_valid, validation_errors = validate_inputs(X_input, required_columns)
            
#             # Display validation results
#             if validation_errors:
#                 st.subheader("Validation Results")
#                 for error in validation_errors:
#                     if error.startswith('❌'):
#                         st.error(error)
#                     else:
#                         st.warning(error)
            
#             if is_valid:
#                 st.success("✅ All validation checks passed!")
                
#                 # Feature engineering: حساب الأعمدة الإضافية تلقائيًا
#                 y_U_cols = [f'y_U{i}' for i in range(1,9)]
#                 y_L_cols = [f'y_L{i}' for i in range(1,9)]

#                 y_U_vals = X_input[y_U_cols].values
#                 y_L_vals = X_input[y_L_cols].values

#                 camber = (y_U_vals + y_L_vals)/2
#                 thickness = y_U_vals - y_L_vals

#                 X_input['camber_mean'] = camber.mean(axis=1)
#                 X_input['max_camber'] = camber.max(axis=1)
#                 X_input['max_camber_idx'] = np.argmax(camber, axis=1)
#                 X_input['thickness_mean'] = thickness.mean(axis=1)
#                 X_input['max_thickness'] = thickness.max(axis=1)
#                 X_input['max_thickness_idx'] = np.argmax(thickness, axis=1)

#                 # Prediction
#                 if st.button("Predict"):
#                     feature_cols = input_columns + ['camber_mean', 'max_camber_idx', 'max_camber', 
#                                         'thickness_mean', 'max_thickness', 'max_thickness_idx']
#                     y_pred = model.predict(X_input[feature_cols])
#                     st.subheader("Predicted Aerodynamic Coefficients")
#                     for i, row in X_input.iterrows():
#                         st.write(f"**Sample {i+1}:**")
#                         st.write(f"Lift Coefficient (Cl): {y_pred[i][0]:.4f}")
#                         st.write(f"Moment Coefficient (Cm): {y_pred[i][1]:.4f}")

#                         # Plot airfoil geometry
#                         y_U = row[y_U_cols].values
#                         y_L = row[y_L_cols].values
#                         camber_line = (y_U + y_L)/2
#                         thickness_line = y_U - y_L
#                         x = np.linspace(0, 1, len(y_U))

#                         fig, ax = plt.subplots()
#                         ax.plot(x, y_U, 'b-o', label='Upper Surface')
#                         ax.plot(x, y_L, 'r-o', label='Lower Surface')
#                         ax.plot(x, camber_line, 'g--', label='Camber Line')
#                         ax.scatter(x[np.argmax(thickness_line)], np.max(thickness_line), color='purple', s=80, label='Max Thickness')
#                         ax.scatter(x[np.argmax(camber_line)], np.max(camber_line), color='orange', s=80, label='Max Camber')
#                         ax.set_xlabel("Chordwise Position")
#                         ax.set_ylabel("Coordinate")
#                         ax.set_title(f"Airfoil Geometry Visualization - Sample {i+1}")
#                         ax.legend()
#                         ax.grid(True)
#                         st.pyplot(fig)
#             else:
#                 st.error("❌ Please fix the validation errors before proceeding.")
                
#     except Exception as e:
#         st.error(f"❌ Error reading CSV file: {str(e)}")
#         st.info("Please ensure your CSV file is properly formatted.")
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from datetime import datetime
import json

# Load model
model = joblib.load('model/xgb_cfd_model.pkl')

# الأعمدة اللي الموديل متوقعها بالترتيب
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
    
    # Check angle of attack range (typical: -20° to +20°)
    if 'alpha' in df.columns:
        alpha_values = df['alpha']
        if alpha_values.min() < -30 or alpha_values.max() > 30:
            errors.append(f"⚠️ Warning: Angle of attack (alpha) outside typical range [-30°, 30°]. Found: [{alpha_values.min():.2f}, {alpha_values.max():.2f}]")
    
    # Check Mach number range (typical: 0 to 1 for subsonic, up to 5 for supersonic)
    if 'Mach' in df.columns:
        mach_values = df['Mach']
        if mach_values.min() < 0:
            errors.append(f"❌ Mach number cannot be negative. Found: {mach_values.min():.4f}")
        if mach_values.max() > 5:
            errors.append(f"⚠️ Warning: Mach number unusually high (>{mach_values.max():.2f}). Model may not be reliable.")
    
    # Check for unrealistic airfoil coordinates (typically between -0.5 and 0.5)
    all_y_cols = y_U_cols + y_L_cols
    if all(col in df.columns for col in all_y_cols):
        y_values = df[all_y_cols].values
        if y_values.min() < -1 or y_values.max() > 1:
            errors.append(f"⚠️ Warning: Airfoil coordinates outside typical range [-1, 1]. Found: [{y_values.min():.4f}, {y_values.max():.4f}]")
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        errors.append(f"❌ Infinite values detected in the data")
    
    # Check thickness (upper - lower should be positive everywhere)
    for idx, row in df.iterrows():
        thickness = row[y_U_cols].values - row[y_L_cols].values
        if np.any(thickness < 0):
            errors.append(f"❌ Row {idx+1}: Negative thickness detected (upper surface below lower surface)")
            break
        if np.all(thickness < 0.001):
            errors.append(f"⚠️ Warning Row {idx+1}: Very thin airfoil (max thickness < 0.001). Results may be unreliable.")
            break
    
    is_valid = len([e for e in errors if e.startswith('❌')]) == 0
    return is_valid, errors


def create_results_csv(X_input, y_pred, y_U_cols, y_L_cols):
    """Create a CSV file with input data and predictions"""
    results_df = X_input.copy()
    results_df['Predicted_Cl'] = y_pred[:, 0]
    results_df['Predicted_Cm'] = y_pred[:, 1]
    
    csv_buffer = BytesIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer


def create_results_txt(X_input, y_pred, validation_errors):
    """Create a detailed text report"""
    report = []
    report.append("=" * 70)
    report.append("AIRFOIL AERODYNAMICS PREDICTION REPORT")
    report.append("=" * 70)
    report.append(f"\nGeneration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Samples: {len(X_input)}")
    report.append("\n" + "-" * 70)
    
    # Validation Summary
    report.append("\n1. VALIDATION SUMMARY")
    report.append("-" * 70)
    if validation_errors:
        report.append("Validation Issues Found:")
        for error in validation_errors:
            report.append(f"  • {error}")
    else:
        report.append("✓ All validation checks passed")
    
    # Prediction Results
    report.append("\n" + "-" * 70)
    report.append("2. PREDICTION RESULTS")
    report.append("-" * 70)
    
    for i, row in X_input.iterrows():
        report.append(f"\nSample {i+1}:")
        report.append(f"  Input Parameters:")
        report.append(f"    • Angle of Attack (alpha): {row['alpha']:.2f}°")
        report.append(f"    • Mach Number: {row['Mach']:.4f}")
        report.append(f"  Predicted Coefficients:")
        report.append(f"    • Lift Coefficient (Cl): {y_pred[i][0]:.6f}")
        report.append(f"    • Moment Coefficient (Cm): {y_pred[i][1]:.6f}")
        
        # Geometric features
        y_U_cols = [f'y_U{j}' for j in range(1,9)]
        y_L_cols = [f'y_L{j}' for j in range(1,9)]
        thickness = row[y_U_cols].values - row[y_L_cols].values
        camber = (row[y_U_cols].values + row[y_L_cols].values) / 2
        
        report.append(f"  Geometric Features:")
        report.append(f"    • Maximum Thickness: {thickness.max():.6f}")
        report.append(f"    • Mean Thickness: {thickness.mean():.6f}")
        report.append(f"    • Maximum Camber: {camber.max():.6f}")
        report.append(f"    • Mean Camber: {camber.mean():.6f}")
    
    # Statistical Summary
    if len(y_pred) > 1:
        report.append("\n" + "-" * 70)
        report.append("3. STATISTICAL SUMMARY")
        report.append("-" * 70)
        report.append(f"Lift Coefficient (Cl):")
        report.append(f"  • Mean: {y_pred[:, 0].mean():.6f}")
        report.append(f"  • Std Dev: {y_pred[:, 0].std():.6f}")
        report.append(f"  • Min: {y_pred[:, 0].min():.6f}")
        report.append(f"  • Max: {y_pred[:, 0].max():.6f}")
        
        report.append(f"\nMoment Coefficient (Cm):")
        report.append(f"  • Mean: {y_pred[:, 1].mean():.6f}")
        report.append(f"  • Std Dev: {y_pred[:, 1].std():.6f}")
        report.append(f"  • Min: {y_pred[:, 1].min():.6f}")
        report.append(f"  • Max: {y_pred[:, 1].max():.6f}")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def create_results_json(X_input, y_pred, validation_errors):
    """Create a JSON file with structured results"""
    results = {
        "metadata": {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_samples": len(X_input),
            "model_version": "XGBoost CFD v1.0"
        },
        "validation": {
            "status": "passed" if not any(e.startswith('❌') for e in validation_errors) else "failed",
            "errors": validation_errors
        },
        "predictions": []
    }
    
    y_U_cols = [f'y_U{i}' for i in range(1,9)]
    y_L_cols = [f'y_L{i}' for i in range(1,9)]
    
    for i, row in X_input.iterrows():
        thickness = row[y_U_cols].values - row[y_L_cols].values
        camber = (row[y_U_cols].values + row[y_L_cols].values) / 2
        
        sample_data = {
            "sample_id": i + 1,
            "input_parameters": {
                "alpha": float(row['alpha']),
                "mach": float(row['Mach']),
                "upper_surface": [float(row[col]) for col in y_U_cols],
                "lower_surface": [float(row[col]) for col in y_L_cols]
            },
            "predictions": {
                "lift_coefficient": float(y_pred[i][0]),
                "moment_coefficient": float(y_pred[i][1])
            },
            "geometric_features": {
                "max_thickness": float(thickness.max()),
                "mean_thickness": float(thickness.mean()),
                "max_camber": float(camber.max()),
                "mean_camber": float(camber.mean())
            }
        }
        results["predictions"].append(sample_data)
    
    if len(y_pred) > 1:
        results["statistics"] = {
            "lift_coefficient": {
                "mean": float(y_pred[:, 0].mean()),
                "std": float(y_pred[:, 0].std()),
                "min": float(y_pred[:, 0].min()),
                "max": float(y_pred[:, 0].max())
            },
            "moment_coefficient": {
                "mean": float(y_pred[:, 1].mean()),
                "std": float(y_pred[:, 1].std()),
                "min": float(y_pred[:, 1].min()),
                "max": float(y_pred[:, 1].max())
            }
        }
    
    return json.dumps(results, indent=2)


st.title("Airfoil Aerodynamics Predictor")
st.write("Predict Lift Coefficient (Cl) and Moment Coefficient (Cm) from airfoil geometry and flow conditions.")

# Display expected input format
with st.expander("ℹ️ Expected Input Format"):
    st.write("""
    **Required Columns (in order):**
    - `y_U1` to `y_U8`: Upper surface coordinates (8 points)
    - `y_L1` to `y_L8`: Lower surface coordinates (8 points)
    - `alpha`: Angle of attack (degrees, typically -20° to +20°)
    - `Mach`: Mach number (0 to 5, typically 0 to 1 for subsonic)
    
    **Physical Constraints:**
    - Upper surface must be above lower surface (y_U > y_L)
    - Mach number must be non-negative
    - All values must be numeric with no missing data
    """)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with 18 inputs (same order as required)", type="csv")

if uploaded_file is not None:
    try:
        X_input = pd.read_csv(uploaded_file)
        
        # الأعمدة المطلوبة للرفع
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
                    
                    # Display results
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
                    
                    # Export Results Section
                    st.divider()
                    st.subheader("📥 Export Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # CSV Export
                    with col1:
                        csv_buffer = create_results_csv(X_input, y_pred, y_U_cols, y_L_cols)
                        st.download_button(
                            label="📊 Download as CSV",
                            data=csv_buffer,
                            file_name=f"airfoil_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    # TXT Report Export
                    with col2:
                        txt_report = create_results_txt(X_input, y_pred, validation_errors)
                        st.download_button(
                            label="📄 Download Report (TXT)",
                            data=txt_report,
                            file_name=f"airfoil_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    # JSON Export
                    with col3:
                        json_data = create_results_json(X_input, y_pred, validation_errors)
                        st.download_button(
                            label="📋 Download as JSON",
                            data=json_data,
                            file_name=f"airfoil_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Display statistics if multiple samples
                    if len(y_pred) > 1:
                        st.divider()
                        st.subheader("📈 Statistical Summary")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Lift Coefficient (Cl)**")
                            stats_cl = pd.DataFrame({
                                'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                                'Value': [
                                    f"{y_pred[:, 0].mean():.6f}",
                                    f"{y_pred[:, 0].std():.6f}",
                                    f"{y_pred[:, 0].min():.6f}",
                                    f"{y_pred[:, 0].max():.6f}"
                                ]
                            })
                            st.dataframe(stats_cl, hide_index=True)
                        
                        with col2:
                            st.write("**Moment Coefficient (Cm)**")
                            stats_cm = pd.DataFrame({
                                'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                                'Value': [
                                    f"{y_pred[:, 1].mean():.6f}",
                                    f"{y_pred[:, 1].std():.6f}",
                                    f"{y_pred[:, 1].min():.6f}",
                                    f"{y_pred[:, 1].max():.6f}"
                                ]
                            })
                            st.dataframe(stats_cm, hide_index=True)
            else:
                st.error("❌ Please fix the validation errors before proceeding.")
                
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")