from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.inspection import permutation_importance
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io
import base64
import logging
import os
from datetime import datetime
import time
import threading
import webbrowser
import warnings; warnings.filterwarnings('ignore')

# Configure logging
os.makedirs('tmp', exist_ok=True)
logging.basicConfig(
    filename='tmp/app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

app = Flask(__name__)

# Global variables
global_data = None
global_model = None
feature_importance = None
model_accuracy = None
churn_rate = None
is_churn = None
current_revenue = 0
model_type = None  # "classification" or "regression"


def to_binary_churn(series):
    """
    Safely convert churn-like values to binary (0/1) whenever possible.
    Handles yes/no, true/false, 1/0, strings, booleans.
    """
    s = series.copy()

    mapping = {
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        True: 1, False: 0
    }

    # If object/string, normalize first
    if s.dtype == 'object':
        s = s.astype(str).str.strip().str.lower()
        s = s.replace(mapping)
    else:
        s = s.replace(mapping)

    # Try numeric conversion
    s_num = pd.to_numeric(s, errors='coerce')

    # If mostly valid after conversion, use numeric
    if s_num.notna().sum() > 0:
        return s_num

    return s


def is_classification_target(y):
    """
    Decide whether target should be treated as classification or regression.
    """
    y_temp = y.copy()

    # First try binary churn conversion
    y_bin = to_binary_churn(y_temp)
    y_num = pd.to_numeric(y_bin, errors='coerce')

    # If fully numeric and only few unique discrete values => classification
    if y_num.notna().all():
        unique_vals = np.sort(y_num.dropna().unique())
        nunique = len(unique_vals)

        # Binary or low-cardinality integer-like target => classification
        if nunique <= 10:
            if np.all(np.equal(np.mod(unique_vals, 1), 0)):
                return True

        # Exactly {0,1}
        if set(unique_vals).issubset({0, 1}):
            return True

        # Otherwise continuous numeric => regression
        return False

    # Non-numeric categorical => classification
    return True


def clean_data(df):
    """Clean and preprocess the input dataframe."""
    try:
        logging.info("Starting data cleaning process")

        # Validate dataset
        if df.empty:
            raise ValueError("Uploaded dataset is empty")
        if len(df) < 10:
            raise ValueError("Dataset too small (minimum 10 rows required)")

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Strip string values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # Convert TotalCharges to numeric if present
        if 'totalcharges' in df.columns:
            df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

        # Convert MonthlyCharges to numeric if present
        if 'monthlycharges' in df.columns:
            df['monthlycharges'] = pd.to_numeric(df['monthlycharges'], errors='coerce')

        # Convert tenure if present
        if 'tenure' in df.columns:
            df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

        # Recalculate cols after type conversions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Handle missing values
        for col in numeric_cols:
            if df[col].isnull().all():
                raise ValueError(f"Column {col} contains only missing/invalid values")
            df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            if df[col].isnull().all():
                raise ValueError(f"Column {col} contains only missing values")
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            df[col] = df[col].fillna(fill_val)

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {initial_rows - len(df)} duplicates")

        # Remove ID columns except churn if name accidentally contains id
        id_columns = [col for col in df.columns if 'id' in col.lower() and col.lower() != 'churn']
        df = df.drop(columns=id_columns, errors='ignore')

        # Calculate data quality score
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_ratio = (initial_rows - len(df)) / initial_rows if initial_rows > 0 else 0
        data_quality_score = 100 * (1 - (missing_ratio + duplicate_ratio) / 2)

        logging.info("Data cleaning completed successfully")
        return df, data_quality_score

    except Exception as e:
        logging.error(f"Error in clean_data: {str(e)}")
        raise


def prepare_and_train_model(df, churn_col):
    """
    Shared training logic for both /upload and /filter_by_date.
    Auto-selects classification or regression based on target type.
    """
    global global_model, feature_importance, model_accuracy, churn_rate, model_type

    # Calculate churn rate using safe binary conversion when possible
    churn_values = to_binary_churn(df[churn_col])
    churn_numeric = pd.to_numeric(churn_values, errors='coerce')

    if churn_numeric.notna().sum() == 0:
        raise ValueError("Churn column contains invalid values")

    # For dashboard churn rate, use binary if possible; else fallback to normalized mean if numeric
    unique_numeric = set(churn_numeric.dropna().unique())
    if unique_numeric.issubset({0, 1}):
        churn_rate = float(churn_numeric.mean())
    else:
        # If continuous target, dashboard still needs a value; use normalized proportion-ish indicator
        # Better than crashing. If target > 1 scale, min-max normalize.
        y_min = churn_numeric.min()
        y_max = churn_numeric.max()
        if y_max > y_min:
            churn_rate = float(((churn_numeric - y_min) / (y_max - y_min)).mean())
        else:
            churn_rate = 0.0

    logging.info(f"Computed churn indicator rate: {churn_rate*100:.2f}%")

    # Build model dataframe
    df_model = df.copy()

    # Encode object columns individually
    for col in df_model.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model.drop(churn_col, axis=1)
    y_raw = df[churn_col].copy()

    if X.empty or len(X.columns) < 1:
        raise ValueError("No valid features for model training")

    # Determine task type
    if is_classification_target(y_raw):
        model_type = "classification"
        y = to_binary_churn(y_raw)

        # If still not numeric, label encode
        y_num = pd.to_numeric(y, errors='coerce')
        if not y_num.notna().all():
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
        else:
            y = y_num.astype(int)

        # Ensure enough classes
        if len(pd.Series(y).unique()) < 2:
            raise ValueError("Churn target must have at least 2 classes for classification")

    else:
        model_type = "regression"
        y = pd.to_numeric(y_raw, errors='coerce')
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        if len(y) < 10:
            raise ValueError("Not enough valid target rows after cleaning for regression")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    if model_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    if len(X_train) < 5 or len(X_test) < 2:
        raise ValueError("Dataset too small for model training")

    # Train model WITHOUT GridSearchCV (stable fix)
    if model_type == "classification":
        global_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        )
    else:
        global_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        )

    global_model.fit(X_train, y_train)

    # Predict
    y_pred = global_model.predict(X_test)

    # Metric
    if model_type == "classification":
        model_accuracy = float(accuracy_score(y_test, y_pred))
        logging.info(f"Classification accuracy: {model_accuracy*100:.2f}%")
    else:
        model_accuracy = float(r2_score(y_test, y_pred))
        logging.info(f"Regression R2 score: {model_accuracy:.4f}")

    # Feature importance (prefer model.feature_importances_)
    if hasattr(global_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': global_model.feature_importances_
        })
    else:
        # Fallback
        perm_importance = permutation_importance(
            global_model, X_test, y_test, n_repeats=5, random_state=42
        )
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    feature_importance = importance_df.to_dict(orient='records')
    logging.info(f"Feature importance: {feature_importance[:5]}")

    return {
        'churn_rate': float(churn_rate),
        'model_accuracy': float(model_accuracy),
        'model_type': model_type
    }


def generate_charts(df):
    """Generate charts for visualization with error handling."""
    try:
        logging.info("Generating charts")
        churn_col = is_churn
        charts = {}

        if churn_col not in df.columns:
            return charts

        churn_series = to_binary_churn(df[churn_col])
        churn_numeric = pd.to_numeric(churn_series, errors='coerce')

        # Churn Distribution (only if binary-ish)
        if churn_numeric.notna().sum() > 0 and set(churn_numeric.dropna().unique()).issubset({0, 1}):
            plt.figure(figsize=(8, 6))
            churn_counts = churn_numeric.value_counts().sort_index()
            labels = ['Retained', 'Churned'] if 0 in churn_counts.index and 1 in churn_counts.index else ['Churned']
            plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'][:len(churn_counts)])
            plt.title('Customer Churn Distribution', fontweight='bold')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            charts['churn_distribution'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        # Tenure vs Churn
        if 'tenure' in df.columns and not df['tenure'].isnull().all() and model_type == "classification":
            plot_df = df.copy()
            plot_df['_churn_binary'] = pd.to_numeric(to_binary_churn(df[churn_col]), errors='coerce')
            plot_df = plot_df.dropna(subset=['_churn_binary'])

            if not plot_df.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=plot_df, x='tenure', hue='_churn_binary', multiple='stack')
                plt.title("Tenure vs Churn", fontweight='bold')
                plt.xlabel("Tenure (Months)", fontweight='bold')
                plt.ylabel("Count", fontweight='bold')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                charts['tenure_vs_churn'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

        # Monthly Charges vs Churn
        if 'monthlycharges' in df.columns and model_type == "classification":
            plot_df = df.copy()
            plot_df['_churn_binary'] = pd.to_numeric(to_binary_churn(df[churn_col]), errors='coerce')
            plot_df = plot_df.dropna(subset=['_churn_binary'])

            if not plot_df.empty:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=plot_df, x='monthlycharges', hue='_churn_binary', fill=True)
                plt.title("Monthly Charges vs Churn", fontweight='bold')
                plt.xlabel("Monthly Charges", fontweight='bold')
                plt.ylabel("Density", fontweight='bold')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                charts['charges_vs_churn'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

        # Contract Type vs Churn
        if 'contract' in df.columns and not df['contract'].isnull().all() and model_type == "classification":
            plot_df = df.copy()
            plot_df['_churn_binary'] = pd.to_numeric(to_binary_churn(df[churn_col]), errors='coerce')
            plot_df = plot_df.dropna(subset=['_churn_binary'])

            if not plot_df.empty:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=plot_df, x='contract', hue='_churn_binary')
                plt.title("Churn by Contract Type", fontweight='bold')
                plt.xlabel("Contract Type", fontweight='bold')
                plt.ylabel("Count", fontweight='bold')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                charts['contract_vs_churn'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

        # Feature Importance
        if feature_importance is not None:
            plt.figure(figsize=(12, 6))
            importance_df = pd.DataFrame(feature_importance[:10])
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='#4B5EAA')
            plt.xlabel("Feature Importance", fontweight='bold')
            plt.ylabel("Features", fontweight='bold')
            plt.title("Most Important Features for Churn Prediction", fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            charts['feature_importance'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        # Churn Rate Over Time
        if 'tenure' in df.columns and 'monthlycharges' in df.columns and model_type == "classification":
            plot_df = df.copy()
            plot_df['_churn_binary'] = pd.to_numeric(to_binary_churn(df[churn_col]), errors='coerce')
            plot_df = plot_df.dropna(subset=['_churn_binary'])

            if not plot_df.empty:
                plt.figure(figsize=(12, 6))
                tenure_bins = pd.cut(plot_df['tenure'], bins=10)
                churn_by_tenure = plot_df.groupby(tenure_bins, observed=True)['_churn_binary'].mean() * 100
                revenue_loss = plot_df.groupby(tenure_bins, observed=True).apply(
                    lambda x: x['_churn_binary'].sum() * x['monthlycharges'].mean()
                )

                ax1 = plt.gca()
                line1, = ax1.plot(
                    churn_by_tenure.index.astype(str),
                    churn_by_tenure.values,
                    marker='o',
                    color='#D32F2F',
                    label='Churn Rate (%)'
                )
                ax1.set_xlabel("Tenure Range", fontweight='bold')
                ax1.set_ylabel("Churn Rate (%)", fontweight='bold')
                ax1.set_title("Churn Rate and Revenue Loss Over Tenure", fontweight='bold')

                ax2 = ax1.twinx()
                line2, = ax2.plot(
                    revenue_loss.index.astype(str),
                    revenue_loss.values,
                    marker='s',
                    color='#1976D2',
                    label='Revenue Loss (₹)'
                )
                ax2.set_ylabel("Revenue Loss (₹)", fontweight='bold')

                lines = [line1, line2]
                labels = [line.get_label() for line in lines]
                ax1.legend(lines, labels, loc='upper right')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                charts['churn_over_time'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

        logging.info(f"Generated {len(charts)} charts successfully")
        return charts

    except Exception as e:
        logging.error(f"Error in generate_charts: {str(e)}")
        return {}


def generate_recommendations(feature_importance):
    """Generate actionable recommendations based on feature importance."""
    try:
        logging.info("Generating recommendations")
        recommendations = []
        for feature in feature_importance[:5]:
            feature_name = feature['Feature'].lower()
            if 'contract' in feature_name:
                recommendations.append("Offer longer-term contracts with discounts to boost loyalty.")
            elif 'tenure' in feature_name:
                recommendations.append("Introduce loyalty programs for long-term customers.")
            elif 'monthlycharges' in feature_name:
                recommendations.append("Optimize pricing with competitive or tiered plans.")
            elif 'totalcharges' in feature_name:
                recommendations.append("Provide discounts for high-spending customers.")
            else:
                recommendations.append(f"Optimize {feature_name} based on customer feedback.")
        recommendations.extend([
            "Engage at-risk customers with personalized offers.",
            "Streamline onboarding to highlight product value."
        ])
        return recommendations
    except Exception as e:
        logging.error(f"Error in generate_recommendations: {str(e)}")
        return []


def generate_pdf_report(data_info, insights, charts, recommendations, data_quality_score, revenue_message):
    """Generate a formatted PDF report."""
    try:
        logging.info("Generating PDF report")
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Customer Churn Analysis Report", styles['Title']))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Dataset Overview
        story.append(Paragraph("Dataset Overview", styles['Heading2']))
        story.append(Paragraph(f"Total Customers: {data_info['rows']}", styles['Normal']))
        story.append(Paragraph(f"Columns: {data_info['columns']}", styles['Normal']))
        story.append(Paragraph(f"Missing Values: {data_info['missing_values']}", styles['Normal']))
        story.append(Paragraph(f"Data Quality Score: {data_quality_score:.2f}%", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Churn Statistics
        story.append(Paragraph("Churn Statistics", styles['Heading2']))
        story.append(Paragraph(f"Churn Rate Indicator: {insights['churn_rate']*100:.2f}%", styles['Normal']))

        if insights['model_type'] == 'classification':
            story.append(Paragraph(f"Model Accuracy: {insights['model_accuracy']*100:.2f}%", styles['Normal']))
        else:
            story.append(Paragraph(f"Model R² Score: {insights['model_accuracy']:.4f}", styles['Normal']))

        if insights['potential_monthly_loss'] is not None and insights['potential_monthly_loss'] > 0:
            story.append(Paragraph(f"Monthly Revenue at Risk: ₹{insights['potential_monthly_loss']:.2f}", styles['Normal']))
            story.append(Paragraph(f"Annual Revenue at Risk: ₹{insights['potential_yearly_loss']:.2f}", styles['Normal']))
        else:
            story.append(Paragraph(revenue_message, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Charts
        story.append(Paragraph("Graphical Analysis", styles['Heading2']))
        for chart_name, chart_data in charts.items():
            if chart_name == 'feature_importance':
                continue
            img_buffer = io.BytesIO(base64.b64decode(chart_data))
            img = Image(img_buffer, width=5*inch, height=3*inch)
            story.append(img)
            story.append(Paragraph(chart_name.replace('_', ' ').title(), styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logging.error(f"Error in generate_pdf_report: {str(e)}")
        raise


@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload and analysis."""
    global global_data, current_revenue, is_churn

    try:
        logging.info("Processing file upload")

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Invalid file format. Please upload CSV.'})

        # Get current_revenue from FormData
        current_revenue = request.form.get('current_revenue', 0)
        try:
            current_revenue = float(current_revenue)
            if current_revenue < 0:
                current_revenue = 0
        except (ValueError, TypeError):
            current_revenue = 0

        df = pd.read_csv(file)
        df, data_quality_score = clean_data(df)
        global_data = df

        # Find Churn column (case-insensitive)
        churn_cols = [col for col in df.columns if col.lower() == 'churn']
        if not churn_cols:
            return jsonify({
                'success': False,
                'data_info': {
                    'rows': int(len(df)),
                    'columns': int(len(df.columns)),
                    'missing_values': int(df.isnull().sum().sum()),
                    'column_names': df.columns.tolist() + ['All'],
                    'data_quality_score': float(data_quality_score)
                },
                'warning': 'No Churn column found.'
            })

        is_churn = churn_cols[0]

        data_info = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'missing_values': int(df.isnull().sum().sum()),
            'column_names': df.columns.tolist() + ['All'],
            'data_quality_score': float(data_quality_score)
        }

        # Train model safely
        training_result = prepare_and_train_model(df, is_churn)

        # Revenue loss based on current_revenue
        monthly_loss = float(current_revenue * churn_rate) if current_revenue > 0 else 0
        yearly_loss = float(monthly_loss * 12) if current_revenue > 0 else 0
        revenue_message = "Current Revenue is 0, no revenue loss predicted" if current_revenue == 0 else ""

        insights = {
            'churn_rate': float(churn_rate),
            'model_accuracy': float(model_accuracy),
            'model_type': model_type,
            'potential_monthly_loss': monthly_loss,
            'potential_yearly_loss': yearly_loss,
            'feature_importance': [
                {'Feature': item['Feature'], 'Importance': float(item['Importance'])}
                for item in feature_importance[:5]
            ]
        }

        charts = generate_charts(df)
        if not charts:
            logging.warning("No charts generated")

        response = {
            'success': True,
            'data_info': data_info,
            'insights': insights,
            'charts': charts
        }

        if revenue_message:
            response['revenue_message'] = revenue_message

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in upload: {str(e)}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})


@app.route('/filter_by_date', methods=['POST'])
def filter_by_date():
    """Filter data by month and year."""
    global global_data, current_revenue

    try:
        logging.info("Processing date filter")

        if global_data is None:
            return jsonify({'success': False, 'error': 'Please upload data first'})

        data = request.json
        month = data.get('month')
        year = data.get('year')

        if not month or not year:
            return jsonify({'success': False, 'error': 'Missing month or year'})

        # Assume SignupDate column
        df = global_data.copy()
        if 'signupdate' not in df.columns:
            return jsonify({'success': False, 'error': 'SignupDate column not found or invalid'})

        df['signupdate'] = pd.to_datetime(df['signupdate'], errors='coerce')
        if df['signupdate'].isna().all():
            return jsonify({'success': False, 'error': 'Invalid SignupDate values'})

        df = df[(df['signupdate'].dt.month == int(month)) & (df['signupdate'].dt.year == int(year))]
        if df.empty:
            return jsonify({'success': False, 'error': 'No data for selected date range'})

        global_data = df

        data_info = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'missing_values': int(df.isnull().sum().sum()),
            'column_names': df.columns.tolist() + ['All'],
            'data_quality_score': float(clean_data(df.copy())[1])
        }

        # Re-train model safely on filtered data
        training_result = prepare_and_train_model(df, is_churn)

        # Use stored current_revenue from last upload
        monthly_loss = float(current_revenue * churn_rate) if current_revenue > 0 else 0
        yearly_loss = float(monthly_loss * 12) if current_revenue > 0 else 0
        revenue_message = "Current Revenue is 0, no revenue loss predicted" if current_revenue == 0 else ""

        insights = {
            'churn_rate': float(churn_rate),
            'model_accuracy': float(model_accuracy),
            'model_type': model_type,
            'potential_monthly_loss': monthly_loss,
            'potential_yearly_loss': yearly_loss,
            'feature_importance': [
                {'Feature': item['Feature'], 'Importance': float(item['Importance'])}
                for item in feature_importance[:5]
            ]
        }

        charts = generate_charts(df)

        response = {
            'success': True,
            'data_info': data_info,
            'insights': insights,
            'charts': charts
        }

        if revenue_message:
            response['revenue_message'] = revenue_message

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in filter_by_date: {str(e)}")
        return jsonify({'success': False, 'error': f'Filter failed: {str(e)}'})


@app.route('/predict_revenue', methods=['POST'])
def predict_revenue():
    """Predict future revenue based on current revenue and churn rate."""
    global current_revenue

    try:
        logging.info("Predicting revenue")

        data = request.json
        current_revenue = data.get('current_revenue', 0)

        try:
            current_revenue = float(current_revenue)
            if current_revenue < 0:
                current_revenue = 0
        except (ValueError, TypeError):
            current_revenue = 0

        if current_revenue == 0:
            return jsonify({
                'success': True,
                'monthly_loss': 0.0,
                'yearly_loss': 0.0,
                'future_revenue': 0.0,
                'message': 'Current Revenue is 0, no loss predicted'
            })

        if churn_rate is None:
            return jsonify({
                'success': False,
                'error': 'Please upload and analyze a dataset first'
            })

        monthly_loss = float(current_revenue * churn_rate)
        yearly_loss = float(monthly_loss * 12)
        future_revenue = float(current_revenue - yearly_loss)

        logging.info(f"Revenue prediction: Monthly loss ₹{monthly_loss:.2f}, Yearly loss ₹{yearly_loss:.2f}")

        return jsonify({
            'success': True,
            'monthly_loss': monthly_loss,
            'yearly_loss': yearly_loss,
            'future_revenue': future_revenue
        })

    except Exception as e:
        logging.error(f"Error in predict_revenue: {str(e)}")
        return jsonify({'success': False, 'error': f'Revenue prediction failed: {str(e)}'})


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot queries."""
    try:
        logging.info("Processing chat request")

        if global_data is None:
            return jsonify({'response': 'Please upload a dataset first.'})

        data = request.json
        query = data.get('query', '').lower()

        # Get current_revenue from JSON
        current_revenue_chat = data.get('current_revenue', 0)
        logging.info(f"Received current_revenue: {current_revenue_chat}")

        try:
            current_revenue_chat = float(current_revenue_chat)
            if current_revenue_chat < 0:
                current_revenue_chat = 0
        except (ValueError, TypeError):
            current_revenue_chat = 0

        if 'churn rate' in query:
            response = f"The current churn indicator rate is {churn_rate*100:.2f}%."

        elif 'revenue' in query or 'loss' in query:
            if current_revenue_chat == 0:
                response = "Current Revenue is 0, no revenue loss predicted."
            else:
                monthly_loss = float(current_revenue_chat * churn_rate)
                yearly_loss = float(monthly_loss * 12)
                response = f"Churn impacts revenue by ₹{monthly_loss:.2f} monthly and ₹{yearly_loss:.2f} annually."

        elif 'reasons' in query or 'factors' in query:
            response = "Top 3 factors for churn:\n" + "\n".join(
                f"{i+1}. {f['Feature']} (Importance: {f['Importance']:.4f})"
                for i, f in enumerate(feature_importance[:3])
            )

        elif 'reduce churn' in query or 'recommendations' in query:
            recommendations = generate_recommendations(feature_importance)
            response = "Recommendations to reduce churn:\n" + "\n".join(
                f"{i+1}. {rec}" for i, rec in enumerate(recommendations)
            )

        elif 'model accuracy' in query or 'accuracy' in query or 'accurate' in query:
            if model_type == "classification":
                response = f"The churn prediction model accuracy is {model_accuracy*100:.2f}%."
            else:
                response = f"The churn prediction model R² score is {model_accuracy:.4f}."

        elif 'trend' in query:
            response = f"Churn rate trend indicator: {churn_rate*100:.2f}% currently, analyze over time in the Insights section."

        elif 'segment' in query:
            if 'contract' in global_data.columns:
                churn_bin = pd.to_numeric(to_binary_churn(global_data[is_churn]), errors='coerce')
                temp_df = global_data.copy()
                temp_df['_churn_binary'] = churn_bin
                temp_df = temp_df.dropna(subset=['_churn_binary'])

                if not temp_df.empty:
                    segment_churn = temp_df.groupby('contract')['_churn_binary'].mean().to_dict()
                    response = "Churn by contract type:\n" + "\n".join(
                        f"{k}: {v*100:.2f}%" for k, v in segment_churn.items()
                    )
                else:
                    response = "Segment analysis unavailable because churn values are not binary."
            else:
                response = "Segment analysis unavailable due to missing contract data."

        else:
            response = "Try asking about churn rate, revenue impact, churn reasons, recommendations, model accuracy, churn trend, or customer segments."

        return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({'response': f'Error: {str(e)}'})


@app.route('/download_report', methods=['GET'])
def download_report():
    """Download a PDF report."""
    try:
        logging.info("Generating PDF report")

        if global_data is None:
            return jsonify({'error': 'No data uploaded yet'})

        data_info = {
            'rows': int(len(global_data)),
            'columns': int(len(global_data.columns)),
            'missing_values': int(global_data.isnull().sum().sum()),
            'data_quality_score': float(clean_data(global_data.copy())[1])
        }

        monthly_loss = float(current_revenue * churn_rate) if current_revenue > 0 else 0
        yearly_loss = float(monthly_loss * 12) if current_revenue > 0 else 0
        revenue_message = "Current Revenue is 0, no revenue loss predicted" if current_revenue == 0 else ""

        insights = {
            'churn_rate': float(churn_rate),
            'model_accuracy': float(model_accuracy),
            'model_type': model_type,
            'potential_monthly_loss': monthly_loss,
            'potential_yearly_loss': yearly_loss
        }

        charts = generate_charts(global_data)
        recommendations = generate_recommendations(feature_importance)

        pdf_buffer = generate_pdf_report(
            data_info,
            insights,
            charts,
            recommendations,
            data_info['data_quality_score'],
            revenue_message
        )

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'churn_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

    except Exception as e:
        logging.error(f"Error in download_report: {str(e)}")
        return jsonify({'error': f'Error generating report: {str(e)}'})


def open_browser():
    time.sleep(1)
    webbrowser.open_new("http://localhost:5000")
    os.system("echo(")
    os.system("echo The webpage has opened successfully.")
    os.system("echo(")
    os.system("echo Press Ctrl+C to close the server.")
    os.system("echo(")
    os.system("echo For more details visit the following files: ")
    os.system("echo # '%CD%\\tmp\\app_output.txt' for console outputs.")
    os.system("echo # '%CD%\\tmp\\app.log' for detailed logs.")
    os.system("echo(")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get("RENDER") is None:
        threading.Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=port, debug=False)