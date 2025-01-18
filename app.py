# Library Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
import seaborn as sns
import time
import joblib  # For saving/loading machine learning models

# Define valid credentials (for demo purposes)
VALID_USERS = {
    "admin": "password123",
    "user1": "pass456",
    "guest": "guest789"
}

# Page Configuration
st.set_page_config(page_title="AI-Enhanced DLP Tool", layout="wide")

# User Authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state['authenticated'] = True
            st.success("Login successful! Please refresh the page if not redirected.")
        else:
            st.error("Invalid username or password.")
else:
    # Logout Button
    if st.button("Logout"):
        st.session_state['authenticated'] = False

    # Sidebar Navigation
    options = st.sidebar.radio(
        "Choose a Section:",
        ("Home", "Upload Data", "AI Anomaly Detection", "Advanced Insights", "Real-Time Monitoring", "Settings")
    )

    # Home Section
    if options == "Home":
        st.title("Welcome to the AI-Enhanced DLP Tool")
        st.markdown("Please choose a module from the sidebar.")

    # Upload Data Section
    elif options == "Upload Data":
        st.title("Upload Dataset")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df  # Save dataset in session state
            st.success("Dataset uploaded successfully!")
            st.write("Preview of the uploaded dataset:")
            st.dataframe(df.head())

    # AI Anomaly Detection Section
    elif options == "AI Anomaly Detection":
        st.title("AI-Powered Anomaly Detection")

        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("Dataset Overview:")
            st.dataframe(df.head())

            # Configurable Settings
            st.markdown("### Configurable Anomaly Detection Settings")
            contamination = st.slider("Set Contamination Level (Anomalies %)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
            random_state = st.number_input("Set Random State (Optional)", min_value=0, value=42, step=1)

            # Feature Selection
            st.markdown("Select Features for Anomaly Detection")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = st.multiselect("Choose numeric features", numeric_columns, default=numeric_columns)

            # Model Saving and Loading
            st.markdown("### Save or Load a Model")
            save_model = st.checkbox("Save trained model")
            load_model_file = st.file_uploader("Upload a pre-trained model (optional)", type="pkl")

            model = None
            if load_model_file:
                model = joblib.load(load_model_file)
                st.success("Pre-trained model loaded successfully!")

            try:
                if len(selected_features) >= 2:
                    # Anomaly Detection
                    df_filtered = df[selected_features].dropna()
                    if not model:  # Train a new model if none is loaded
                        model = IsolationForest(contamination=contamination, random_state=random_state)
                        model.fit(df_filtered)
                    anomaly_scores = model.decision_function(df_filtered)
                    anomalies = model.predict(df_filtered)

                    df['Anomaly Score'] = np.nan
                    df['Anomaly'] = 'No'
                    df.loc[df_filtered.index, 'Anomaly Score'] = anomaly_scores
                    df.loc[df_filtered.index, 'Anomaly'] = ['Yes' if x == -1 else 'No' for x in anomalies]

                    st.success("Anomaly detection completed!")
                    st.write("Detected Anomalies:")
                    anomalies_df = df[df['Anomaly'] == 'Yes']
                    st.dataframe(anomalies_df)

                    # Save Model
                    if save_model:
                        model_path = "trained_anomaly_model.pkl"
                        joblib.dump(model, model_path)
                        st.success(f"Model saved as {model_path}")

                    # Export Results
                    st.markdown("### Export Results")
                    csv_buffer = io.StringIO()
                    anomalies_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download Anomalies as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="anomalies.csv",
                        mime="text/csv"
                    )

                    # Visualization
                    st.markdown("Anomaly Visualization")
                    x_feature = selected_features[0]
                    y_feature = selected_features[1]
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(
                        df_filtered[x_feature],
                        df_filtered[y_feature],
                        c=anomalies,
                        cmap="coolwarm",
                        s=50
                    )
                    ax.set_title("Anomaly Detection Visualization")
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    st.pyplot(fig)
                else:
                    st.warning("Please select at least two numeric features for analysis.")
            except Exception as e:
                st.error(f"Error during anomaly detection: {e}")
        else:
            st.warning("Please upload data in the **Upload Data** section first.")

     # Advanced Insights Section
    elif options == "Advanced Insights":
        st.title("Advanced Insights Dashboard")

        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("Dataset Overview:")
            st.dataframe(df.head())

            col1, col2, col3 = st.columns([1.2, 1.5, 1.3])  # Adjust column widths for better spacing

            # Bar Chart
            with col1:
                st.markdown("### Histogram")
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_feature = st.selectbox("Select a numeric feature for histogram:", numeric_columns, key="bar_chart")
                if selected_feature:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.hist(df[selected_feature], bins=20, color="skyblue", edgecolor="black")
                    ax.set_title(f"Distribution of {selected_feature}", fontsize=12)
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

            # Heatmap
            with col2:
                st.markdown("### Correlation Heatmap")
                if len(numeric_columns) > 1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    correlation_matrix = df[numeric_columns].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation Heatmap", fontsize=12)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough numeric features for correlation analysis.")

            # Pie Chart
            with col3:
                st.markdown("### Pie Chart")
                categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
                selected_category = st.selectbox("Select a categorical feature for pie chart:", categorical_columns, key="pie_chart")
                if selected_category:
                    category_counts = df[selected_category].value_counts()
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f"Distribution of {selected_category}", fontsize=12)
                    st.pyplot(fig)
        else:
            st.warning("Please upload data in the **Upload Data** section first.")

    # Real-Time Monitoring Section
    elif options == "Real-Time Monitoring":
        st.title("Real-Time Monitoring (Simulated Data Stream)")

        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("Initial Dataset Overview:")
            st.dataframe(df.head())

            st.markdown("### Stream Settings")
            stream_speed = st.slider("Stream Speed (seconds per update)", min_value=1, max_value=10, value=2)
            start_stream = st.button("Start Stream")
            if start_stream:
                placeholder = st.empty()
                for i in range(10):
                    new_row = {
                        "UserID": f"U{100 + i}",
                        "File Size (MB)": np.random.randint(1, 500),
                        "Access Frequency": np.random.randint(0, 50)
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    placeholder.dataframe(df)
                    time.sleep(stream_speed)
        else:
            st.warning("Please upload data in the **Upload Data** section first.")

    # Settings Section
    elif options == "Settings":
        st.title("Settings")
        dark_mode = st.checkbox("Enable Dark Mode")
        st.write(f"Dark Mode is {'enabled' if dark_mode else 'disabled'}.")
