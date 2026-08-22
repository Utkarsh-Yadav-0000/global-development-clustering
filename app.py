# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Global Development Cluster Analysis",
    page_icon="🌍",
    layout="wide"
)

# Custom button styling
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #f28c28;
        border-color: #f28c28;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #e67e17;
        border-color: #e67e17;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Loading the Saved Artifacts
@st.cache_resource
def load_models():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    metadata = joblib.load("model_metadata.pkl")
    imputation_medians = joblib.load("imputation_medians.pkl")
    
    return (
        model,
        scaler,
        feature_columns,
        metadata,
        imputation_medians
    )

# Loading the Country Development Profiles Data
@st.cache_data
def load_country_data():
    return pd.read_csv("country_development_profiles.csv")


(model, scaler, feature_columns, metadata, imputation_medians) = load_models()
country_data = load_country_data()

# Preprocessing Function for New Country Data
def preprocess_new_country(input_df):
    """
    Applying the same preprocessing used during model training.
    """
    
    # Maintaining the exact training feature order
    input_df = input_df[feature_columns].copy()
    
    # Filling missing values using training medians
    input_df = input_df.fillna(imputation_medians)
    
    # Applying log1p transformation
    input_df[metadata['log_transformed_features']] = np.log1p(
        input_df[metadata['log_transformed_features']]
    )
    
    # Applying the fitted scaler
    X_new_scaled = scaler.transform(input_df)
    
    return X_new_scaled

# Defining Cluster Descriptions
cluster_descriptions = {
    0: {
        "name": "Lower Development Profile",
        "description": (
            "Countries in this cluster generally show lower "
            "GDP and health expenditure per capita, lower "
            "internet and mobile usage, lower life expectancy, "
            "higher birth and infant mortality rates, and a "
            "younger population structure."
        )
    },
    1: {
        "name": "Higher Development Profile",
        "description": (
            "Countries in this cluster generally show higher "
            "GDP and health expenditure per capita, higher "
            "internet and mobile usage, higher life expectancy, "
            "lower birth and infant mortality rates, and a "
            "more mature population structure."
        )
    }
}

# Sidebar Navigation
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


def set_page(page_name):
    st.session_state.page = page_name


st.sidebar.title("🗺️ Navigation")

st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Country Analysis",
        "Cluster Explorer"
    ],
    index=[
        "Dashboard",
        "Country Analysis",
        "Cluster Explorer"
    ].index(
        st.session_state.page
        if st.session_state.page in [
            "Dashboard",
            "Country Analysis",
            "Cluster Explorer"
        ]
        else "Dashboard"
    ),
    key="navigation_radio",
    on_change=lambda: set_page(
        st.session_state.navigation_radio
    ),
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.button(
    "🧮 New Country Prediction",
    width="stretch",
    on_click=set_page,
    args=("New Country Prediction",)
)

page = st.session_state.page

# Dashboard Page
if page == "Dashboard":

    st.title("🌍 Global Development Cluster Analysis")

    st.markdown(
        """
        This application analyzes development indicators for countries
        across the globe and assigns countries to development profiles
        using clustering techniques.
        """
    )

    st.divider()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Countries",
        country_data["Country"].nunique()
    )

    col2.metric(
        "Development Indicators",
        len(feature_columns)
    )

    col3.metric(
        "Final Model",
        "K-Means"
    )

    col4.metric(
        "Clusters",
        model.n_clusters
    )

    st.subheader("Model Comparison")
    # Model comparison results from the clustering analysis
    comparison_display = pd.DataFrame({
        "Model": [
            "K-Means",
            "Hierarchical - Complete Linkage",
            "DBSCAN"
        ],
        "Clusters": [
            2,
            2,
            3
        ],
        "Silhouette Score": [
            0.309315,
            0.294261,
            0.266013
        ],
        "Coverage": [
            "100%",
            "100%",
            "73.08%"
        ]
    })

    st.dataframe(
        comparison_display,
        width="stretch",
        hide_index=True
    )

    st.info(
        "K-Means was selected as the final deployment model "
        "because it provided the highest Silhouette Score among "
        "the models with complete country coverage."
    )

# Country Analysis Page
elif page == "Country Analysis":

    st.title("🔎 Country Development Analysis")

    selected_country = st.selectbox(
        "Select a country",
        sorted(country_data["Country"].unique())
    )

    country_row = country_data[
        country_data["Country"] == selected_country
    ].iloc[0]

    cluster = int(country_row["KMeans_Cluster"])

    cluster_name = cluster_descriptions[
        cluster
    ]["name"]

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        st.metric(
            "Country",
            selected_country
        )

    with col2:
        st.metric(
            "K-Means Cluster",
            f"Cluster {cluster}"
        )

    with col3:
        st.metric(
            "Development Profile",
            cluster_name
        )
    st.info(
        cluster_descriptions[cluster]["description"]
    )

    st.subheader("Development Indicators")

    indicators = pd.DataFrame({
        "Indicator": feature_columns,
        "Country Value": [
            country_row[col]
            for col in feature_columns
        ]
    })

    st.dataframe(
        indicators,
        width="stretch",
        hide_index=True
    )
    # Country vs Cluster Comparison
    st.subheader("Country vs Cluster Average")

    cluster_average = (
        country_data[
            country_data["KMeans_Cluster"] == cluster
        ][feature_columns]
        .mean()
    )

    comparison = pd.DataFrame({
        "Indicator": feature_columns,
        "Country Value": [
            country_row[col]
            for col in feature_columns
        ],
        "Cluster Average": [
            cluster_average[col]
            for col in feature_columns
        ]
    })
 
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )

# Cluster Explorer Page
elif page == "Cluster Explorer":

    st.title("📊 Cluster Explorer")

    selected_cluster = st.selectbox(
        "Select a cluster",
        sorted(country_data["KMeans_Cluster"].unique()),
        format_func=lambda x: (
            f"Cluster {x} — {cluster_descriptions[x]['name']}"
        )
    )

    cluster_name = cluster_descriptions[
        selected_cluster
    ]["name"]

    st.header(
        f"Cluster {selected_cluster} — {cluster_name}"
    )

    st.write(
        cluster_descriptions[
            selected_cluster
        ]["description"]
    )

    cluster_data = country_data[
        country_data["KMeans_Cluster"] == selected_cluster
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Countries in Cluster",
        len(cluster_data)
    )

    col2.metric(
        "Percentage of Countries",
        f"{len(cluster_data) / len(country_data) * 100:.2f}%"
    )

    st.subheader("Countries")

    st.dataframe(
        cluster_data[
            ["Country"]
        ].sort_values("Country"),
        width="stretch",
        hide_index=True
    )
    # Cluster profile
    st.subheader("Average Development Indicators")

    profile = (
        cluster_data[feature_columns]
        .mean()
        .reset_index()
    )

    profile.columns = [
        "Indicator",
        "Cluster Average"
    ]

    st.dataframe(
        profile,
        width="stretch",
        hide_index=True
    )

# New Country Prediction Page
elif page == "New Country Prediction":

    st.title("🧮 New Country Prediction")

    st.markdown(
        """
        Enter the development indicators for a new country.
        The trained K-Means model will assign the country to
        one of the two development profiles.
        """
    )

    st.divider()
    country_name = st.text_input(
        "Country Name",
        placeholder="Enter country name"
    )

    st.subheader("👥 Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        birth_rate = st.number_input(
            "Birth Rate",
            min_value=0.0,
            value=0.02,
            format="%.6f"
        )

        infant_mortality = st.number_input(
            "Infant Mortality Rate",
            min_value=0.0,
            value=0.02,
            format="%.6f"
        )

        life_expectancy_female = st.number_input(
            "Life Expectancy Female",
            min_value=0.0,
            value=75.0,
            format="%.2f"
        )

    with col2:
        life_expectancy_male = st.number_input(
            "Life Expectancy Male",
            min_value=0.0,
            value=69.0,
            format="%.2f"
        )

        population_0_14 = st.number_input(
            "Population 0-14",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            format="%.6f"
        )

        population_15_64 = st.number_input(
            "Population 15-64",
            min_value=0.0,
            max_value=1.0,
            value=0.64,
            format="%.6f"
        )

    with col3:
        population_65_plus = st.number_input(
            "Population 65+",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            format="%.6f"
        )

        population_total = st.number_input(
            "Population Total",
            min_value=0,
            value=5700000
        )

        population_urban = st.number_input(
            "Population Urban",
            min_value=0.0,
            max_value=1.0,
            value=0.56,
            format="%.6f"
        )

    st.subheader("💼 Economy & Business")

    col1, col2, col3 = st.columns(3)

    with col1:
        gdp = st.number_input(
            "GDP",
            min_value=0.0,
            value=1.7e10,
            format="%.2f"
        )

        business_tax_rate = st.number_input(
            "Business Tax Rate",
            min_value=0.0,
            value=41.0,
            format="%.4f"
        )

    with col2:
        days_to_start_business = st.number_input(
            "Days to Start Business",
            min_value=0.0,
            value=27.0,
            format="%.2f"
        )

        ease_of_business = st.number_input(
            "Ease of Business",
            min_value=0.0,
            value=94.0,
            format="%.2f"
        )

    with col3:
        hours_to_do_tax = st.number_input(
            "Hours to do Tax",
            min_value=0.0,
            value=241.0,
            format="%.2f"
        )

        lending_interest = st.number_input(
            "Lending Interest",
            min_value=0.0,
            value=0.12,
            format="%.6f"
        )

    st.subheader("🏥 Health & Technology")

    col1, col2, col3 = st.columns(3)

    with col1:
        health_exp_gdp = st.number_input(
            "Health Exp % GDP",
            min_value=0.0,
            value=0.06,
            format="%.6f"
        )

        health_exp_capita = st.number_input(
            "Health Exp/Capita",
            min_value=0.0,
            value=220.0,
            format="%.2f"
        )

    with col2:
        internet_usage = st.number_input(
            "Internet Usage",
            min_value=0.0,
            max_value=1.0,
            value=0.17,
            format="%.6f"
        )

        mobile_phone_usage = st.number_input(
            "Mobile Phone Usage",
            min_value=0.0,
            max_value=1.0,
            value=0.56,
            format="%.6f"
        )

    st.subheader("🌱 Environment & Tourism")

    col1, col2, col3 = st.columns(3)

    with col1:
        co2_emissions = st.number_input(
            "CO2 Emissions",
            min_value=0.0,
            value=7856.0,
            format="%.2f"
        )

        energy_usage = st.number_input(
            "Energy Usage",
            min_value=0.0,
            value=8974.0,
            format="%.2f"
        )

    with col2:
        tourism_inbound = st.number_input(
            "Tourism Inbound",
            min_value=0.0,
            value=6.57e8,
            format="%.2f"
        )

        tourism_outbound = st.number_input(
            "Tourism Outbound",
            min_value=0.0,
            value=4.35e8,
            format="%.2f"
        )

    # Prediction Button
    st.divider()

    predict_button = st.button(
        "▶️ Predict Country Cluster",
        type="primary",
        width="stretch"
    )
    if predict_button:

        new_country_data = {
            "Birth Rate": birth_rate,
            "Business Tax Rate": business_tax_rate,
            "CO2 Emissions": co2_emissions,
            "Days to Start Business": days_to_start_business,
            "Ease of Business": ease_of_business,
            "Energy Usage": energy_usage,
            "GDP": gdp,
            "Health Exp % GDP": health_exp_gdp,
            "Health Exp/Capita": health_exp_capita,
            "Hours to do Tax": hours_to_do_tax,
            "Infant Mortality Rate": infant_mortality,
            "Internet Usage": internet_usage,
            "Lending Interest": lending_interest,
            "Life Expectancy Female": life_expectancy_female,
            "Life Expectancy Male": life_expectancy_male,
            "Mobile Phone Usage": mobile_phone_usage,
            "Population 0-14": population_0_14,
            "Population 15-64": population_15_64,
            "Population 65+": population_65_plus,
            "Population Total": population_total,
            "Population Urban": population_urban,
            "Tourism Inbound": tourism_inbound,
            "Tourism Outbound": tourism_outbound
        }

        input_df = pd.DataFrame(
            [new_country_data]
        )

        # Ensuring exact feature order used during training
        input_df = input_df[feature_columns]

        # Displaying Input Summary
        st.subheader("Input Summary")

        input_summary = input_df.T.rename(
            columns={0: "Input Value"}
        ).reset_index()

        input_summary.columns = ["Feature", "Input Value"]
        
        # Converting all input values to string for better display
        input_summary["Input Value"] = input_summary["Input Value"].astype(str)

        st.dataframe(
            input_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn(
                    "Feature",
                    width="stretch"
                ),
                "Input Value": st.column_config.TextColumn(
                    "Input Value",
                    width="stretch"
                )
            }
        )

        # Applying the same preprocessing used during model training
        X_new_scaled = preprocess_new_country(input_df)

        # Predicting Cluster
        predicted_cluster = int(
            model.predict(X_new_scaled)[0]
        )

        predicted_profile = cluster_descriptions[
            predicted_cluster
        ]["name"]   

        predicted_description = cluster_descriptions[
            predicted_cluster
        ]["description"]       

        # Calculating distance from the new country
        # to each K-Means cluster centroid
        distances = model.transform(X_new_scaled)[0]

        distance_df = pd.DataFrame({
            "Cluster": [
                "Cluster 0",
                "Cluster 1"
            ],
            "Development Profile": [
                cluster_descriptions[0]["name"],
                cluster_descriptions[1]["name"]
            ],
            "Distance from Centroid": distances
        })

        distance_df["Distance from Centroid"] = (
            distance_df["Distance from Centroid"].round(4)
        )

        # Displaying Prediction Results
        st.divider()

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Assigned Cluster",
                f"Cluster {predicted_cluster}"
            )

        with result_col2:
            st.metric(
                "Development Profile",
                predicted_profile
            )

        display_name = country_name.strip() if country_name.strip() else "The new country"

        st.success(
            f"**{display_name}** is assigned to **Cluster {predicted_cluster}**, "
            f"which corresponds to the **{predicted_profile}**."
        )
        st.info(
            predicted_description
        )
        st.caption(
            "The country is assigned to the cluster whose centroid is closest "
            "in the standardized feature space."
        )

        st.subheader("📏 Distance from Cluster Centroids")

        st.dataframe(
            distance_df,
            width="stretch",
            hide_index=True
        ) 

   