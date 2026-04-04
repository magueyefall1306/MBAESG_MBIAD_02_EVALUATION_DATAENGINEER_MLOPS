import streamlit as st
import pandas as pd
import numpy as np
import json
from snowflake.snowpark.context import get_active_session

session = get_active_session()

st.set_page_config(page_title="Estimation du Prix Immobilier", layout="wide")
st.title("Estimation du Prix Immobilier")
st.markdown("Renseignez les caractéristiques de la maison et obtenez une estimation de prix.")

@st.cache_data
def load_reference_data():
    return session.table("HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE").to_pandas().drop_duplicates()

@st.cache_data
def load_scaler_params():
    df = session.table("HOUSE_PRICE_DB.ML_SCHEMA.SCALER_PARAMS").to_pandas()
    means = df.set_index('FEATURE')['MEAN_VAL']
    stds  = df.set_index('FEATURE')['STD_VAL'].replace(0, 1)
    return means, stds

ref_df      = load_reference_data()
means, stds = load_scaler_params()

feature_cols = ['AREA','BEDROOMS','BATHROOMS','STORIES','MAINROAD',
                'GUESTROOM','BASEMENT','HOTWATERHEATING','AIRCONDITIONING',
                'PARKING','PREFAREA','FURNISHINGSTATUS']

st.sidebar.header("Caractéristiques de la maison")
area      = st.sidebar.slider("Surface (m²)",    int(ref_df['AREA'].min()),      int(ref_df['AREA'].max()),      int(ref_df['AREA'].median()))
bedrooms  = st.sidebar.slider("Chambres",         int(ref_df['BEDROOMS'].min()),  int(ref_df['BEDROOMS'].max()),  int(ref_df['BEDROOMS'].median()))
bathrooms = st.sidebar.slider("Salles de bain",   int(ref_df['BATHROOMS'].min()), int(ref_df['BATHROOMS'].max()), int(ref_df['BATHROOMS'].median()))
stories   = st.sidebar.slider("Étages",           int(ref_df['STORIES'].min()),   int(ref_df['STORIES'].max()),   int(ref_df['STORIES'].median()))
parking   = st.sidebar.slider("Parking",          int(ref_df['PARKING'].min()),   int(ref_df['PARKING'].max()),   int(ref_df['PARKING'].median()))

st.sidebar.subheader("Options")
mainroad        = st.sidebar.selectbox("Route principale",     ["Oui", "Non"])
guestroom       = st.sidebar.selectbox("Chambre d'amis",       ["Oui", "Non"])
basement        = st.sidebar.selectbox("Sous-sol",             ["Oui", "Non"])
hotwaterheating = st.sidebar.selectbox("Chauffage eau chaude", ["Oui", "Non"])
airconditioning = st.sidebar.selectbox("Climatisation",        ["Oui", "Non"])
prefarea        = st.sidebar.selectbox("Zone privilégiée",     ["Oui", "Non"])
furnishing      = st.sidebar.selectbox("Ameublement",          ["Meublé", "Semi-meublé", "Non meublé"])

if st.sidebar.button("Estimer le prix", type="primary"):
    with st.spinner("Calcul en cours..."):

        oui_non        = {'Oui': 1, 'Non': 0}
        furnishing_map = {'Non meublé': 0, 'Semi-meublé': 1, 'Meublé': 2}

        input_raw = pd.Series({
            'AREA':             area,
            'BEDROOMS':         bedrooms,
            'BATHROOMS':        bathrooms,
            'STORIES':          stories,
            'MAINROAD':         oui_non[mainroad],
            'GUESTROOM':        oui_non[guestroom],
            'BASEMENT':         oui_non[basement],
            'HOTWATERHEATING':  oui_non[hotwaterheating],
            'AIRCONDITIONING':  oui_non[airconditioning],
            'PARKING':          parking,
            'PREFAREA':         oui_non[prefarea],
            'FURNISHINGSTATUS': furnishing_map[furnishing]
        })

        input_scaled = (input_raw - means) / stds

        values_str = ', '.join([str(round(float(v), 6)) for v in input_scaled[feature_cols].values])

        sql = f"""
            WITH input_data AS (
                SELECT {values_str}
            )
            SELECT MODEL(HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_PREDICTOR, LAST)!PREDICT(
                {values_str}
            ) AS PREDICTED_PRICE
        """
        try:
            result          = session.sql(sql).collect()
            raw             = result[0]['PREDICTED_PRICE']
            predicted_price = float(json.loads(raw)['output_feature_0'])

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prix estimé",  f"{predicted_price:,.0f} USD")
            col2.metric("Surface",       f"{area} m²")
            col3.metric("Chambres",     bedrooms)

            st.subheader("Positionnement du bien")
            comp = pd.DataFrame({
                'Indicateur': ['Prix estimé', 'Prix moyen', 'Prix médian'],
                'Valeur':     [predicted_price, ref_df['PRICE'].mean(), ref_df['PRICE'].median()]
            })
            st.bar_chart(comp.set_index('Indicateur'))

            st.subheader("Récapitulatif")
            recap = pd.DataFrame({
                'Caractéristique': [
                    'Surface (m²)', 'Chambres', 'Salles de bain', 'Étages', 'Parking',
                    'Route principale', "Chambre d'amis", 'Sous-sol',
                    'Chauffage eau chaude', 'Climatisation', 'Zone privilégiée', 'Ameublement'
                ],
                'Valeur': [
                    f"{area} m²", bedrooms, bathrooms, stories, parking,
                    mainroad, guestroom, basement,
                    hotwaterheating, airconditioning, prefarea, furnishing
                ]
            })
            st.table(recap)

            st.markdown("---")
            st.caption(
                "Modèle : XGBoost optimisé (Grid Search) | "
                "R² = 0.6636 | MAE ≈ 47 644 USD | "
                "Dataset : 545 maisons (après déduplication)"
            )

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

else:
    st.info("Renseignez les caractéristiques et cliquez sur Estimer le prix")
