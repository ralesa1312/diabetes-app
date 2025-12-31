import os
import sys

import numpy as np
import streamlit as st

from ml.utils import DEFAULT_MODEL_PATH, load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)  # ajoute le projet au PYTHONPATH


model = load_model(DEFAULT_MODEL_PATH)


st.title("🩺 Prédiction du diabète")

with st.form("diabete_form"):
    HighBP = st.number_input("Pression artérielle élevée", 0, 200)
    HighChol = st.number_input("Taux de cholestérol", 0, 300)
    CholCheck = st.number_input("Pression sanguine", 0, 200)
    BMI = st.number_input("Épaisseur de la peau", 0, 100)
    Smoker = st.number_input("Insuline", 0, 900)
    Stroke = st.number_input("IMC (BMI)", 0.0, 70.0)
    HeartDiseaseorAttack = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    PhysActivity = st.number_input("Activité physique", 0, 120)
    Fruits = st.number_input("Fruits", 0, 120)
    Veggies = st.number_input("Légumes", 0, 120)
    HvyAlcoholConsump = st.number_input("Consommation d'alcool", 0, 120)
    AnyHealthcare = st.number_input("Soins de santé", 0, 120)
    NoDocbcCost = st.number_input("Coût du médecin", 0, 120)
    GenHlth = st.number_input("Santé générale", 0, 120)
    MentHlth = st.number_input("Santé mentale", 0, 120)
    PhysHlth = st.number_input("Santé physique", 0, 120)
    DiffWalk = st.number_input("Difficulté à marcher", 0, 120)
    Sex = st.number_input("Sexe", 0, 120)
    Age = st.number_input("Âge", 0, 120)
    Education = st.number_input("Éducation", 0, 120)
    Income = st.number_input("Revenu", 0, 120)

    submit = st.form_submit_button("Prédire")

    if submit:
        input_data = np.array(
            [
                [
                    HighBP,
                    HighChol,
                    CholCheck,
                    BMI,
                    Smoker,
                    Stroke,
                    HeartDiseaseorAttack,
                    PhysActivity,
                    Fruits,
                    Veggies,
                    HvyAlcoholConsump,
                    AnyHealthcare,
                    NoDocbcCost,
                    GenHlth,
                    MentHlth,
                    PhysHlth,
                    DiffWalk,
                    Sex,
                    Age,
                    Education,
                    Income,
                ]
            ]
        )

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Le modèle prédit que vous êtes à risque de diabète.")
        else:
            st.success("✅ Le modèle prédit que vous n'êtes pas à risque de diabète.")
