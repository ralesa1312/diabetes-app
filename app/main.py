import streamlit as st
import pandas as pd
import yaml
import os
import sys
from openai import OpenAI

# --- 1. CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ml.utils import MODEL_DIR, load_model

# --- 2. CONFIGURATION FOURNISSEURS LLM ---
PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "base_url": None
    },
    "Groq (Llama/Mixtral)": {
        "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        "base_url": "https://api.groq.com/openai/v1"
    }
}

# --- 3. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Diabetes AI Expert", page_icon="🩺", layout="wide")

# Initialisation des états de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "current_res" not in st.session_state:
    st.session_state.current_res = "Non analysé"

# --- 4. CHARGEMENT DU MODÈLE ML ---
@st.cache_resource
def get_ml_model():
    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return load_model(os.path.join(MODEL_DIR, f"{config['model']['name']}.pkl"))

try:
    ml_pipeline = get_ml_model()
except Exception as e:
    st.error(f"Erreur de chargement du modèle ML : {e}")
    st.stop()

# --- 5. BARRE LATÉRALE (CONFIGURATION & SAISIE) ---
st.sidebar.title("⚙️ Configuration & Profil")

# Choix du LLM
with st.sidebar.expander("🔑 Intelligence Artificielle", expanded=True):
    provider_name = st.selectbox("Fournisseur", list(PROVIDERS.keys()))
    llm_model = st.selectbox("Modèle", PROVIDERS[provider_name]["models"])
    user_api_key = st.text_input(f"Clé API {provider_name}", type="password")

st.sidebar.divider()

# Saisie des 21 colonnes organisée
with st.sidebar:
    st.subheader("📋 Données du Patient")
    
    with st.expander("🩺 Indicateurs Médicaux", expanded=True):
        high_bp = st.selectbox("Hypertension (HighBP)", [0, 1])
        high_chol = st.selectbox("Cholestérol élevé (HighChol)", [0, 1])
        chol_check = st.selectbox("Vérif. Cholestérol 5 ans", [0, 1])
        bmi = st.number_input("IMC (BMI)", 10.0, 60.0, 25.0)
        stroke = st.selectbox("Antécédent AVC", [0, 1])
        heart_disease = st.selectbox("Maladie Cardiaque", [0, 1])

    with st.expander("🏃 Mode de Vie"):
        phys_activity = st.selectbox("Activité Physique (30j)", [0, 1])
        smoker = st.selectbox("Fumeur (+100 cig.)", [0, 1])
        hvy_alcohol = st.selectbox("Grosse conso Alcool", [0, 1])
        fruits = st.selectbox("Consomme Fruits (1+/j)", [0, 1])
        veggies = st.selectbox("Consomme Légumes (1+/j)", [0, 1])

    with st.expander("🌡️ État de Santé & Social"):
        gen_hlth = st.slider("Santé Générale (1:Ex, 5:Mauvais)", 1, 5, 2)
        ment_hlth = st.slider("Santé Mentale (Jours mal/mois)", 0, 30, 0)
        phys_hlth = st.slider("Santé Physique (Jours mal/mois)", 0, 30, 0)
        diff_walk = st.selectbox("Difficulté à marcher", [0, 1])
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
        age = st.slider("Tranche d'âge (1:18-24, 13:80+)", 1, 13, 5)
        education = st.slider("Niveau d'études (1-6)", 1, 6, 4)
        income = st.slider("Revenus (1-8)", 1, 8, 5)
        any_healthcare = st.selectbox("Couverture Santé", [0, 1])
        no_doc_cost = st.selectbox("Renoncement soins (Coût)", [0, 1])

# --- 6. PRÉPARATION DES DONNÉES ---
input_dict = {
    'HighBP': float(high_bp), 'HighChol': float(high_chol), 'CholCheck': float(chol_check),
    'BMI': float(bmi), 'Smoker': float(smoker), 'Stroke': float(stroke),
    'HeartDiseaseorAttack': float(heart_disease), 'PhysActivity': float(phys_activity),
    'Fruits': float(fruits), 'Veggies': float(veggies), 'HvyAlcoholConsump': float(hvy_alcohol),
    'AnyHealthcare': float(any_healthcare), 'NoDocbcCost': float(no_doc_cost),
    'GenHlth': float(gen_hlth), 'MentHlth': float(ment_hlth), 'PhysHlth': float(phys_hlth),
    'DiffWalk': float(diff_walk), 'Sex': float(sex), 'Age': float(age),
    'Education': float(education), 'Income': float(income)
}
input_df = pd.DataFrame([input_dict])

# --- 7. INTERFACE PRINCIPALE ---
st.title("🩺 Assistant Médical Augmenté par l'IA")
st.markdown("---")

col_ml, col_llm = st.columns([1, 1.2])

# --- A. SECTION MACHINE LEARNING ---
with col_ml:
    st.subheader("🔍 Analyse Prédictive (ML)")
    if st.button("Lancer l'Analyse du Risque", type="primary", use_container_width=True):
        prediction = ml_pipeline.predict(input_df)[0]
        proba = ml_pipeline.predict_proba(input_df)[0]
        
        labels = {0.0: "Faible Risque", 1.0: "Pré-diabète", 2.0: "Diabète"}
        st.session_state.current_res = labels[prediction]
        st.session_state.prediction_done = True
        
        if prediction == 0.0:
            st.success(f"### {st.session_state.current_res}")
        elif prediction == 1.0:
            st.warning(f"### {st.session_state.current_res}")
        else:
            st.error(f"### {st.session_state.current_res}")
            
        st.metric("Confiance du modèle", f"{proba[int(prediction)]:.1%}")
        st.progress(proba[int(prediction)])

# --- B. SECTION CHATBOT EXPERT ---
with col_llm:
    st.subheader("💬 Conseils & Interprétation IA")
    
    if not user_api_key:
        st.info("💡 Saisissez votre clé API dans la barre latérale pour activer l'assistant.")
    else:
        chat_box = st.container(height=500)
        
        # Affichage de l'historique
        for m in st.session_state.messages:
            chat_box.chat_message(m["role"]).write(m["content"])

        if prompt := st.chat_input("Expliquez-moi mes résultats..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            chat_box.chat_message("user").write(prompt)

            # Construction du System Prompt avec les 21 colonnes
            details_cliniques = "\n".join([f"- {k}: {v}" for k, v in input_dict.items()])

            system_instruction = f"""
            Tu es un assistant expert en endocrinologie et en médecine préventive. 
            Ton rôle est d'interpréter le risque de diabète calculé par un modèle de Machine Learning.

            CONTEXTE DU PATIENT :
            Résultat du modèle : {st.session_state.current_res}
            Données complètes (21 variables) :
            {details_cliniques}

            DIRECTIVES DE RÉPONSE :
            1. ANALYSE CROISÉE : Ne te contente pas de lister les données. Fais des liens (ex: Lien entre BMI élevé, HighBP et manque d'activité physique).
            2. PERSONNALISATION : Si le patient a des difficultés à marcher (DiffWalk), suggère des activités à faible impact (natation, chaise).
            3. PSYCHOLOGIE : Si MentHlth est élevé, souligne l'importance du bien-être mental dans la gestion métabolique.
            4. SÉCURITÉ : Tu DOIS toujours finir par : "Cette analyse automatique ne remplace pas un diagnostic médical. Veuillez consulter un professionnel de santé."
            5. INTERDICTION : Ne prescris JAMAIS de médicaments ni de dosages (ex: Insuline, Metformine).

            Réponds de manière structurée avec des puces, en français, sur un ton bienveillant mais professionnel.
            """

            try:
                client = OpenAI(
                    api_key=user_api_key,
                    base_url=PROVIDERS[provider_name]["base_url"]
                )
                
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    ]
                )
                
                full_res = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                chat_box.chat_message("assistant").write(full_res)
                
            except Exception as e:
                st.error(f"Erreur LLM : {str(e)}")