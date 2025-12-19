"""
Sistema Inteligente de Predicción - Ley de Urgencia (Decreto 34)
Aplicación Web con ML para evaluación clínica integral
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from resolucion_ml_v2 import (
    prepare_dataframe,
    build_feature_pipeline,
    build_xgboost_model,
    add_text_alerts,
    add_decree_flags,
    parse_decimal_series,
    normalize_fio2_fraction,
    to_bool,
)
from sklearn.pipeline import Pipeline


@st.cache_resource
def load_model(data_path: Path) -> Dict:
    """Carga y entrena el modelo XGBoost con todos los datos históricos"""
    df, alert_cols, decree_cols, df_raw = prepare_dataframe(data_path)

    label_col = "Resolucion"
    labeled_mask = df[label_col].notna()

    numeric_base = [
        "PA_Sistolica", "PA_Diastolica", "PA_Media", "Temperatura",
        "SatO2", "FC", "FR", "Glasgow", "PCR", "Hemoglobina",
        "Creatinina", "BUN", "Sodio", "Potasio", "FiO2",
    ]
    binary_base = [
        "FiO2_ge50_flag", "Ventilacion_Mecanica", "Cirugia", "Cirugia_mismo_dia",
        "Hemodinamia", "Hemodinamia_mismo_dia", "Endoscopia", "Endoscopia_mismo_dia",
        "Dialisis", "Trombolisis", "Trombolisis_mismo_dia", "DVA", "Transfusiones",
        "Troponinas_Alteradas", "ECG_Alterado", "RNM_Stroke", "Compromiso_Conciencia",
        "Antecedente_Cardiaco", "Antecedente_Diabetico", "Antecedente_HTA",
    ]

    present_numeric = [c for c in numeric_base if c in df.columns]
    present_binary = [c for c in binary_base if c in df.columns]
    present_binary += alert_cols + decree_cols
    
    categorical_cols = ["Tipo_Cama"]
    if "Categoria_Inferida" in df.columns:
        categorical_cols.append("Categoria_Inferida")

    feature_cols = present_numeric + present_binary + categorical_cols

    for col in present_binary:
        if col in df.columns:
            df[col] = df[col].astype(int)

    X_all = df[feature_cols].copy()
    y_all = df[label_col]

    X_train = X_all[labeled_mask]
    y_train = y_all[labeled_mask]

    classes_sorted = sorted(y_train.unique())
    label_to_int = {cls: idx for idx, cls in enumerate(classes_sorted)}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y_train_enc = y_train.map(label_to_int)

    class_counts = y_train.value_counts()
    class_weights = (len(y_train) / (len(class_counts) * class_counts)).to_dict()

    preprocessor = build_feature_pipeline(feature_cols, present_binary, categorical_cols)
    clf = build_xgboost_model(num_classes=len(classes_sorted))
    
    model = {
        "pipeline": None, 
        "feature_cols": feature_cols, 
        "present_binary": present_binary,
        "categorical_cols": categorical_cols, 
        "alert_cols": alert_cols, 
        "decree_cols": decree_cols,
        "int_to_label": int_to_label, 
        "label_to_int": label_to_int,
        "num_classes": len(classes_sorted)
    }

    sample_weights = y_train_enc.map(lambda lbl: class_weights[int_to_label[lbl]]).values
    pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    pipeline.fit(X_train, y_train_enc, **{"clf__sample_weight": sample_weights})
    model["pipeline"] = pipeline
    
    return model


def build_input_df(user_inputs: dict, model_meta: dict) -> pd.DataFrame:
    # Create single-row DataFrame with expected columns
    row = {k: v for k, v in user_inputs.items()}
    df_in = pd.DataFrame([row])

    # Normaliza FiO2 a fracción (NO usar parse_decimal_series aquí porque viene limpio de Streamlit)
    if "FiO2" not in df_in.columns:
        df_in["FiO2"] = np.nan
    if "FiO2_raw" in df_in.columns:
        # FiO2_raw ya es numérico desde Streamlit, solo normalizamos a fracción
        df_in["FiO2"] = normalize_fio2_fraction(df_in["FiO2_raw"])

    # Booleans a bool
    bool_cols = [
        "FiO2_ge50_flag",
        "Ventilacion_Mecanica",
        "Cirugia",
        "Cirugia_mismo_dia",
        "Hemodinamia",
        "Hemodinamia_mismo_dia",
        "Endoscopia",
        "Endoscopia_mismo_dia",
        "Dialisis",
        "Trombolisis",
        "Trombolisis_mismo_dia",
        "DVA",
        "Transfusiones",
        "Troponinas_Alteradas",
        "ECG_Alterado",
        "RNM_Stroke",
        "Compromiso_Conciencia",
        "Antecedente_Cardiaco",
        "Antecedente_Diabetico",
        "Antecedente_HTA",
    ]
    for c in bool_cols:
        if c in df_in.columns:
            df_in[c] = df_in[c].apply(to_bool)

    # Texto libre alertas y flags decreto
    df_in, alert_cols = add_text_alerts(df_in)
    df_in, decree_cols = add_decree_flags(df_in)

    # Tipo cama a string
    if "Tipo_Cama" in df_in.columns:
        df_in["Tipo_Cama"] = df_in["Tipo_Cama"].fillna("desconocido").astype(str)
    else:
        df_in["Tipo_Cama"] = "desconocido"

    # Booleans a int para pipeline
    for col in model_meta["present_binary"]:
        if col in df_in.columns:
            df_in[col] = df_in[col].astype(int)
        else:
            df_in[col] = 0

    # Asegura todas las columnas presentes
    for col in model_meta["feature_cols"]:
        if col not in df_in.columns:
            df_in[col] = np.nan

    return df_in[model_meta["feature_cols"]]


def main():
    # Configuración de página con diseño moderno
    st.set_page_config(
        page_title="Sistema IA - Ley de Urgencia",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS Personalizado - Diseño Moderno y Compacto
    st.markdown("""
        <style>
        .main {
            padding: 0.5rem 1.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1.5rem;
            font-weight: 600;
        }
        /* Reducir espaciado de inputs */
        .stNumberInput, .stCheckbox, .stSelectbox, .stSlider {
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.5rem;
        }
        h1 {
            color: #1e3a8a;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        h2 {
            color: #1e40af;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        h3 {
            color: #3b82f6;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        /* Ocultar íconos de enlace en headers */
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a,
        .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
            display: none !important;
        }
        header[data-testid="stHeader"] a {
            display: none !important;
        }
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .alert-success {
            background-color: #d1fae5;
            border-left: 5px solid #10b981;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .alert-danger {
            background-color: #fee2e2;
            border-left: 5px solid #ef4444;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stNumberInput > div > div > input {
            font-size: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header con diseño moderno y centrado
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h1 style='color: #1e3a8a; font-size: 2.5rem; margin-bottom: 0.5rem;'>
                🏥 Sistema Inteligente - Ley de Urgencia
            </h1>
            <p style='color: #64748b; font-size: 1.1rem; margin: 0;'>
                Evaluación de Criterios del Decreto 34
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Cargar modelo automáticamente al inicio (sin sidebar)
    if "model_meta" not in st.session_state:
        with st.spinner("🔄 Cargando modelo de inteligencia artificial..."):
            try:
                st.session_state["model_meta"] = load_model(Path("Data.xlsx"))
            except Exception as e:
                st.error(f"❌ Error al cargar el modelo: {str(e)}")
                st.error("Verifica que el archivo 'Data.xlsx' exista en el directorio de la aplicación.")
                st.stop()
    
    model_meta = st.session_state["model_meta"]

    # ============ PANEL PRINCIPAL - Formulario de Datos ============
    st.markdown("### 📋 Ingreso de Datos del Paciente")
    
    # Usar tabs para evitar problema de scroll
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Signos Vitales", 
        "🧪 Laboratorio", 
        "💉 Procedimientos",
        "📝 Adicional"
    ])
    
    # TAB 1: SIGNOS VITALES
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Presión Arterial**")
            pa_sys = st.number_input("PA Sistólica (mmHg)", value=110.0, step=1.0)
            pa_dias = st.number_input("PA Diastólica (mmHg)", value=70.0, step=1.0)
            pa_med = st.number_input("PA Media (mmHg)", value=83.0, step=1.0)
            temp = st.number_input("Temperatura (°C)", value=36.5, step=0.1)
            
        with col2:
            st.markdown("**Frecuencias y Oxigenación**")
            fc = st.number_input("Frecuencia Cardíaca (lpm)", value=70.0, step=1.0)
            fr = st.number_input("Frecuencia Respiratoria (rpm)", value=14.0, step=1.0)
            sat = st.number_input("Saturación O₂ (%)", value=99.0, step=1.0)
            fio2 = st.number_input("FiO₂ (%)", value=21.0, min_value=21.0, step=1.0)
            
        with col3:
            st.markdown("**Evaluación Neurológica y Triage**")
            gcs = st.number_input("Escala de Glasgow", min_value=3, max_value=15, value=15, step=1)
            triage = st.selectbox("Triage", options=[1,2,3,4,5], index=4, 
                                format_func=lambda x: f"C{x} - {'Crítico' if x<=2 else 'Urgente' if x==3 else 'No urgente'}")
            fio2_flag = st.checkbox("FiO₂ ≥ 50%", value=False, key="fio2_flag")
    
    # TAB 2: LABORATORIO
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Hematología y Función Renal**")
            hemoglobina = st.number_input("Hemoglobina (g/dL)", value=13.0, step=0.1)
            pcr = st.number_input("PCR (mg/L)", value=3.0, step=0.5)
            creatinina = st.number_input("Creatinina (mg/dL)", value=1.0, step=0.1)
            bun = st.number_input("BUN (mg/dL)", value=15.0, step=1.0)
            
        with col2:
            st.markdown("**Electrolitos**")
            sodio = st.number_input("Sodio (mEq/L)", value=140.0, step=1.0)
            potasio = st.number_input("Potasio (mEq/L)", value=4.0, step=0.1)
            dreo = st.number_input("DREO", value=0.0, step=0.1)
            
        with col3:
            st.markdown("**Estudios Complementarios**")
            troponinas = st.checkbox("Troponinas Alteradas", value=False, key="troponinas")
            ecg_alt = st.checkbox("ECG Alterado", value=False, key="ecg_alterado")
            rnm = st.checkbox("RNM Protocolo Stroke", value=False, key="rnm_stroke")
            dialisis = st.checkbox("Diálisis", value=False, key="dialisis")
    
    # TAB 3: PROCEDIMIENTOS
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Soporte Vital**")
            vm = st.checkbox("Ventilación Mecánica", value=False, key="vm")
            dva = st.checkbox("Drogas Vasoactivas (DVA)", value=False, key="dva")
            transf = st.checkbox("Transfusiones", value=False, key="transfusiones")
            comp_conc = st.checkbox("Compromiso de Conciencia", value=False, key="comp_conciencia")
            
            st.markdown("**Procedimientos Diagnósticos**")
            hemodinamia = st.checkbox("Hemodinamia", value=False, key="hemodinamia")
            hemo_mismo = st.checkbox("└─ Mismo día de ingreso", value=False, disabled=not hemodinamia, key="hemo_mismo_dia")
            
        with col2:
            st.markdown("**Intervenciones**")
            cirugia = st.checkbox("Cirugía", value=False, key="cirugia")
            cir_mismo = st.checkbox("└─ Mismo día de ingreso", value=False, disabled=not cirugia, key="cir_mismo_dia")
            
            endoscopia = st.checkbox("Endoscopia", value=False, key="endoscopia")
            endo_mismo = st.checkbox("└─ Mismo día de ingreso", value=False, disabled=not endoscopia, key="endo_mismo_dia")
            
            trombolisis = st.checkbox("Trombólisis", value=False, key="trombolisis")
            tromb_mismo = st.checkbox("└─ Mismo día de ingreso", value=False, disabled=not trombolisis, key="tromb_mismo_dia")
    
    # TAB 4: INFORMACIÓN ADICIONAL
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Antecedentes Médicos**")
            ant_cardiaco = st.checkbox("Antecedente Cardíaco", value=False, key="ant_cardiaco")
            ant_diabetico = st.checkbox("Antecedente Diabético", value=False, key="ant_diabetico")
            ant_hta = st.checkbox("Antecedente HTA", value=False, key="ant_hta")
        
        with col2:
            st.markdown("**Tipo de Cama**")
            tipo_cama = st.selectbox("Ubicación", 
                options=["desconocido", "UCI Adulto", "UCI Pediátrico", "UCI Neonatal", 
                        "Intermedio", "Básico", "Box Urgencia"], index=0)
        
        st.markdown("**Motivo de Consulta / Diagnóstico**")
        texto_libre = st.text_area("Descripción clínica",
            placeholder="Ej: Paciente con control de rutina, sin síntomas agudos...", 
            height=100,
            value="")

    # ============ BOTÓN DE PREDICCIÓN ============
    st.markdown("---")
    
    # Contenedor para el resultado (se mostrará aquí, sin necesidad de scroll)
    resultado_container = st.container()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 CALCULAR PREDICCIÓN",
            type="primary",
            use_container_width=True
        )

    if predict_button:
        # Preparar inputs
        user_inputs = {
            "PA_Sistolica": pa_sys,
            "PA_Diastolica": pa_dias,
            "PA_Media": pa_med,
            "Temperatura": temp,
            "SatO2": sat,
            "FR": fr,
            "FC": fc,
            "Glasgow": gcs,
            "FiO2_raw": fio2,
            "FiO2_ge50_flag": fio2_flag,
            "Ventilacion_Mecanica": vm,
            "Compromiso_Conciencia": comp_conc,
            "DVA": dva,
            "Hemodinamia": hemodinamia,
            "Hemodinamia_mismo_dia": hemo_mismo,
            "Cirugia": cirugia,
            "Cirugia_mismo_dia": cir_mismo,
            "Endoscopia": endoscopia,
            "Endoscopia_mismo_dia": endo_mismo,
            "Dialisis": dialisis,
            "Trombolisis": trombolisis,
            "Trombolisis_mismo_dia": tromb_mismo,
            "Transfusiones": transf,
            "RNM_Stroke": rnm,
            "Troponinas_Alteradas": troponinas,
            "ECG_Alterado": ecg_alt,
            "Antecedente_Cardiaco": ant_cardiaco,
            "Antecedente_Diabetico": ant_diabetico,
            "Antecedente_HTA": ant_hta,
            "Tipo_Cama": tipo_cama,
            "Texto_Libre": texto_libre,
            "Triage": triage,
            "PCR": pcr,
            "Hemoglobina": hemoglobina,
            "Creatinina": creatinina,
            "BUN": bun,
            "Sodio": sodio,
            "Potasio": potasio,
            "DREO": dreo,
        }

        # Realizar predicción
        with st.spinner("🤖 Procesando datos con IA..."):
            input_df = build_input_df(user_inputs, model_meta)
            pipeline = model_meta["pipeline"]
            proba = pipeline.predict_proba(input_df)[0]
            class_labels = [model_meta["int_to_label"][c] for c in pipeline.named_steps["clf"].classes_]
            
            # LÓGICA MEJORADA: Override basado en Decreto 34
            # Si NINGÚN criterio clínico grave está presente, forzar NO PERTINENTE
            tiene_flag_grave = (
                input_df["flag_resp_grave"].values[0] or
                input_df["flag_circ_grave"].values[0] or
                input_df["flag_neuro_grave"].values[0]
            )
            
            pertinente_idx = class_labels.index("PERTINENTE")
            
            if tiene_flag_grave:
                # Si cumple criterios del Decreto 34, usar predicción directa
                pred_label = "PERTINENTE" if proba[pertinente_idx] >= 0.5 else "NO PERTINENTE"
                max_proba = proba[pertinente_idx] * 100 if pred_label == "PERTINENTE" else proba[1 - pertinente_idx] * 100
                override_aplicado = False
            else:
                # Si NO cumple criterios del Decreto 34, requerir alta confianza (95%)
                if proba[pertinente_idx] >= 0.95:
                    pred_label = "PERTINENTE"
                    max_proba = proba[pertinente_idx] * 100
                    override_aplicado = False
                else:
                    pred_label = "NO PERTINENTE"
                    # Cuando aplicamos override, mostrar que es por criterios clínicos
                    override_aplicado = True
                    max_proba = 100.0  # Confianza alta porque es basado en reglas del Decreto 34

        # ============ VISUALIZACIÓN DE RESULTADOS ============
        st.markdown("---")
        st.header("🎯 Resultado de la Predicción")
        
        # Resultado principal con diseño destacado
        if pred_label == "PERTINENTE":
            st.markdown(f"""
                <div class='alert-danger'>
                    <h2 style='color: #dc2626; margin: 0;'>⚠️ SÍ ES LEY DE URGENCIA</h2>
                    <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                        <strong>El paciente CUMPLE con los criterios del Decreto 34</strong>
                    </p>
                    <p style='margin: 0;'>Confianza del modelo: {max_proba:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            mensaje_confianza = "Basado en criterios clínicos del Decreto 34" if override_aplicado else f"Confianza del modelo: {max_proba:.1f}%"
            st.markdown(f"""
                <div class='alert-success'>
                    <h2 style='color: #16a34a; margin: 0;'>✅ NO ES LEY DE URGENCIA</h2>
                    <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                        <strong>El paciente NO cumple con los criterios del Decreto 34</strong>
                    </p>
                    <p style='margin: 0;'>{mensaje_confianza}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Información adicional sobre los datos procesados
        with st.expander("🔍 Ver detalles técnicos de la predicción"):
            st.markdown("**Features procesados por el modelo:**")
            st.caption(f"Total de características: {len(model_meta['feature_cols'])}")
            st.json({
                "Clasificación predicha": pred_label,
                "Probabilidad máxima": f"{max_proba:.2f}%",
                "Clases disponibles": class_labels,
                "Features numéricos": len([c for c in model_meta['feature_cols'] if c not in model_meta['present_binary'] + model_meta['categorical_cols']]),
                "Features binarios": len(model_meta['present_binary']),
                "Features categóricos": len(model_meta['categorical_cols'])
            })
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; padding: 2rem 0;'>
            <p style='font-size: 0.9rem; margin: 0.5rem 0;'>
                🏥 <strong>Sistema Inteligente de Predicción - Ley de Urgencia (Decreto 34)</strong>
            </p>
            <p style='font-size: 0.85rem; margin: 0.5rem 0;'>
                Modelo: XGBoost | Versión: 2.0 | Machine Learning para apoyo a decisiones clínicas
            </p>
            <p style='font-size: 0.8rem; color: #94a3b8; margin: 0.5rem 0;'>
                ⚠️ Esta herramienta es de apoyo. La decisión clínica final debe ser tomada por personal médico calificado.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
