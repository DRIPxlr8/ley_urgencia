import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ============================================================
# Helpers
# ============================================================


def strip_accents(text: str) -> str:
    if text is None:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn"
    )


def normalize_header(h: str) -> str:
    # Trim and collapse inner whitespace
    return re.sub(r"\s+", " ", str(h)).strip()


def normalize_key(h: str) -> str:
    return normalize_header(strip_accents(h)).lower()


def to_bool(x) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    if isinstance(x, (int, float)) and not pd.isna(x):
        return x != 0
    t = str(x).strip().lower()
    if t in {"", "null", "none", "nan"}:
        return False
    return t in {"si", "sí", "s", "y", "x", "true", "1", "t", "yes", "verdadero"}


def parse_decimal_series(series: pd.Series) -> pd.Series:
    def _conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s.lower() in {"", "null", "none", "nan"}:
            return np.nan
        s = s.replace(".", "")  # thousands
        s = s.replace(",", ".")  # decimal
        s = s.replace("%", "")
        return pd.to_numeric(s, errors="coerce")

    return series.apply(_conv)


def normalize_fio2_fraction(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mask = s > 1.0
    s.loc[mask] = s.loc[mask] / 100.0
    return s


TEXT_ALERTS: List[Tuple[str, str]] = [
    (r"paro\s*card|arresto", "paro_cardiorrespiratorio"),
    (r"shock", "shock"),
    (r"sepsis|septico", "sepsis"),
    (r"acv|ictus|stroke|ataque cerebro|accidente cerebro|hsa|hcia", "acv"),
    (r"convulsion|status epilept", "convulsiones"),
    (r"coma|inconsciente|no responde", "coma"),
    (r"infarto|iam|stemi|scacest|scai|coronario agudo", "sindrome_coronario"),
    (r"dolor torac", "dolor_toracico"),
    (r"disnea|insuficiencia respiratoria|sdra|broncoaspir|crisis asma", "disnea"),
    (r"embol.? pulmon|tromboembolismo pulmon|\btep\b", "tep"),
    (r"hemorragia|sangrado|hematemesis|melena|hemoptisis", "hemorragia"),
    (r"trauma|politrauma|tce|accidente", "trauma"),
    (r"neumon", "neumonia"),
]

# Normalized header -> canonical
HEADER_MAP: Dict[str, str] = {
    "id": "ID",
    "fecha formulario": "Fecha_Formulario",
    "episodio": "Episodio",
    "antecedentes cardiacos": "Antecedente_Cardiaco",
    "antecedentes diabeticos": "Antecedente_Diabetico",
    "antecedentes de hipertension arterial": "Antecedente_HTA",
    "triage": "Triage",
    "presion arterial sistolica": "PA_Sistolica",
    "presion arterial diastolica": "PA_Diastolica",
    "presion arterial media": "PA_Media",
    "temperatura en c": "Temperatura",
    "temperatura en °c": "Temperatura",
    "saturacion oxigeno": "SatO2",
    "frecuencia cardiaca": "FC",
    "frecuencia respiratoria": "FR",
    "tipo de cama": "Tipo_Cama",
    "glasgow": "Glasgow",
    "fio2": "FiO2_raw",
    "fio2 > o igual a 50%": "FiO2_ge50_flag",
    "fio2 >= o igual a 50%": "FiO2_ge50_flag",
    "ventilacion mecanica": "Ventilacion_Mecanica",
    "cirugia realizada": "Cirugia",
    "cirugia mismo dia ingreso": "Cirugia_mismo_dia",
    "hemodinamia realizada": "Hemodinamia",
    "hemodinamia mismo dia ingreso": "Hemodinamia_mismo_dia",
    "endoscopia": "Endoscopia",
    "endoscopia mismo dia ingreso": "Endoscopia_mismo_dia",
    "dialisis": "Dialisis",
    "trombolisis": "Trombolisis",
    "trombolisis mismo dia ingreso": "Trombolisis_mismo_dia",
    "pcr": "PCR",
    "hemoglobina": "Hemoglobina",
    "creatinina": "Creatinina",
    "nitrogeno ureico": "BUN",
    "sodio": "Sodio",
    "potasio": "Potasio",
    "dreo": "DREO",
    "troponinas alteradas": "Troponinas_Alteradas",
    "ecg alterado": "ECG_Alterado",
    "rnm protocolo stroke": "RNM_Stroke",
    "dva": "DVA",
    "transfusiones": "Transfusiones",
    "compromiso conciencia": "Compromiso_Conciencia",
    "texto libre": "Texto_Libre",
    "resolucion": "Resolucion",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = normalize_key(col)
        if key in HEADER_MAP:
            rename_map[col] = HEADER_MAP[key]
    return df.rename(columns=rename_map)


def add_text_alerts(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    text_series = df.get("Texto_Libre", pd.Series(index=df.index, data=""))
    cleaned = (
        text_series.fillna("")
        .apply(strip_accents)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
    )
    alert_cols: List[str] = []
    for pattern, slug in TEXT_ALERTS:
        col = f"alert_{slug}"
        df[col] = cleaned.str.contains(pattern, regex=True)
        alert_cols.append(col)
    return df, alert_cols

def infer_patient_category(row) -> str:
    """
    Infiere categoría etaria basada en el tipo de cama y texto libre.
    Prioridad: NEO > PEDIATRICO > ADULTO
    """
    cama = str(row.get('Tipo_Cama', '')).lower()
    texto = str(row.get('Texto_Libre', '')).lower()
    full_context = f"{cama} {texto}"
    
    # Palabras clave basadas en nomenclatura hospitalaria estándar
    if re.search(r'\b(neo|rn|recien nacido|prematuro|incubadora|cuna)\b', full_context):
        return 'NEO'
    if re.search(r'\b(ped|pediat|infantil|niño|escolar|lactante|cuna)\b', full_context):
        return 'PEDIATRICO'
    
    return 'ADULTO'

def add_decree_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # 1. Inferir tipo de paciente fila por fila
    df['Categoria_Inferida'] = df.apply(infer_patient_category, axis=1)

    # 2. Preparar variables numéricas
    fio2 = df.get("FiO2", pd.Series(index=df.index, data=np.nan))
    sat = pd.to_numeric(df.get("SatO2", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    gcs = pd.to_numeric(df.get("Glasgow", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    pas = pd.to_numeric(df.get("PA_Sistolica", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    hb = pd.to_numeric(df.get("Hemoglobina", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    triage = pd.to_numeric(df.get("Triage", pd.Series(index=df.index, data=np.nan)), errors="coerce")
    
    # Variables Booleanas
    vm = df.get("Ventilacion_Mecanica", pd.Series(index=df.index, data=False))
    fio2_flag = df.get("FiO2_ge50_flag", pd.Series(index=df.index, data=False))
    comp_conc = df.get("Compromiso_Conciencia", pd.Series(index=df.index, data=False))
    dva = df.get("DVA", pd.Series(index=df.index, data=False))
    hemo = df.get("Hemodinamia_mismo_dia", pd.Series(index=df.index, data=False))
    ciru = df.get("Cirugia_mismo_dia", pd.Series(index=df.index, data=False))
    endo = df.get("Endoscopia_mismo_dia", pd.Series(index=df.index, data=False))
    transf = df.get("Transfusiones", pd.Series(index=df.index, data=False))
    rnm = df.get("RNM_Stroke", pd.Series(index=df.index, data=False))
    tromb = df.get("Trombolisis_mismo_dia", pd.Series(index=df.index, data=False))
    
    # 3. Criterios Respiratorios Diferenciados (Decreto 34)
    # Adulto: (SatO2 <= 90% AND FiO2 >= 50%) OR Ventilación Mecánica
    crit_resp_adult = (df['Categoria_Inferida'] == 'ADULTO') & (
        vm | (((fio2 >= 0.50) | fio2_flag) & (sat <= 90))
    )
    # Pediátrico: (SatO2 < 92% AND FiO2 >= 50%) OR Ventilación Mecánica  
    crit_resp_ped = (df['Categoria_Inferida'] == 'PEDIATRICO') & (
        vm | (((fio2 >= 0.50) | fio2_flag) & (sat < 92))
    )
    # Neonatal: (SatO2 <= 92% AND FiO2 >= 40%) OR Ventilación Mecánica
    crit_resp_neo = (df['Categoria_Inferida'] == 'NEO') & (
        vm | ((fio2 >= 0.40) & (sat <= 92))
    )
    
    df["flag_resp_grave"] = crit_resp_adult | crit_resp_ped | crit_resp_neo

    # 4. Criterios Neurológicos Diferenciados (Decreto 34)
    # Pediátrico: Glasgow <= 12
    crit_neuro_ped = (df['Categoria_Inferida'] == 'PEDIATRICO') & (gcs <= 12)
    # Adulto/Neo: Glasgow <= 8 (coma severo) o compromiso de conciencia con Glasgow <= 12
    crit_neuro_adult = (df['Categoria_Inferida'] != 'PEDIATRICO') & (
        (gcs <= 8) | (comp_conc & (gcs <= 12))
    )
    df["flag_neuro_grave"] = crit_neuro_ped | crit_neuro_adult | rnm | tromb

    # 5. Criterios Circulatorios (Decreto 34)
    # DVA: Universal para todos los grupos etarios
    # Hipotensión (PAS < 90): Solo adultos (en neonatos/pediátricos es tardía)
    crit_shock_adult = (df['Categoria_Inferida'] == 'ADULTO') & (pas < 90) & pas.notna()
    df["flag_circ_grave"] = (
        dva | crit_shock_adult | hemo | ciru | (endo & transf) | (transf & (hb < 7))
    )

    # 6. Triage Crítico (C1 y C2)
    df["flag_triage_critico"] = triage.isin([1, 2])

    # 7. Regla Maestra: Cumplimiento del Decreto 34
    # Solo los criterios clínicos graves activan la regla dura
    df["CUMPLE_CRITERIO_DECRETO"] = (
        df["flag_resp_grave"] | df["flag_circ_grave"] | df["flag_neuro_grave"]
    )

    # Retornar solo los flags individuales como features (NO incluir CUMPLE_CRITERIO_DECRETO para evitar data leakage)
    return df, ["flag_resp_grave", "flag_circ_grave", "flag_neuro_grave", "flag_triage_critico"]

def prepare_dataframe(raw_path: Path) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    df_raw = pd.read_excel(raw_path)
    
    # Filtrar filas sin Episodio ANTES de normalizar
    if "Episodio" in df_raw.columns:
        initial_rows = len(df_raw)
        df_raw = df_raw[df_raw["Episodio"].notna()].copy()
        if len(df_raw) != initial_rows:
            print(f"Filas filtradas por Episodio: {initial_rows} -> {len(df_raw)}")
    
    df = normalize_columns(df_raw)

    # Convert numeric columns
    numeric_cols = [
        "PA_Sistolica",
        "PA_Diastolica",
        "PA_Media",
        "Temperatura",
        "SatO2",
        "FC",
        "FR",
        "Glasgow",
        "PCR",
        "Hemoglobina",
        "Creatinina",
        "BUN",
        "Sodio",
        "Potasio",
        "DREO",
        "FiO2_raw",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = parse_decimal_series(df[c])

    if "FiO2_raw" in df.columns:
        df["FiO2"] = normalize_fio2_fraction(df["FiO2_raw"])
    else:
        df["FiO2"] = np.nan

    # Convert booleans
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
        if c in df.columns:
            df[c] = df[c].apply(to_bool)

    # Normalize resolution label
    if "Resolucion" in df.columns:
        df["Resolucion"] = df["Resolucion"].replace(
            {"ND": np.nan, "Nd": np.nan, "nd": np.nan, "RECHAZO": "NO PERTINENTE"}
        )

    # Text alerts and decree-inspired flags
    df, alert_cols = add_text_alerts(df)
    df, decree_cols = add_decree_flags(df)

    # Ensure Tipo_Cama exists as string
    if "Tipo_Cama" not in df.columns:
        df["Tipo_Cama"] = "desconocido"
    else:
        df["Tipo_Cama"] = df["Tipo_Cama"].fillna("desconocido").astype(str)

    return df, alert_cols, decree_cols, df_raw


def build_feature_pipeline(
    feature_cols: List[str],
    binary_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    transformers = []

    numeric_cols = [c for c in feature_cols if c not in binary_cols + categorical_cols]
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if binary_cols:
        transformers.append(
            (
                "bin",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                binary_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def build_xgboost_model(num_classes: int) -> XGBClassifier:
    """Construye modelo XGBoost optimizado para clasificación de Ley de Urgencia."""
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    num_class = num_classes if objective.startswith("multi") else None
    return XGBClassifier(
        n_estimators=600,          # Aumentado de 400 a 600
        learning_rate=0.05,        # Reducido de 0.08 a 0.05 para mejor convergencia
        max_depth=6,               # Aumentado de 4 a 6 para capturar más patrones
        subsample=0.85,            # Ajustado para mejor generalización
        colsample_bytree=0.85,     # Ajustado para mejor generalización
        min_child_weight=3,        # Agregado para evitar overfitting
        gamma=0.1,                 # Agregado para regularización
        objective=objective,
        num_class=num_class,
        eval_metric="mlogloss",
        n_jobs=4,
        tree_method="hist",
        random_state=42,
    )


def main():
    input_path = Path("Data.xlsx")
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    try:
        print(f"Leyendo y normalizando datos desde {input_path}...")
        df, alert_cols, decree_cols, df_raw_full = prepare_dataframe(input_path)
    except PermissionError:
        fallback = input_path.with_name("tmp_resolucion_input.xlsx")
        shutil.copy(input_path, fallback)
        print(f"Archivo original bloqueado. Usando copia temporal {fallback}...")
        df, alert_cols, decree_cols, df_raw_full = prepare_dataframe(fallback)

    label_col = "Resolucion"
    labeled_mask = df[label_col].notna()

    # Base feature sets
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

    # Conversión a enteros para que los imputadores funcionen bien
    for col in present_binary:
        if col in df.columns:
            df[col] = df[col].astype(int)

    X_all = df[feature_cols].copy()
    y_all = df[label_col]

    print(f"Casos etiquetados (no ND): {labeled_mask.sum()} de {len(df)}")
    
    # Entrenamiento del modelo XGBoost
    X_train_full = X_all[labeled_mask]
    y_train_full = y_all[labeled_mask]

    classes_sorted = sorted(y_train_full.unique())
    label_to_int = {cls: idx for idx, cls in enumerate(classes_sorted)}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y_train_full_enc = y_train_full.map(label_to_int)
    num_classes = len(classes_sorted)

    class_counts = y_train_full.value_counts()
    class_weights = (len(y_train_full) / (len(class_counts) * class_counts)).to_dict()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_full, y_train_full_enc, test_size=0.2, stratify=y_train_full_enc, random_state=42,
    )

    preprocessor = build_feature_pipeline(feature_cols, present_binary, categorical_cols)
    clf = build_xgboost_model(num_classes)
    model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    print("Entrenando y validando...")
    w_tr = y_tr.map(lambda lbl: class_weights[int_to_label[lbl]]).values
    model.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    
    # Reporte de validación
    y_pred = model.predict(X_va)
    y_pred_labels = pd.Series(y_pred).map(int_to_label)
    y_va_labels = pd.Series(y_va).map(int_to_label)
    print("\nReporte validación (split 80/20):")
    print(classification_report(y_va_labels, y_pred_labels))

    # Reentrenamiento con todos los casos etiquetados
    print("Reentrenando con todos los casos etiquetados...")
    w_full = y_train_full_enc.map(lambda lbl: class_weights[int_to_label[lbl]]).values
    model.fit(X_train_full, y_train_full_enc, clf__sample_weight=w_full)

    # Predicción final para todas las filas
    print("Generando predicciones para todas las filas...")
    preds_all = model.predict(X_all)
    preds_all_labels = pd.Series(preds_all).map(int_to_label)
    
    # Obtener probabilidades para decisión inteligente
    proba_all = model.predict_proba(X_all)
    confianza_ml = proba_all.max(axis=1)

    # Generación del archivo de salida (Sistema Híbrido: Reglas + ML)
    df_out = df_raw_full.copy()
    
    # Solo creamos las columnas que realmente necesitamos en el CSV final
    if "CUMPLE_CRITERIO_DECRETO" in df.columns:
        # Variables temporales para la lógica híbrida
        cumple_decreto = df["CUMPLE_CRITERIO_DECRETO"].astype(bool)
        ia_prediccion = preds_all_labels.values
        confianza = confianza_ml
        
        # LÓGICA HÍBRIDA INTELIGENTE:
        # 1. Si el modelo ML tiene alta confianza (>85%), usar su predicción siempre
        # 2. Si cumple Decreto Y modelo dice PERTINENTE → PERTINENTE (refuerzo)
        # 3. Si cumple Decreto pero ML dice NO PERTINENTE con alta confianza → usar ML
        # 4. En otros casos, usar ML
        
        ETIQUETA_POSITIVA = "PERTINENTE"
        UMBRAL_CONFIANZA_ALTA = 0.85
        
        # Crear array de decisiones
        decisiones = []
        for i in range(len(df)):
            cumple_d = cumple_decreto.iloc[i]
            pred_ml = ia_prediccion[i]
            conf = confianza[i]
            
            # Alta confianza del ML → siempre usar ML
            if conf >= UMBRAL_CONFIANZA_ALTA:
                decisiones.append(pred_ml)
            # Cumple decreto Y ML dice PERTINENTE → Refuerzo positivo
            elif cumple_d and pred_ml == ETIQUETA_POSITIVA:
                decisiones.append(ETIQUETA_POSITIVA)
            # En cualquier otro caso, confiar en ML
            else:
                decisiones.append(pred_ml)
        
        df_out["Prediccion"] = decisiones
        
        # === ANÁLISIS INTERNO DE RENDIMIENTO ===
        # Usar el DataFrame procesado que tiene las resoluciones normalizadas
        resolucion_procesada = df[label_col]  # Este ya tiene ND convertido a NaN
        
        # Filtrar solo casos con resolución válida
        mask_con_resolucion = resolucion_procesada.notna()
        indices_validos = resolucion_procesada[mask_con_resolucion].index
        
        resoluciones_reales_norm = resolucion_procesada[indices_validos].astype(str).str.strip().str.upper()
        predicciones_norm = df_out.loc[indices_validos, "Prediccion"].astype(str).str.strip().str.upper()
        
        # Calcular métricas
        total_con_resolucion = len(indices_validos)
        if total_con_resolucion > 0:
            correctos = (predicciones_norm == resoluciones_reales_norm).sum()
            incorrectos = total_con_resolucion - correctos
            accuracy = (correctos / total_con_resolucion) * 100
            
            # Desglose por tipo de decisión
            casos_por_decreto = cumple_decreto[indices_validos].sum()
            casos_por_ml = total_con_resolucion - casos_por_decreto
            
            # Accuracy por tipo de decisión
            if casos_por_decreto > 0:
                correctos_decreto = ((predicciones_norm.values == resoluciones_reales_norm.values) & cumple_decreto[indices_validos].values).sum()
                acc_decreto = (correctos_decreto / casos_por_decreto) * 100
            else:
                correctos_decreto = 0
                acc_decreto = 0
                
            if casos_por_ml > 0:
                correctos_ml = ((predicciones_norm.values == resoluciones_reales_norm.values) & ~cumple_decreto[indices_validos].values).sum()
                acc_ml = (correctos_ml / casos_por_ml) * 100
            else:
                correctos_ml = 0
                acc_ml = 0
            
            print(f"\n{'='*60}")
            print(f"ANÁLISIS DE RENDIMIENTO DEL SISTEMA HÍBRIDO")
            print(f"{'='*60}")
            print(f"\nTotal casos con resolución: {total_con_resolucion}")
            print(f"Predicciones correctas: {correctos} ({accuracy:.2f}%)")
            print(f"Predicciones incorrectas: {incorrectos} ({100-accuracy:.2f}%)")
            print(f"\n--- Desglose por Tipo de Decisión ---")
            print(f"Casos decididos por Decreto 34: {casos_por_decreto} ({acc_decreto:.2f}% accuracy)")
            print(f"Casos decididos por ML: {casos_por_ml} ({acc_ml:.2f}% accuracy)")
            print(f"{'='*60}\n")

    # Eliminar columnas sin nombre (Unnamed, Columna1, etc.) que vienen del Excel
    cols_to_drop = [col for col in df_out.columns if 'unnamed' in str(col).lower() or str(col).startswith('Columna')]
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)
        print(f"Columnas eliminadas: {cols_to_drop}")

    out_full = Path("Prediccion.csv")
    try:
        df_out.to_csv(out_full, index=False)
    except PermissionError:
        out_full = out_full.with_name("Prediccion_xgb.csv")
        df_out.to_csv(out_full, index=False)
    
    print(f"Archivo completo guardado en {out_full} ({len(df_out)} filas)")


if __name__ == "__main__":
    main()