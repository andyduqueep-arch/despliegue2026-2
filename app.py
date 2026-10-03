
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Predicción de Aprobación",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 50%,
        #e0f2fe 100%
    );
}

.header-card {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #2563eb
    );

    padding: 35px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin-bottom: 25px;

    box-shadow:
        0 12px 30px rgba(37, 99, 235, 0.25);
}

.header-card h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 5px 0 10px 0;
}

.header-card p {
    font-size: 17px;
    opacity: 0.92;
    margin: 0;
}

.custom-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 28px;
    border-radius: 20px;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.08);

    border: 1px solid rgba(226, 232, 240, 0.8);

    margin-bottom: 20px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #1e293b;
    margin-bottom: 5px;
}

.section-description {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 10px;
}

.info-box {
    background: #eff6ff;
    border-left: 5px solid #3b82f6;

    padding: 15px 18px;

    border-radius: 10px;

    color: #1e40af;

    margin-top: 15px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #d1fae5
    );

    border: 1px solid #86efac;

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    margin-top: 25px;

    box-shadow:
        0 8px 25px rgba(16, 185, 129, 0.12);
}

.result-title {
    color: #065f46;
    font-size: 18px;
    font-weight: 600;
}

.result-value {
    color: #047857;
    font-size: 48px;
    font-weight: 850;
    margin: 5px 0;
}

.result-subtitle {
    color: #047857;
    font-size: 14px;
}

.stButton > button {
    width: 100%;
    height: 55px;

    border-radius: 14px;

    border: none;

    background: linear-gradient(
        135deg,
        #4f46e5,
        #2563eb
    );

    color: white;

    font-size: 17px;
    font-weight: 700;

    box-shadow:
        0 8px 18px rgba(37, 99, 235, 0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 25px rgba(37, 99, 235, 0.35);
}

div[data-baseweb="select"] > div {
    border-radius: 12px;
}

div[data-baseweb="input"] > div {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 30px 0 10px 0;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {

    .header-card h1 {
        font-size: 29px;
    }

    .header-card {
        padding: 25px 18px;
    }

    .custom-card {
        padding: 20px;
    }

    .result-value {
        font-size: 40px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# OPCIONES FELDER
# ============================================================

OPCIONES_FELDER = [
    "sensorial",
    "activo",
    "visual",
    "equilibrio",
    "secuencial",
    "reflexivo",
    "verbal",
    "intuitivo"
]


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelos():

    try:

        one_hot = joblib.load(
            "one_hot_columns.joblib"
        )

        scaler = joblib.load(
            "min_max_scaler.joblib"
        )

        model = joblib.load(
            "bagging_optimizado.joblib"
        )

        return one_hot, scaler, model

    except FileNotFoundError as e:

        st.error(
            "❌ No se encontró uno de los archivos del modelo."
        )

        st.info(
            "Asegúrate de tener en la misma carpeta de app.py:"
        )

        st.code("""
one_hot_columns.joblib
min_max_scaler.joblib
bagging_optimizado.joblib
        """)

        st.stop()


# ============================================================
# FUNCIÓN PARA PROCESAR DATOS
# ============================================================

def procesar_datos(df):

    # --------------------------------------------------------
    # Copia
    # --------------------------------------------------------

    df = df.copy()


    # --------------------------------------------------------
    # Limpiar nombres de columnas
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Aceptar ambas versiones de examen
    # --------------------------------------------------------

    if (
        "Examen_admisión" not in df.columns
        and
        "Examen_admision" in df.columns
    ):

        df = df.rename(
            columns={
                "Examen_admision":
                "Examen_admisión"
            }
        )


    # --------------------------------------------------------
    # Validar columnas
    # --------------------------------------------------------

    columnas_requeridas = [
        "Felder",
        "Examen_admisión"
    ]

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Faltan las columnas: "
            + ", ".join(faltantes)
        )


    # --------------------------------------------------------
    # Limpiar Felder
    # --------------------------------------------------------

    df["Felder"] = (
        df["Felder"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    # --------------------------------------------------------
    # Validar Felder
    # --------------------------------------------------------

    invalidos = (
        ~df["Felder"].isin(
            OPCIONES_FELDER
        )
    )

    if invalidos.any():

        valores = (
            df.loc[
                invalidos,
                "Felder"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Valores de Felder no válidos: "
            + ", ".join(
                map(str, valores)
            )
        )


    # --------------------------------------------------------
    # Convertir examen a número
    # --------------------------------------------------------

    df["Examen_admisión"] = pd.to_numeric(
        df["Examen_admisión"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Validar examen
    # --------------------------------------------------------

    if df["Examen_admisión"].isna().any():

        raise ValueError(
            "Hay valores vacíos o no numéricos "
            "en Examen_admisión."
        )


    if (
        (df["Examen_admisión"] < 0).any()
        or
        (df["Examen_admisión"] > 5).any()
    ):

        raise ValueError(
            "Los valores de Examen_admisión "
            "deben estar entre 0 y 5."
        )


    # --------------------------------------------------------
    # Cargar componentes
    # --------------------------------------------------------

    one_hot, scaler, model = cargar_modelos()


    # --------------------------------------------------------
    # ONE-HOT
    # --------------------------------------------------------

    if isinstance(one_hot, list):

        columnas_felder = [
            columna
            for columna in one_hot
            if columna.startswith("Felder_")
        ]

        for columna in columnas_felder:

            valor = columna.replace(
                "Felder_",
                ""
            )

            df[columna] = (
                df["Felder"] == valor
            ).astype(int)

    else:

        # Si se guardó un encoder de sklearn
        try:

            encoded = one_hot.transform(
                df[["Felder"]]
            )

            if hasattr(
                one_hot,
                "get_feature_names_out"
            ):

                nombres = (
                    one_hot
                    .get_feature_names_out(
                        ["Felder"]
                    )
                )

                encoded_df = pd.DataFrame(
                    encoded,
                    columns=nombres,
                    index=df.index
                )

            else:

                encoded_df = pd.DataFrame(
                    encoded,
                    index=df.index
                )

            df = pd.concat(
                [
                    df,
                    encoded_df
                ],
                axis=1
            )

            columnas_felder = [
                columna
                for columna in df.columns
                if str(columna).startswith(
                    "Felder_"
                )
            ]

        except Exception:

            encoded_df = pd.get_dummies(
                df["Felder"],
                prefix="Felder"
            )

            df = pd.concat(
                [
                    df,
                    encoded_df
                ],
                axis=1
            )

            columnas_felder = [
                columna
                for columna in df.columns
                if str(columna).startswith(
                    "Felder_"
                )
            ]


    # --------------------------------------------------------
    # Asegurar columnas
    # --------------------------------------------------------

    if isinstance(one_hot, list):

        for columna in columnas_felder:

            if columna not in df.columns:

                df[columna] = 0


    # --------------------------------------------------------
    # Eliminar Felder
    # --------------------------------------------------------

    df = df.drop(
        columns=["Felder"],
        errors="ignore"
    )


    # --------------------------------------------------------
    # NORMALIZAR EXAMEN
    # --------------------------------------------------------

    examen_escalado = scaler.transform(
        df[
            ["Examen_admisión"]
        ]
    )


    df["Examen_admision_scaled"] = (
        np.asarray(
            examen_escalado
        ).reshape(-1)
    )


    # --------------------------------------------------------
    # Eliminar examen original
    # --------------------------------------------------------

    df = df.drop(
        columns=["Examen_admisión"],
        errors="ignore"
    )


    # --------------------------------------------------------
    # ORDEN DE COLUMNAS
    # --------------------------------------------------------

    columnas_modelo = (
        columnas_felder
        + [
            "Examen_admision_scaled"
        ]
    )


    df_modelo = df[
        columnas_modelo
    ].copy()


    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    predicciones = model.predict(
        df_modelo
    )


    return (
        df_modelo,
        np.asarray(predicciones)
    )


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown("""
<div class="header-card">

    <div style="font-size: 55px;">
        🎓
    </div>

    <h1>
        Predicción de Aprobación
    </h1>

    <p>
        Sistema inteligente para estimar la nota final
        utilizando Machine Learning.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INFORMACIÓN
# ============================================================

st.markdown("""
<div class="custom-card">

    <div class="section-title">
        📊 Sistema de Predicción Académica
    </div>

    <div class="section-description">

        Selecciona una opción para realizar una predicción
        individual o procesar múltiples estudiantes mediante
        un archivo Excel.

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PESTAÑAS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "👤 Predicción individual",
        "📂 Predicción con Excel"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.markdown("""
    <div class="custom-card">

        <div class="section-title">
            👨‍🎓 Datos del estudiante
        </div>

        <div class="section-description">
            Introduce los datos para generar una predicción.
        </div>

    </div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 🧠 Estilo de aprendizaje"
        )

        felder_input = st.selectbox(
            "Estilo:",
            OPCIONES_FELDER,
            label_visibility="collapsed"
        )


    with col2:

        st.markdown(
            "### 📝 Examen de admisión"
        )

        examen_input = st.number_input(
            "Examen:",
            min_value=0.0,
            max_value=5.0,
            value=3.83,
            step=0.01,
            format="%.2f",
            label_visibility="collapsed"
        )


    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "🧠 Estilo seleccionado",
            felder_input.capitalize()
        )


    with col2:

        st.metric(
            "📝 Examen",
            f"{examen_input:.2f}"
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # Botón
    # --------------------------------------------------------

    if st.button(
        "🚀 REALIZAR PREDICCIÓN",
        key="boton_individual"
    ):

        try:

            df_input = pd.DataFrame(
                {
                    "Felder": [
                        felder_input
                    ],
                    "Examen_admisión": [
                        examen_input
                    ]
                }
            )


            with st.spinner(
                "🤖 Ejecutando modelo..."
            ):

                df_modelo, predicciones = (
                    procesar_datos(
                        df_input
                    )
                )


            resultado = float(
                predicciones[0]
            )


            # ------------------------------------------------
            # Datos procesados
            # ------------------------------------------------

            st.markdown("""
            <div class="custom-card">

                <div class="section-title">
                    ⚙️ Datos procesados
                </div>

                <div class="section-description">
                    Variables enviadas al modelo.
                </div>

            </div>
            """, unsafe_allow_html=True)


            st.dataframe(
                df_modelo,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Resultado
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-title">
                        🎯 Nota Final Estimada
                    </div>

                    <div class="result-value">
                        {resultado:.4f}
                    </div>

                    <div class="result-subtitle">
                        Predicción generada por el modelo
                        de Bagging
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                f"❌ Error durante la predicción: {e}"
            )


# ============================================================
# TAB 2 - EXCEL
# ============================================================

with tab2:

    st.markdown("""
    <div class="custom-card">

        <div class="section-title">
            📂 Predicción mediante Excel
        </div>

        <div class="section-description">

            Carga un archivo Excel para realizar
            predicciones de múltiples estudiantes
            automáticamente.

        </div>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Formato requerido
    # --------------------------------------------------------

    st.markdown("""
    <div class="info-box">

        📌 <b>Formato del archivo</b>

        <br><br>

        El Excel debe contener estas columnas:

        <br><br>

        <b>Felder</b>

        <br>

        Estilo de aprendizaje.

        <br><br>

        <b>Examen_admisión</b>

        <br>

        Nota del examen entre 0 y 5.

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Ejemplo
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Ejemplo de Excel"
    )


    ejemplo = pd.DataFrame(
        {
            "Felder": [
                "sensorial",
                "activo",
                "visual",
                "reflexivo"
            ],
            "Examen_admisión": [
                3.83,
                4.20,
                3.50,
                4.75
            ]
        }
    )


    st.dataframe(
        ejemplo,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Cargar archivo
    # --------------------------------------------------------

    st.markdown(
        "### 📁 Seleccionar archivo"
    )


    archivo = st.file_uploader(
        "Sube tu archivo Excel",
        type=["xlsx", "xls"],
        help=(
            "El archivo debe contener Felder "
            "y Examen_admisión."
        )
    )


    if archivo is not None:

        try:

            # ------------------------------------------------
            # Leer archivo
            # ------------------------------------------------

            df_excel = pd.read_excel(
                archivo
            )


            # ------------------------------------------------
            # Limpiar nombres
            # ------------------------------------------------

            df_excel.columns = (
                df_excel.columns
                .astype(str)
                .str.strip()
            )


            # ------------------------------------------------
            # Mostrar información
            # ------------------------------------------------

            st.success(
                "✅ Archivo cargado correctamente."
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "👥 Estudiantes",
                    len(df_excel)
                )


            with col2:

                st.metric(
                    "📋 Columnas",
                    len(df_excel.columns)
                )


            with col3:

                st.metric(
                    "📊 Filas",
                    df_excel.shape[0]
                )


            st.markdown(
                "### 👀 Datos cargados"
            )


            st.dataframe(
                df_excel,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Procesar
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)


            if st.button(
                "🚀 PROCESAR ARCHIVO EXCEL",
                key="boton_excel"
            ):

                try:

                    with st.spinner(
                        "🤖 Procesando estudiantes..."
                    ):

                        df_modelo, predicciones = (
                            procesar_datos(
                                df_excel
                            )
                        )


                    # ----------------------------------------
                    # Crear resultado
                    # ----------------------------------------

                    df_resultado = (
                        df_excel.copy()
                    )


                    df_resultado[
                        "Nota_Final_Estimada"
                    ] = np.round(
                        predicciones,
                        4
                    )


                    # ----------------------------------------
                    # Resultado
                    # ----------------------------------------

                    st.success(
                        "🎉 ¡Predicciones generadas correctamente!"
                    )


                    st.markdown(
                        "### 🎯 Resultados"
                    )


                    st.dataframe(
                        df_resultado,
                        use_container_width=True,
                        hide_index=True
                    )


                    # ----------------------------------------
                    # Estadísticas
                    # ----------------------------------------

                    st.markdown(
                        "### 📊 Resumen"
                    )


                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )


                    with col1:

                        st.metric(
                            "👥 Estudiantes",
                            len(df_resultado)
                        )


                    with col2:

                        promedio = (
                            df_resultado[
                                "Nota_Final_Estimada"
                            ].mean()
                        )

                        st.metric(
                            "📈 Promedio",
                            f"{promedio:.4f}"
                        )


                    with col3:

                        minimo = (
                            df_resultado[
                                "Nota_Final_Estimada"
                            ].min()
                        )

                        st.metric(
                            "📉 Mínimo",
                            f"{minimo:.4f}"
                        )


                    with col4:

                        maximo = (
                            df_resultado[
                                "Nota_Final_Estimada"
                            ].max()
                        )

                        st.metric(
                            "📈 Máximo",
                            f"{maximo:.4f}"
                        )


                    # ----------------------------------------
                    # Crear Excel de salida
                    # ----------------------------------------

                    output = BytesIO()


                    with pd.ExcelWriter(
                        output,
                        engine="openpyxl"
                    ) as writer:

                        df_resultado.to_excel(
                            writer,
                            index=False,
                            sheet_name="Predicciones"
                        )


                    output.seek(0)


                    # ----------------------------------------
                    # Descargar
                    # ----------------------------------------

                    st.markdown(
                        "### 📥 Descargar resultados"
                    )


                    st.download_button(
                        label=(
                            "📥 DESCARGAR EXCEL "
                            "CON PREDICCIONES"
                        ),
                        data=output.getvalue(),
                        file_name=(
                            "predicciones_estudiantes.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                        key="descargar_resultados"
                    )


                except Exception as e:

                    st.error(
                        "❌ No se pudo procesar el Excel."
                    )

                    st.exception(e)


        except Exception as e:

            st.error(
                "❌ No se pudo leer el archivo Excel."
            )

            st.exception(e)


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("""
<div class="footer">

    🎓 Sistema de Predicción Académica

    <br>

    Machine Learning + Streamlit

</div>
""", unsafe_allow_html=True)
