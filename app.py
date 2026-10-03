import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Predicción de Aprobación",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ESTILOS CSS
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
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
    }

    .header-card h1 {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .header-card p {
        font-size: 17px;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 28px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
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
        margin-bottom: 20px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
    }

    div[data-baseweb="input"] > div {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
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
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.35);
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
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.12);
    }

    .result-title {
        color: #065f46;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 5px;
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

    .info-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 15px 18px;
        border-radius: 10px;
        color: #1e40af;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .upload-box {
        background: #f8fafc;
        border: 2px dashed #94a3b8;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 25px 0 10px 0;
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
# ENCABEZADO
# ============================================================

st.markdown("""
<div class="header-card">

    <div style="font-size: 52px;">🎓</div>

    <h1>Predicción de Aprobación</h1>

    <p>
        Sistema inteligente para estimar la nota final
        utilizando Machine Learning.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DESCRIPCIÓN
# ============================================================

st.markdown("""
<div class="custom-card">

    <div class="section-title">
        📊 Sistema de Predicción Académica
    </div>

    <div class="section-description">
        Puedes realizar una predicción individual o cargar
        un archivo Excel para procesar varios estudiantes
        automáticamente.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PESTAÑAS
# ============================================================

tab1, tab2 = st.tabs([
    "👤 Predicción individual",
    "📂 Predicción con Excel"
])


# ============================================================
# OPCIONES FELDER
# ============================================================

opciones_felder = [
    'sensorial',
    'activo',
    'visual',
    'equilibrio',
    'secuencial',
    'reflexivo',
    'verbal',
    'intuitivo'
]


# ============================================================
# FUNCIONES DEL MODELO
# ============================================================

def cargar_componentes():

    one_hot_transformer = joblib.load(
        'one_hot_columns.joblib'
    )

    scaler = joblib.load(
        'min_max_scaler.joblib'
    )

    model = joblib.load(
        'bagging_optimizado.joblib'
    )

    return one_hot_transformer, scaler, model


def procesar_datos(df_input):

    one_hot_transformer, scaler, model = cargar_componentes()

    df_procesado = df_input.copy()

    # --------------------------------------------------------
    # Validar columnas
    # --------------------------------------------------------

    columnas_necesarias = [
        'Felder',
        'Examen_admisión'
    ]

    for columna in columnas_necesarias:

        if columna not in df_procesado.columns:

            raise ValueError(
                f"El archivo no contiene la columna "
                f"'{columna}'."
            )


    # --------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------

    if isinstance(one_hot_transformer, list):

        si_columnas_one_hot = [
            col
            for col in one_hot_transformer
            if 'Felder_' in col
        ]

        for col_name in si_columnas_one_hot:

            valor_esperado = col_name.replace(
                'Felder_',
                ''
            )

            df_procesado[col_name] = (
                df_procesado['Felder'] == valor_esperado
            ).astype(int)

    else:

        df_encoded = pd.get_dummies(
            df_procesado[['Felder']]
        )

        df_procesado = pd.concat(
            [
                df_procesado,
                df_encoded
            ],
            axis=1
        )

        si_columnas_one_hot = [
            col
            for col in df_procesado.columns
            if 'Felder_' in col
        ]


    # --------------------------------------------------------
    # Eliminar Felder
    # --------------------------------------------------------

    df_procesado = df_procesado.drop(
        columns=['Felder'],
        errors='ignore'
    )


    # --------------------------------------------------------
    # Asegurar columnas
    # --------------------------------------------------------

    if isinstance(one_hot_transformer, list):

        for col in si_columnas_one_hot:

            if col not in df_procesado.columns:

                df_procesado[col] = 0


    # --------------------------------------------------------
    # Normalizar examen
    # --------------------------------------------------------

    df_procesado['Examen_admision_scaled'] = (
        scaler.transform(
            df_procesado[['Examen_admisión']]
        )
    )


    # --------------------------------------------------------
    # Eliminar variable original
    # --------------------------------------------------------

    df_procesado = df_procesado.drop(
        columns=['Examen_admisión'],
        errors='ignore'
    )


    # --------------------------------------------------------
    # Ordenar columnas
    # --------------------------------------------------------

    columnas_ordenadas = (
        si_columnas_one_hot
        + ['Examen_admision_scaled']
    )

    df_procesado = df_procesado[
        columnas_ordenadas
    ]


    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    predicciones = model.predict(
        df_procesado
    )

    return (
        df_procesado,
        predicciones
    )


# ============================================================
# TAB 1 - PREDICCIÓN INDIVIDUAL
# ============================================================

with tab1:

    st.markdown("""
    <div class="custom-card">

        <div class="section-title">
            👨‍🎓 Datos del estudiante
        </div>

        <div class="section-description">
            Ingresa la información del estudiante para
            obtener una predicción.
        </div>

    </div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### 🧠 Estilo de aprendizaje")

        felder_input = st.selectbox(
            "Selecciona el estilo:",
            opciones_felder,
            label_visibility="collapsed"
        )


    with col2:

        st.markdown("### 📝 Examen de admisión")

        examen_input = st.number_input(
            "Puntuación:",
            min_value=0.0,
            max_value=5.0,
            value=3.83,
            step=0.01,
            format="%.2f",
            label_visibility="collapsed"
        )


    # Información de los datos

    col_info1, col_info2 = st.columns(2)

    with col_info1:

        st.metric(
            "🧠 Estilo seleccionado",
            felder_input.capitalize()
        )

    with col_info2:

        st.metric(
            "📝 Examen",
            f"{examen_input:.2f}"
        )


    st.markdown("<br>", unsafe_allow_html=True)


    if st.button(
        "🚀 REALIZAR PREDICCIÓN",
        key="prediccion_individual"
    ):

        try:

            df_input = pd.DataFrame({
                'Felder': [felder_input],
                'Examen_admisión': [examen_input]
            })


            df_procesado, predicciones = procesar_datos(
                df_input
            )


            resultado = float(
                predicciones[0]
            )


            st.markdown("""
            <div class="custom-card">

                <div class="section-title">
                    ⚙️ Datos procesados
                </div>

                <div class="section-description">
                    Variables utilizadas por el modelo.
                </div>

            </div>
            """, unsafe_allow_html=True)


            st.dataframe(
                df_procesado,
                use_container_width=True,
                hide_index=True
            )


            st.markdown(f"""
            <div class="result-card">

                <div class="result-title">
                    🎯 Nota Final Estimada
                </div>

                <div class="result-value">
                    {resultado:.4f}
                </div>

                <div class="result-subtitle">
                    Predicción generada por el modelo de Bagging
                </div>

            </div>
            """, unsafe_allow_html=True)


        except Exception as e:

            st.error(
                f"❌ Error durante la predicción: {e}"
            )


# ============================================================
# TAB 2 - PREDICCIÓN CON EXCEL
# ============================================================

with tab2:

    st.markdown("""
    <div class="custom-card">

        <div class="section-title">
            📂 Predicción mediante archivo Excel
        </div>

        <div class="section-description">
            Sube un archivo Excel con los estudiantes y
            el sistema realizará las predicciones
            automáticamente.
        </div>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Formato esperado
    # --------------------------------------------------------

    st.markdown("""
    <div class="info-box">

        📌 <b>Formato requerido del Excel</b><br><br>

        El archivo debe contener exactamente estas columnas:

        <br><br>

        <b>Felder</b> → estilo de aprendizaje

        <br>

        <b>Examen_admisión</b> → nota del examen de admisión

        <br><br>

        Ejemplo:

    </div>
    """, unsafe_allow_html=True)


    ejemplo = pd.DataFrame({
        'Felder': [
            'sensorial',
            'activo',
            'visual',
            'reflexivo'
        ],
        'Examen_admisión': [
            3.83,
            4.20,
            3.50,
            4.75
        ]
    })


    st.dataframe(
        ejemplo,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Cargar archivo
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)


    archivo_excel = st.file_uploader(
        "📁 Selecciona tu archivo Excel",
        type=["xlsx", "xls"],
        help="Sube un archivo Excel con las columnas Felder y Examen_admisión."
    )


    if archivo_excel is not None:

        try:

            # ------------------------------------------------
            # Leer Excel
            # ------------------------------------------------

            df_excel = pd.read_excel(
                archivo_excel
            )


            st.success(
                f"✅ Archivo cargado correctamente: "
                f"{len(df_excel)} estudiantes encontrados."
            )


            # ------------------------------------------------
            # Mostrar datos originales
            # ------------------------------------------------

            st.markdown("### 👀 Datos cargados")

            st.dataframe(
                df_excel.head(10),
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Información
            # ------------------------------------------------

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

                memoria = df_excel.memory_usage(
                    deep=True
                ).sum() / 1024

                st.metric(
                    "💾 Tamaño",
                    f"{memoria:.1f} KB"
                )


            st.markdown("<br>", unsafe_allow_html=True)


            # ------------------------------------------------
            # Botón procesar
            # ------------------------------------------------

            if st.button(
                "🚀 PROCESAR ARCHIVO EXCEL",
                key="procesar_excel"
            ):

                try:

                    # ----------------------------------------
                    # Validar columnas
                    # ----------------------------------------

                    columnas_requeridas = [
                        'Felder',
                        'Examen_admisión'
                    ]


                    columnas_faltantes = [
                        col
                        for col in columnas_requeridas
                        if col not in df_excel.columns
                    ]


                    if columnas_faltantes:

                        st.error(
                            "❌ Faltan las siguientes columnas: "
                            + ", ".join(
                                columnas_faltantes
                            )
                        )

                        st.stop()


                    # ----------------------------------------
                    # Validar Felder
                    # ----------------------------------------

                    valores_invalidos = (
                        df_excel[
                            ~df_excel['Felder'].isin(
                                opciones_felder
                            )
                        ]['Felder']
                        .dropna()
                        .unique()
                        .tolist()
                    )


                    if valores_invalidos:

                        st.error(
                            "❌ Se encontraron estilos "
                            "de aprendizaje no válidos:"
                        )

                        st.write(
                            valores_invalidos
                        )

                        st.info(
                            "Los valores permitidos son: "
                            + ", ".join(opciones_felder)
                        )

                        st.stop()


                    # ----------------------------------------
                    # Validar examen
                    # ----------------------------------------

                    if (
                        df_excel['Examen_admisión']
                        .isna()
                        .any()
                    ):

                        st.error(
                            "❌ Existen valores vacíos "
                            "en Examen_admisión."
                        )

                        st.stop()


                    if (
                        (df_excel['Examen_admisión'] < 0)
                        .any()
                        or
                        (df_excel['Examen_admisión'] > 5)
                        .any()
                    ):

                        st.error(
                            "❌ El Examen_admisión debe "
                            "estar entre 0 y 5."
                        )

                        st.stop()


                    # ----------------------------------------
                    # Procesar modelo
                    # ----------------------------------------

                    with st.spinner(
                        "🤖 Procesando estudiantes..."
                    ):

                        df_procesado, predicciones = (
                            procesar_datos(
                                df_excel[
                                    columnas_requeridas
                                ]
                            )
                        )


                    # ----------------------------------------
                    # Agregar predicciones
                    # ----------------------------------------

                    df_resultado = df_excel.copy()


                    df_resultado[
                        'Nota_Final_Estimada'
                    ] = np.round(
                        predicciones,
                        4
                    )


                    # ----------------------------------------
                    # Mostrar resultados
                    # ----------------------------------------

                    st.success(
                        "🎉 ¡Predicción completada!"
                    )


                    st.markdown("### 🎯 Resultados")


                    st.dataframe(
                        df_resultado,
                        use_container_width=True,
                        hide_index=True
                    )


                    # ----------------------------------------
                    # Estadísticas
                    # ----------------------------------------

                    st.markdown("### 📊 Resumen de predicciones")


                    col1, col2, col3 = st.columns(3)


                    with col1:

                        st.metric(
                            "👥 Estudiantes",
                            len(df_resultado)
                        )


                    with col2:

                        promedio = (
                            df_resultado[
                                'Nota_Final_Estimada'
                            ].mean()
                        )

                        st.metric(
                            "📈 Promedio estimado",
                            f"{promedio:.4f}"
                        )


                    with col3:

                        maximo = (
                            df_resultado[
                                'Nota_Final_Estimada'
                            ].max()
                        )

                        st.metric(
                            "🏆 Mayor predicción",
                            f"{maximo:.4f}"
                        )


                    # ----------------------------------------
                    # Crear Excel descargable
                    # ----------------------------------------

                    output = BytesIO()


                    with pd.ExcelWriter(
                        output,
                        engine='openpyxl'
                    ) as writer:

                        df_resultado.to_excel(
                            writer,
                            index=False,
                            sheet_name='Predicciones'
                        )


                    output.seek(0)


                    # ----------------------------------------
                    # Descargar
                    # ----------------------------------------

                    st.markdown("<br>", unsafe_allow_html=True)


                    st.download_button(
                        label="📥 DESCARGAR EXCEL CON PREDICCIONES",
                        data=output,
                        file_name="predicciones_estudiantes.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                        key="descargar_excel"
                    )


                except Exception as e:

                    st.error(
                        "❌ Ocurrió un error al procesar "
                        f"el archivo: {e}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    🎓 Sistema de Predicción Académica
    <br>
    Machine Learning + Streamlit

</div>
""", unsafe_allow_html=True)

