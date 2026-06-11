import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sklearn as skl
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)

# ──────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Câncer de Mama · Análise Tumoral",
    page_icon="🎗️",
    layout="wide",
)

# ──────────────────────────────────────────────
# TOGGLE DE MODO
# ──────────────────────────────────────────────
modo_escuro = st.sidebar.toggle("🌙 Modo escuro", value=False)

if modo_escuro:
    BG_APP      = "#0f0f1a"
    BG_SIDEBAR  = "#1a1025"
    BG_CARD     = "#1e1030"
    BG_PLOT     = "#16102a"
    BG_INFOBOX  = "#2a1040"
    COR_BORDA   = "#5a2555"
    COR_TITULO  = "#f5c2d9"
    COR_TEXTO   = "#d4a0be"
    COR_CAPTION = "#7a5070"
    COR_GRID    = "#2a1040"
    COR_MET_V   = "#f9d4e8"
    COR_MET_L   = "#c77daa"
else:
    BG_APP      = "#fdf6f9"
    BG_SIDEBAR  = "#ebb7d3"
    BG_CARD     = "#ffffff"
    BG_PLOT     = "#fff8fb"
    BG_INFOBOX  = "#fff0f6"
    COR_BORDA   = "#d194b1"
    COR_TITULO  = "#7b1c4b"
    COR_TEXTO   = "#5a1a35"
    COR_CAPTION = "#9b4468"
    COR_GRID    = "#f0d0de"
    COR_MET_V   = "#7b1c4b"
    COR_MET_L   = "#9b4468"

# ──────────────────────────────────────────────
# ESTILOS GLOBAIS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Lato', sans-serif; }}
  .stApp {{ background-color: {BG_APP}; }}

  section[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR};
    border-right: 2px solid {COR_BORDA};
  }}

  h1, h2, h3 {{
    color: {COR_TITULO} !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    text-align: center !important;
  }}

  p, span, label {{ color: {COR_TEXTO}; }}

  [data-testid="metric-container"] {{
    background: {BG_CARD};
    border: 1px solid {COR_BORDA};
    border-left: 5px solid #c0396e;
    border-radius: 10px;
    padding: 16px !important;
    text-align: center;
  }}
  [data-testid="metric-container"] label {{
    color: {COR_MET_L} !important;
    font-size: 0.8rem !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {COR_MET_V} !important;
    font-size: 1.8rem !important;
  }}

  .stTabs [data-baseweb="tab-list"] {{ border-bottom: 2px solid {COR_BORDA}; }}
  .stTabs [data-baseweb="tab"] {{ color: {COR_CAPTION}; font-weight: 600; }}
  .stTabs [aria-selected="true"] {{
    color: #c0396e !important;
    border-bottom: 3px solid #c0396e !important;
  }}

  .info-box {{
    background: {BG_INFOBOX};
    border-left: 4px solid #c0396e;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: {COR_TEXTO};
    margin-top: 8px;
    text-align: center;
  }}

  .svm-box {{
    background: {BG_INFOBOX};
    border-left: 4px solid #c0396e;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.9rem;
    color: {COR_TEXTO};
    margin: 12px 0;
    line-height: 1.7;
  }}

  .metric-badge {{
    display: inline-block;
    background: {BG_CARD};
    border: 1px solid {COR_BORDA};
    border-radius: 8px;
    padding: 10px 18px;
    margin: 6px;
    text-align: center;
    min-width: 110px;
  }}
  .metric-badge .val {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #c0396e;
    display: block;
  }}
  .metric-badge .lbl {{
    font-size: 0.75rem;
    color: {COR_CAPTION};
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DADOS
# ──────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_excel("dataset_cancer_mama_02.xlsx")
    df = df[df["raio médio"] < 100].copy()
    df["diagnóstico_label"] = df["diagnóstico"].map({0: "Maligno", 1: "Benigno"})
    return df

df = carregar_dados()

FEATURES = [
    "raio médio", "textura média", "perímetro médio", "área média",
    "suavidade média", "compacidade média", "concavidade média",
    "pontos côncavos médios", "simetria média", "dimensão fractal média",
    "erro do raio", "erro da textura", "erro do perímetro", "erro da área",
    "erro da suavidade", "erro da compacidade", "erro da concavidade",
    "erro dos pontos côncavos", "erro da simetria", "erro da dimensão fractal",
    "pior raio", "pior textura", "pior perímetro", "pior área",
    "pior suavidade", "pior compacidade", "pior concavidade",
    "piores pontos côncavos", "pior simetria", "pior dimensão fractal",
]

COR_MAL = "#c0396e"
COR_BEN = "#62c83a"
CORES   = {"Maligno": COR_MAL, "Benigno": COR_BEN}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_PLOT,
    font=dict(family="Lato", color=COR_TEXTO, size=13),
    xaxis=dict(gridcolor=COR_GRID, showline=True, linecolor=COR_BORDA),
    yaxis=dict(gridcolor=COR_GRID, showline=True, linecolor=COR_BORDA),
    legend=dict(bgcolor=BG_CARD, bordercolor=COR_BORDA, borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<h2 style='text-align:center;color:{COR_TITULO};'>🎗️ Câncer de Mama</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:{COR_TEXTO};'>Análise de Características Tumorais</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;font-weight:700;color:{COR_TITULO};'>Tipos de Diagnóstico</p>",
        unsafe_allow_html=True,
    )
    diag_sel = st.multiselect(
        "Selecione:",
        options=["Maligno", "Benigno"],
        default=["Maligno", "Benigno"],
    )
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;font-size:0.78rem;color:{COR_CAPTION};'>"
        "Dataset Wisconsin Breast Cancer · 569 amostras · 30 features</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        f"<h3 style='text-align:center;font-weight:700;color:{COR_TITULO};'>Contatos 📩</h3>",
     unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:8px;">
          <a href="https://github.com/mariacamargo-ds" target="_blank"
             style="display:inline-block; background-color:{COR_CAPTION};
                    color:#ffffff; text-decoration:none; padding:10px 20px;
                    border-radius:8px; font-size:0.88rem; font-weight:700;
                    letter-spacing:0.04em;">
            💻 GitHub
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

dff = df[df["diagnóstico_label"].isin(diag_sel)].copy() if diag_sel else df.copy()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def titulo(texto):
    st.markdown(f"<h3>{texto}</h3>", unsafe_allow_html=True)

def infobox(texto):
    st.markdown(
        f"<div class='info-box'>ℹ️ <b>Informações sobre o Gráfico</b><br>{texto}</div>",
        unsafe_allow_html=True,
    )

def svmbox(texto):
    st.markdown(f"<div class='svm-box'>{texto}</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────
st.markdown(
    "<h1>🎗️ Câncer de Mama: Análise de Características Tumorais</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:{COR_TEXTO};'>"
    "Exploração de biomarcadores extraídos por imagem digital de biópsia por agulha fina (FNA).</p>",
    unsafe_allow_html=True,
)
st.divider()

# ──────────────────────────────────────────────
# CARDS
# ──────────────────────────────────────────────
n_total = len(dff)
n_mal   = (dff["diagnóstico_label"] == "Maligno").sum()
n_ben   = (dff["diagnóstico_label"] == "Benigno").sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total de Amostras", f"{n_total}")
c2.metric("Casos Malignos",    f"{n_mal}",
          f"{n_mal/n_total*100:.1f}% do total" if n_total else "")
c3.metric("Casos Benignos",    f"{n_ben}",
          f"{n_ben/n_total*100:.1f}% do total" if n_total else "")

st.divider()

# ──────────────────────────────────────────────
# ABAS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Visão Geral",
    "Textura Média",
    "Área × Concavidade",
    "Compacidade Média",
    "Simetria × Concavidade",
    "Classificador SVM",
])

# ════════════════════════════════════════════
# TAB 1 · VISÃO GERAL
# ════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        titulo("Área Média por Diagnóstico")
        media_area = (
            dff.groupby("diagnóstico_label")["área média"]
            .mean().reset_index()
            .rename(columns={"área média": "Área Média (mm²)",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig1 = px.bar(
            media_area, x="Diagnóstico", y="Área Média (mm²)",
            color="Diagnóstico", color_discrete_map=CORES, text_auto=".0f",
        )
        fig1.update_traces(textposition="outside", textfont_size=13)
        fig1.update_layout(**LAYOUT_BASE, height=360, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        infobox("Tumores malignos apresentam área média significativamente maior, "
                "refletindo crescimento celular desordenado e expansão tumoral.")

    with col_b:
        titulo("Concavidade Média por Diagnóstico")
        media_conc = (
            dff.groupby("diagnóstico_label")["concavidade média"]
            .mean().reset_index()
            .rename(columns={"concavidade média": "Concavidade Média",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig2 = px.bar(
            media_conc, x="Diagnóstico", y="Concavidade Média",
            color="Diagnóstico", color_discrete_map=CORES, text_auto=".4f",
        )
        fig2.update_traces(textposition="outside", textfont_size=13)
        fig2.update_layout(**LAYOUT_BASE, height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        infobox("A concavidade descreve a severidade das reentrâncias no contorno celular. "
                "Valores mais altos indicam bordas irregulares, típicas de células malignas.")

# ════════════════════════════════════════════
# TAB 2 · TEXTURA MÉDIA
# ════════════════════════════════════════════
with tab2:
    titulo("Textura Média por Tipos de Diagnóstico")
    media_tex = (
        dff.groupby("diagnóstico_label")["textura média"]
        .mean().reset_index()
        .rename(columns={"textura média": "Textura Média",
                         "diagnóstico_label": "Diagnóstico"})
    )
    fig3 = px.bar(
        media_tex, x="Diagnóstico", y="Textura Média",
        color="Diagnóstico", color_discrete_map=CORES, text_auto=".2f",
    )
    fig3.update_traces(textposition="outside", textfont_size=14)
    fig3.update_layout(**LAYOUT_BASE, height=420, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    infobox("A textura mede o desvio-padrão dos valores de escala de cinza na imagem. "
            "Tecidos malignos tendem a apresentar textura mais heterogênea, "
            "decorrente da variabilidade nuclear elevada.")

# ════════════════════════════════════════════
# TAB 3 · ÁREA × CONCAVIDADE
# ════════════════════════════════════════════
with tab3:
    titulo("Área × Concavidade por Diagnóstico")
    fig4 = px.scatter(
        dff, x="área média", y="concavidade média",
        color="diagnóstico_label", color_discrete_map=CORES,
        labels={
            "área média": "Área Média (mm²)",
            "concavidade média": "Concavidade Média",
            "diagnóstico_label": "Diagnóstico",
        },
        opacity=0.75,
        hover_data={"área média": ":.1f", "concavidade média": ":.4f"},
    )
    fig4.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#fff")))
    fig4.update_layout(**LAYOUT_BASE, height=460)
    st.plotly_chart(fig4, use_container_width=True)
    infobox("A dispersão revela que casos malignos concentram-se em regiões de maior área "
            "<i>e</i> maior concavidade — as duas variáveis atuam juntas como marcadores "
            "de agressividade tumoral.")

# ════════════════════════════════════════════
# TAB 4 · COMPACIDADE MÉDIA
# ════════════════════════════════════════════
with tab4:
    titulo("Compacidade e Textura Média por Diagnóstico")
    col_c, col_d = st.columns(2)

    with col_c:
        media_comp = (
            dff.groupby("diagnóstico_label")["compacidade média"]
            .mean().reset_index()
            .rename(columns={"compacidade média": "Compacidade Média",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig5a = px.bar(
            media_comp, x="Diagnóstico", y="Compacidade Média",
            color="Diagnóstico", color_discrete_map=CORES,
            text_auto=".4f", title="Compacidade Média",
        )
        fig5a.update_traces(textposition="outside", textfont_size=13)
        fig5a.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                            title_x=0.5, title_font_color=COR_TITULO)
        st.plotly_chart(fig5a, use_container_width=True)

    with col_d:
        media_tex2 = (
            dff.groupby("diagnóstico_label")["textura média"]
            .mean().reset_index()
            .rename(columns={"textura média": "Textura Média",
                             "diagnóstico_label": "Diagnóstico"})
        )
        fig5b = px.bar(
            media_tex2, x="Diagnóstico", y="Textura Média",
            color="Diagnóstico", color_discrete_map=CORES,
            text_auto=".2f", title="Textura Média",
        )
        fig5b.update_traces(textposition="outside", textfont_size=13)
        fig5b.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                            title_x=0.5, title_font_color=COR_TITULO)
        st.plotly_chart(fig5b, use_container_width=True)

    infobox("A compacidade combina perímetro e área para medir a irregularidade do contorno. "
            "Quando associada à textura elevada, forma um par diagnóstico de alta relevância clínica.")

# ════════════════════════════════════════════
# TAB 5 · SIMETRIA × CONCAVIDADE
# ════════════════════════════════════════════
with tab5:
    titulo("Simetria Média × Concavidade Média por Diagnóstico")
    fig6 = px.scatter(
        dff, x="simetria média", y="concavidade média",
        color="diagnóstico_label", color_discrete_map=CORES,
        labels={
            "simetria média": "Simetria Média",
            "concavidade média": "Concavidade Média",
            "diagnóstico_label": "Diagnóstico",
        },
        opacity=0.75,
        hover_data={"simetria média": ":.4f", "concavidade média": ":.4f"},
    )
    fig6.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#fff")))
    fig6.update_layout(**LAYOUT_BASE, height=460)
    st.plotly_chart(fig6, use_container_width=True)
    infobox("Células malignas frequentemente exibem assimetria nuclear combinada com alta "
            "concavidade. A perda de simetria é um marcador morfológico clássico de malignidade "
            "tecidual.")

# ════════════════════════════════════════════
# TAB 6 · CLASSIFICADOR SVM
# ════════════════════════════════════════════
with tab6:
    st.markdown("<h2>🤖 Classificador SVM — Support Vector Machine</h2>",
                unsafe_allow_html=True)

    svmbox("""
    <b>Por que SVM neste contexto?</b><br><br>
    O Support Vector Machine é um dos algoritmos de classificação mais adequados para dados
    biomédicos de alta dimensionalidade — e o dataset Wisconsin Breast Cancer é um exemplo
    clássico disso: 30 features numéricas, classes bem separadas em algumas dimensões, mas
    com sobreposição em outras.<br><br>
    O SVM encontra o <i>hiperplano de margem máxima</i> que separa as classes no espaço
    das features. Com o kernel RBF (Radial Basis Function), ele consegue capturar fronteiras
    de decisão não-lineares, o que é especialmente útil quando variáveis como área, concavidade
    e compacidade interagem de forma complexa para diferenciar tumores malignos de benignos.<br><br>
    Além disso, o SVM é robusto em conjuntos de dados de tamanho moderado (como os 526 registros
    aqui presentes) e funciona bem mesmo com o leve desbalanceamento de classes existente neste
    dataset — desde que a normalização dos dados seja aplicada corretamente, o que fazemos aqui
    via <code>StandardScaler</code>.
    """)

    st.divider()

    # ── CONFIGURAÇÕES DO MODELO ────────────────
    st.markdown(f"<h3>⚙️ Configurações do Modelo</h3>", unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        kernel = st.selectbox(
            "Kernel",
            options=["rbf", "linear", "poly"],
            help="O kernel define como o SVM mapeia os dados para separar as classes. "
                 "RBF é o mais versátil para dados biomédicos.",
        )
    with col_cfg2:
        C = st.slider(
            "Parâmetro C (regularização)",
            min_value=0.01, max_value=10.0, value=1.0, step=0.01,
            help="Controla o trade-off entre maximizar a margem e minimizar os erros de "
                 "classificação. Valores altos = menos tolerância a erros.",
        )
    with col_cfg3:
        test_size = st.slider(
            "Proporção do conjunto de teste",
            min_value=0.10, max_value=0.40, value=0.20, step=0.05,
            help="Fração dos dados reservada para avaliação do modelo.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TREINO DO MODELO ───────────────────────
    # Sempre usa o dataset completo para treino (não o filtrado pelo sidebar)
    X = df[FEATURES].values
    y = df["diagnóstico"].values   # 0 = Maligno, 1 = Benigno

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modelo = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    modelo.fit(X_train_sc, y_train)

    y_pred      = modelo.predict(X_test_sc)
    y_prob      = modelo.predict_proba(X_test_sc)[:, 1]  # prob. de ser Benigno
    report      = classification_report(
        y_test, y_pred,
        target_names=["Maligno", "Benigno"],
        output_dict=True,
    )
    cm          = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    acuracia   = report["accuracy"]
    precisao_m = report["Maligno"]["precision"]
    recall_m   = report["Maligno"]["recall"]
    f1_m       = report["Maligno"]["f1-score"]
    precisao_b = report["Benigno"]["precision"]
    recall_b   = report["Benigno"]["recall"]
    f1_b       = report["Benigno"]["f1-score"]

    # ── MÉTRICAS PRINCIPAIS ────────────────────
    st.markdown(f"<h3>📈 Métricas de Desempenho</h3>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Acurácia",  f"{acuracia*100:.1f}%")
    m2.metric("AUC-ROC",   f"{roc_auc:.3f}")
    m3.metric("Recall · Maligno",  f"{recall_m*100:.1f}%",
              help="Dos casos realmente malignos, quantos o modelo identificou corretamente.")
    m4.metric("F1-Score · Maligno", f"{f1_m:.3f}",
              help="Média harmônica entre precisão e recall para a classe Maligno.")

    svmbox(f"""
    <b>Como interpretar:</b> O modelo foi treinado com <b>{len(X_train)}</b> amostras e avaliado
    em <b>{len(X_test)}</b>. A acurácia de <b>{acuracia*100:.1f}%</b> indica a proporção de
    diagnósticos corretos no conjunto de teste. Em contexto clínico, o <b>Recall do Maligno
    ({recall_m*100:.1f}%)</b> é a métrica mais crítica: ele mede quantos casos malignos o
    modelo conseguiu detectar — erros de falso negativo (maligno classificado como benigno)
    têm consequências mais graves do que falsos positivos.
    """)

    st.divider()

    # ── GRÁFICOS LADO A LADO ───────────────────
    col_cm, col_roc = st.columns(2)

    # Matriz de Confusão
    with col_cm:
        titulo("Matriz de Confusão")

        labels = ["Maligno", "Benigno"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        text_matrix = [
            [f"{cm[i][j]}<br><span style='font-size:11px'>({cm_pct[i][j]:.1f}%)</span>"
             for j in range(2)]
            for i in range(2)
        ]

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred. Maligno", "Pred. Benigno"],
            y=["Real Maligno",  "Real Benigno"],
            text=text_matrix,
            texttemplate="%{text}",
            colorscale=[
                [0.0, BG_PLOT],
                [0.5, "#8b2252"],
                [1.0, "#c0396e"],
            ],
            showscale=False,
            hovertemplate="Real: %{y}<br>Predito: %{x}<br>Contagem: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
            height=340,
            xaxis=dict(tickfont_color=COR_TEXTO, gridcolor=COR_GRID),
            yaxis=dict(tickfont_color=COR_TEXTO, gridcolor=COR_GRID, autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        infobox("Diagonal principal = acertos. "
                "Célula superior direita = falsos negativos (maligno classificado como benigno) "
                "— o erro mais crítico em oncologia.")

    # Curva ROC
    with col_roc:
        titulo(f"Curva ROC  ·  AUC = {roc_auc:.3f}")

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"SVM ({kernel.upper()})  AUC={roc_auc:.3f}",
            line=dict(color="#c0396e", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(192,57,110,0.12)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Classificador aleatório",
            line=dict(color=COR_CAPTION, width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_PLOT,
    font=dict(family="Lato", color=COR_TEXTO, size=13),
    xaxis=dict(gridcolor=COR_GRID, showline=True, linecolor=COR_BORDA,
               title="Taxa de Falsos Positivos"),
    yaxis=dict(gridcolor=COR_GRID, showline=True, linecolor=COR_BORDA,
               title="Taxa de Verdadeiros Positivos"),
    legend=dict(bgcolor=BG_CARD, bordercolor=COR_BORDA, borderwidth=1,
                x=0.55, y=0.08),
    margin=dict(l=10, r=10, t=40, b=10),
    height=340,
)
    st.plotly_chart(fig_roc, use_container_width=True)
    infobox("A curva ROC mede a capacidade discriminativa do modelo em todos os limiares "
            "de decisão. AUC próximo de 1.0 indica separação quase perfeita entre as classes.")

    st.divider()

    # ── IMPORTÂNCIA DAS FEATURES (coef. linear) ─
    titulo("Importância das Features")

    if kernel == "linear":
        importancias = np.abs(modelo.coef_[0])
        feat_imp = pd.Series(importancias, index=FEATURES).sort_values(ascending=True).tail(15)
        fig_imp = px.bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation="h",
            color=feat_imp.values,
            color_continuous_scale=["#5b9bd5", "#c0396e"],
            labels={"x": "|Coeficiente|", "y": "Feature"},
        )
        fig_imp.update_layout(**LAYOUT_BASE, height=420, showlegend=False,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        infobox("Com kernel linear, os coeficientes do hiperplano revelam diretamente o peso "
                "de cada feature na decisão. Features com maior valor absoluto têm maior "
                "poder discriminativo.")
    else:
        svmbox("""
        <b>Importância de features com kernels não-lineares</b><br><br>
        Com kernels <b>RBF</b> ou <b>Poly</b>, o SVM opera em um espaço dimensional implícito —
        não é possível extrair coeficientes diretos como no kernel linear. Para obter importâncias
        interpretáveis com esses kernels, seria necessário aplicar técnicas complementares como
        <b>SHAP values</b> ou <b>Permutation Importance</b>.<br><br>
        Selecione o kernel <b>Linear</b> para visualizar a contribuição de cada feature
        diretamente a partir dos coeficientes do hiperplano separador.
        """)

    st.divider()

    # ── TABELA DE MÉTRICAS DETALHADAS ──────────
    titulo("Relatório Detalhado por Classe")

    tabela = pd.DataFrame({
        "Classe":    ["Maligno", "Benigno"],
        "Precisão":  [f"{precisao_m*100:.1f}%", f"{precisao_b*100:.1f}%"],
        "Recall":    [f"{recall_m*100:.1f}%",   f"{recall_b*100:.1f}%"],
        "F1-Score":  [f"{f1_m:.3f}",            f"{f1_b:.3f}"],
        "Suporte":   [int(report["Maligno"]["support"]),
                      int(report["Benigno"]["support"])],
    })

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True,
    )

    svmbox("""
    <b>Glossário das métricas:</b><br>
    <b>Precisão</b> — dos casos classificados como X, quantos realmente são X.<br>
    <b>Recall</b> — dos casos que realmente são X, quantos foram corretamente identificados.<br>
    <b>F1-Score</b> — média harmônica entre precisão e recall; útil quando há desbalanceamento
    entre as classes.<br>
    <b>Suporte</b> — número de amostras reais de cada classe no conjunto de teste.
    """)

# ──────────────────────────────────────────────
# RODAPÉ
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='text-align:center;font-size:0.85rem;color:{COR_CAPTION};'>"
    "🎗️ Dataset Wisconsin Breast Cancer | Análise exploratória e classificação SVM | "
    "Streamlit + Plotly + Scikit-learn</p>",
    unsafe_allow_html=True,
)