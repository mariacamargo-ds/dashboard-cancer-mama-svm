# 🎗️ Análise Exploratória e Dashboard – Diagnóstico de Câncer de Mama

Análise exploratória e classificação supervisionada aplicadas ao
Dataset Wisconsin Breast Cancer, com dashboard interativo desenvolvido
em Streamlit.

---

## 📌 Sobre o Projeto

O dashboard explora 30 biomarcadores morfológicos extraídos de biópsias
por agulha fina (FNA), com foco na diferenciação entre tumores malignos
e benignos. Inclui visualizações interativas e um classificador SVM
com curva ROC, matriz de confusão e análise de importância de features.

## 🎯 Objetivos

O projeto tem como objetivo geral integrar técnicas de análise exploratória de dados, 
desenvolvimento de interfaces interativas e classificação supervisionada para 
investigar os biomarcadores tumorais presentes no dataset Wisconsin Breast Cancer, 
com foco na diferenciação entre casos malignos e benignos.

De forma mais específica, os objetivos são:
- Conduzir uma análise exploratória estruturada, identificando padrões de distribuição
  e correlação entre as features morfológicas;
- Desenvolver visualizações estáticas e interativas que comuniquem os achados de forma clara e acessível;
- Implementar um dashboard em Streamlit com suporte a filtros dinâmicos e modo escuro, mantendo
  coerência visual com o painel original elaborado no Power BI;
- Aplicar um classificador SVM ao dataset, avaliando seu desempenho por meio de métricas clínicas
  relevantes e integrando os resultados ao dashboard;
- Articular os resultados da EDA e da classificação com as visualizações do dashboard, produzindo
  uma narrativa analítica integrada sobre os dados.


---

## 🛠️ Tecnologias

- Python 3.11
- Streamlit
- Plotly
- Pandas / NumPy
- scikit-learn (SVM, StandardScaler, métricas)
- openpyxl

---

## 📁 Estrutura do Projeto

├── dash_cancer_mama.py        # código principal do dashboard
├── dataset_cancer_mama_02.xlsx  # dataset Wisconsin Breast Cancer
├── requirements.txt           # dependências do projeto
└── README.md
---

## ▶️ Como Rodar Localmente

```bash
# Clone o repositório
git clone https://github.com/SEU_USUARIO/dashboard-cancer-mama-svm.git

# Acesse a pasta
cd dashboard-cancer-mama-svm

# Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Rode o dashboard
streamlit run dash_cancer_mama.py
```

---

## 🌐 Deploy

Acesse o dashboard em produção:
[https://seu-usuario-dashboard.streamlit.app](https://seu-usuario-dashboard.streamlit.app)

---

## 👩‍💻 Autora

Maria Eduarda da Cruz de Camargo
[LinkedIn](https://linkedin.com/in/seu-perfil) · [GitHub](https://github.com/SEU_USUARIO)
