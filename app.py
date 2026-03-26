# ============================================
# 1. IMPORTAÇÕES E CONFIGURAÇÃO INICIAL
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import requests  # apenas se for baixar modelos de URL

# Configuração da página
st.set_page_config(page_title="Monitor de Risco de Falhas", layout="wide")

# ============================================
# 2. CARREGAMENTO DE MODELOS E ARTEFATOS
# ============================================
@st.cache_resource
def load_models():
    # Se os modelos estiverem locais (como antes)
    rf = joblib.load('random_forest_model.pkl')
    lgbm = joblib.load('lightgbm_model.pkl')
    threshold = joblib.load('best_threshold.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return rf, lgbm, threshold, feature_cols

rf_model, lgbm_model, best_threshold, feature_cols = load_models()

# ============================================
# 3. CARREGAMENTO DOS DADOS (teste ou histórico)
# ============================================
@st.cache_data
def load_data():
    df = pd.read_parquet('challenge_test.parquet')
    df = df.sort_values(['entity_id', 'day'])
    return df

data = load_data()

# ============================================
# 4. FUNÇÃO DE PRÉ-PROCESSAMENTO
# ============================================
def preprocess(df):
    df = df.copy()
    # 1. Preencher missing (forward fill, backward fill, mediana)
    df = df.groupby('entity_id').apply(lambda x: x.ffill()).reset_index(drop=True)
    df = df.groupby('entity_id').apply(lambda x: x.bfill()).reset_index(drop=True)
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].median())
    # 2. Interação idade
    for col in df.columns:
        if 'sensor' in col:
            df[f'{col}_age'] = df[col] * df['equipment_age']
    # 3. Ampliação instabilidade
    for col in df.columns:
        if '_cv_' in col:
            df[f'{col}_sq'] = df[col] ** 2
    # 4. Tendência longa
    for col in df.columns:
        if 'sensor' in col:
            df[f'{col}_trend_7'] = df.groupby('entity_id')[col].diff(7)
    # 5. Estatísticas por máquina
    for col in df.columns:
        if 'sensor' in col:
            df[f'{col}_entity_mean'] = df.groupby('entity_id')[col].transform('mean')
    # 6. One-hot lifecycle_stage
    df = pd.get_dummies(df, columns=['lifecycle_stage'], drop_first=True)
    # 7. Remover colunas não-feature
    drop_cols = ['day', 'entity_id']
    if 'event' in df.columns:
        drop_cols.append('event')
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # 8. Garantir colunas de treino
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return df

# ============================================
# 5. FUNÇÃO AUXILIAR PARA PREDIÇÕES EM MASSA (CACHE)
# ============================================
@st.cache_data
def get_all_predictions(df):
    X = preprocess(df)
    rf_probs = rf_model.predict_proba(X)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X)[:, 1]
    return (rf_probs + lgbm_probs) / 2

# ============================================
# 6. INTERFACE DO USUÁRIO (SIDEBAR)
# ============================================
st.title("🔧 Monitor de Risco de Falhas Industriais")
st.markdown("Previsão de eventos críticos em equipamentos usando sensores e dados operacionais")

# Sidebar com controles
st.sidebar.header("Configurações")
selected_entity = st.sidebar.selectbox("Selecione a máquina", sorted(data['entity_id'].unique()))
lookback_days = st.sidebar.slider("Dias para histórico", 1, 100, 7)

# ============================================
# 7. CÁLCULOS GLOBAIS (RANKING, ATENÇÃO, ETC)
# ============================================

# Predições para todos os registros (cacheado)
all_probs = get_all_predictions(data)
data_with_probs = data.copy()
data_with_probs['prob'] = all_probs

# Ranking das máquinas (último dia)
latest_per_machine = data_with_probs.sort_values('day').groupby('entity_id').last().reset_index()
risk_df = latest_per_machine[['entity_id', 'prob']].rename(columns={'prob': 'risk'})
risk_df = risk_df.sort_values('risk', ascending=False)
top5 = risk_df.head(5)

# Cálculo de tendência para lista de atenção (últimos 7 dias)
def calculate_slope(series):
    x = np.arange(len(series))
    if len(series) < 2:
        return 0
    slope = np.polyfit(x, series, 1)[0]
    return slope

attention_list = []
for entity in data['entity_id'].unique():
    entity_data = data_with_probs[data_with_probs['entity_id'] == entity].sort_values('day')
    if len(entity_data) < 2:
        continue
    # Pegar últimos 7 dias de probabilidades (já calculadas)
    probs_series = entity_data['prob'].values
    n = min(7, len(probs_series))
    recent_probs = probs_series[-n:]
    slope = calculate_slope(recent_probs)
    current_risk = probs_series[-1]
    attention_list.append({
        'entity_id': entity,
        'slope': slope,
        'current_risk': current_risk
    })
attention_df = pd.DataFrame(attention_list)
attention_df = attention_df[attention_df['slope'] > 0].sort_values('slope', ascending=False).head(5)

# ============================================
# 8. LAYOUT PRINCIPAL (COLUNAS SUPERIORES)
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📊 Máquina {selected_entity}")
    # Dados da máquina selecionada
    entity_data = data_with_probs[data_with_probs['entity_id'] == selected_entity].sort_values('day')
    if len(entity_data) > 0:
        last_row = entity_data.iloc[-1]
        current_prob = last_row['prob']
        status = "🔴 Alto risco" if current_prob >= best_threshold else "🟢 Baixo risco"
        st.metric("Probabilidade de falha", f"{current_prob:.2%}")
        st.metric("Status", status)
    else:
        st.warning("Nenhum dado para esta máquina.")

with col2:
    st.subheader("🏆 Top 5 máquinas em risco")
    # Gráfico de barras do ranking
    fig_rank, ax_rank = plt.subplots(figsize=(6,4))
    ax_rank.barh(top5['entity_id'].astype(str), top5['risk'], color='red')
    ax_rank.set_xlabel('Probabilidade')
    ax_rank.set_title('Top 5 máquinas')
    st.pyplot(fig_rank)

# ============================================
# 9. GRÁFICO DE EVOLUÇÃO DO RISCO (COM lookback_days)
# ============================================
st.subheader(f"📈 Evolução do risco - Máquina {selected_entity}")
entity_plot = data_with_probs[data_with_probs['entity_id'] == selected_entity].sort_values('day')
if len(entity_plot) > lookback_days:
    entity_plot = entity_plot.tail(lookback_days)

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(entity_plot['day'], entity_plot['prob'], marker='o', linestyle='-', color='orange')
ax2.axhline(y=best_threshold, color='r', linestyle='--', label='Limiar')
ax2.set_xlabel('Dia')
ax2.set_ylabel('Probabilidade de falha')
ax2.set_title(f'Evolução do risco - Máquina {selected_entity} (últimos {lookback_days} dias)')
ax2.legend()
st.pyplot(fig2)

# ============================================
# 10. LISTA PRÉ DE ATENÇÃO (TENDÊNCIA CRESCENTE)
# ============================================
st.subheader("⚠️ Máquinas em atenção (tendência crescente)")
if not attention_df.empty:
    st.dataframe(
        attention_df[['entity_id', 'slope', 'current_risk']].style.format({
            'slope': '{:.4f}',
            'current_risk': '{:.2%}'
        })
    )
else:
    st.write("Nenhuma máquina com tendência crescente nos últimos dias.")

# ============================================
# 11. FATORES DE FALHA (SHAP) - OPCIONAL
# ============================================
# Apenas se quiser exibir para a máquina selecionada
st.subheader("🔍 Principais fatores para o risco atual")
# Obter a última observação da máquina selecionada
last_record = data[data['entity_id'] == selected_entity].sort_values('day').iloc[[-1]]
last_prepared = preprocess(last_record)

# SHAP explicação
try:
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(last_prepared)
    # Extrair valores da classe positiva conforme formato
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1][0]
    else:
        if shap_values.ndim == 2:
            shap_values_pos = shap_values[0]
        elif shap_values.ndim == 3:
            shap_values_pos = shap_values[0, :, 1]
        else:
            shap_values_pos = shap_values[0] if len(shap_values) > 0 else shap_values
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'contribution': shap_values_pos
    }).sort_values('contribution', key=abs, ascending=False)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Fatores que aumentam o risco**")
        top_pos = shap_df[shap_df['contribution'] > 0].head(10)
        st.dataframe(top_pos.style.format({'contribution': '{:.4f}'}))
    with col_b:
        st.markdown("**Fatores que reduzem o risco**")
        top_neg = shap_df[shap_df['contribution'] < 0].head(10)
        st.dataframe(top_neg.style.format({'contribution': '{:.4f}'}))
except Exception as e:
    st.warning(f"Não foi possível calcular SHAP: {e}")

# ============================================
# FIM
# ============================================