"""
Sistema de Avaliação de Risco de Obesidade
Ferramenta de Apoio à Decisão Clínica
Modelo: LightGBM
Tech Challenge 4 - FIAP
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Sistema de Avaliação de Risco de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para aparência profissional
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #E8F4FD 0%, #B8D4E8 100%);
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    /* Cards de métricas */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E3A5F;
        margin: 0.3rem 0;
    }
    
    /* Resultados - tamanho reduzido */
    .result-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .result-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .result-danger {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    /* Menu lateral */
    .menu-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #1E3A5F;
    }
    
    /* Botões */
    .stButton>button {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        color: white;
        font-size: 1rem;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 60, 95, 0.3);
    }
    
    /* Info cards para dados do paciente - tamanho reduzido */
    .patient-info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        text-align: center;
    }
    .patient-info-card h4 {
        color: #6c757d;
        font-size: 0.7rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .patient-info-card h2 {
        color: #1E3A5F;
        font-size: 1.3rem;
        margin: 0.3rem 0 0 0;
        font-weight: 700;
    }

    /* Mudar cor do radio button selecionado */
    .stRadio > div > label > div:first-child {
        background-color: #1E3A5F !important;
    }
    
    /* Seções */
    .section-divider {
        border-top: 1px solid #dee2e6;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Carregar modelo e componentes
@st.cache_resource
def load_model():
    """Carrega o modelo e componentes salvos"""
    try:
        with open('model_components.pkl', 'rb') as f:
            components = pickle.load(f)
        return components
    except FileNotFoundError:
        st.error("⚠️ Arquivo do modelo não encontrado. Execute 'python train_model_exact.py' primeiro.")
        return None

# Função de previsão
def fazer_previsao(dados_paciente, components):
    """
    Realiza a previsão usando EXATAMENTE o mesmo preprocessamento do notebook.
    """
    model = components['model']
    le_target = components['le_target']
    ordinal_encoder = components['ordinal_encoder']
    minmax_scaler = components['minmax_scaler']
    minmax_features = components['minmax_features']
    feature_columns = components['feature_columns']
    
    # Criar DataFrame com os dados do paciente
    test_sample = pd.DataFrame({
        'Age': [dados_paciente['age']],
        'Gender': [dados_paciente['gender']],
        'Height': [dados_paciente['height']],
        'Weight': [dados_paciente['weight']],
        'family_history': [dados_paciente['family_history']],
        'FAVC': [dados_paciente['favc']],
        'FCVC': [dados_paciente['fcvc']],
        'NCP': [dados_paciente['ncp']],
        'CAEC': [dados_paciente['caec']],
        'SMOKE': [dados_paciente['smoke']],
        'CH2O': [dados_paciente['ch2o']],
        'SCC': [dados_paciente['scc']],
        'FAF': [dados_paciente['faf']],
        'TUE': [dados_paciente['tue']],
        'CALC': [dados_paciente['calc']],
        'MTRANS': [dados_paciente['mtrans']]
    })
    
    # Aplicar One-Hot Encoding
    test_processed = pd.get_dummies(
        test_sample, 
        columns=['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC', 'MTRANS'], 
        drop_first=False
    )
    
    # Adicionar colunas faltantes com valor 0
    for col in feature_columns:
        if col not in test_processed.columns:
            test_processed[col] = 0
    
    # Reordenar colunas para corresponder ao conjunto de treino
    test_processed = test_processed[feature_columns]
    
    # Aplicar Ordinal Encoding
    test_processed[['CAEC', 'CALC']] = ordinal_encoder.transform(
        test_processed[['CAEC', 'CALC']]
    ).astype(int)
    
    # Aplicar MinMax Scaling
    test_processed[minmax_features] = minmax_scaler.transform(
        test_processed[minmax_features]
    )
    
    # Fazer predição
    pred_class = model.predict(test_processed)[0]
    pred_proba = model.predict_proba(test_processed)[0]
    
    return {
        'classe': le_target.classes_[pred_class],
        'confianca': pred_proba[pred_class],
        'probabilidades': dict(zip(le_target.classes_, pred_proba)),
        'classes': le_target.classes_
    }

# Mapeamentos para interface em português
CLASSES_PT = {
    'Insufficient_Weight': 'Peso Insuficiente',
    'Normal_Weight': 'Peso Normal',
    'Obesity_Type_I': 'Obesidade Grau I',
    'Obesity_Type_II': 'Obesidade Grau II',
    'Obesity_Type_III': 'Obesidade Grau III (Mórbida)',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II'
}

CLASSES_DESCRICAO = {
    'Insufficient_Weight': 'IMC abaixo de 18.5 kg/m²',
    'Normal_Weight': 'IMC entre 18.5 e 24.9 kg/m²',
    'Obesity_Type_I': 'IMC entre 30.0 e 34.9 kg/m²',
    'Obesity_Type_II': 'IMC entre 35.0 e 39.9 kg/m²',
    'Obesity_Type_III': 'IMC acima de 40.0 kg/m²',
    'Overweight_Level_I': 'IMC entre 25.0 e 27.4 kg/m²',
    'Overweight_Level_II': 'IMC entre 27.5 e 29.9 kg/m²'
}

NIVEL_RISCO = {
    'Insufficient_Weight': ('Atenção Nutricional', 'warning'),
    'Normal_Weight': ('Baixo Risco', 'success'),
    'Obesity_Type_I': ('Risco Elevado', 'danger'),
    'Obesity_Type_II': ('Risco Alto', 'danger'),
    'Obesity_Type_III': ('Risco Muito Alto', 'danger'),
    'Overweight_Level_I': ('Risco Moderado', 'warning'),
    'Overweight_Level_II': ('Risco Moderado', 'warning')
}

# Carregar modelo
components = load_model()

# Sidebar - Menu de Navegação
with st.sidebar:
    #st.image("https://truehealthpeds.com/wp-content/uploads/truehealth-favicon.png", width=115)
    st.markdown(    """
    <div style="display: flex; justify-content: center;">
        <img src="https://truehealthpeds.com/wp-content/uploads/truehealth-favicon.png" width="115">
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("### Sistema de Avaliação de Risco de Obesidade")
    
    st.markdown("---")
    
    # Menu de navegação claro
    st.markdown('<p class="menu-title">📌 NAVEGAÇÃO</p>', unsafe_allow_html=True)
    
    menu = st.radio(
        "Selecione uma opção:",
        ["📋 Informações do Sistema", "🔬 Realizar Avaliação"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    # Informações do sistema (compactas)
    st.markdown("**Versão:** 1.0.0")
    st.markdown("**Atualização:** Janeiro 2026")
    st.markdown("---")
    st.caption("Tech Challenge 4 - FIAP")
    st.caption("Pós-Graduação em Data Analytics")

# Página de Informações
if menu == "📋 Informações do Sistema":
    st.markdown('<div class="main-header">Sistema de Avaliação de Risco de Obesidade</div>', unsafe_allow_html=True)
    
    # Sobre o Sistema
    st.markdown("## Sobre o Sistema")
    
    st.markdown("""
    Este sistema foi desenvolvido para auxiliar **equipes médicas e profissionais de saúde** na 
    avaliação do nível de risco de obesidade de pacientes. A ferramenta utiliza técnicas avançadas 
    de análise de dados para analisar múltiplos fatores e fornecer uma classificação precisa, 
    apoiando a tomada de decisão clínica.
    """)
    
    # Benefícios
    st.markdown("### 🎯 Benefícios para a Equipe Médica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Agilidade no Diagnóstico**
        - Avaliação rápida e padronizada
        - Resultados em segundos
        - Interface intuitiva
        
        **Precisão na Classificação**
        - Modelo com 96.7% de acurácia
        - Análise de 16 variáveis
        - Classificação em 7 níveis de risco
        """)
    
    with col2:
        st.markdown("""
        **Apoio à Decisão Clínica**
        - Probabilidades detalhadas por classe
        - Recomendações baseadas no resultado
        - Histórico de avaliações
        
        **Padronização do Atendimento**
        - Critérios objetivos de avaliação
        - Protocolo consistente
        - Documentação automática
        """)
    
    st.markdown("---")
    
    # Classificações de Risco
    st.markdown("### 📊 Classificações de Risco")
    
    st.markdown("""
    O sistema classifica os pacientes em **7 níveis de risco**, permitindo uma abordagem 
    personalizada para cada caso:
    """)
    
    classes_df = pd.DataFrame({
        'Classificação': list(CLASSES_PT.values()),
        'Faixa de IMC': list(CLASSES_DESCRICAO.values()),
        'Nível de Risco': [NIVEL_RISCO[k][0] for k in CLASSES_PT.keys()]
    })
    st.table(classes_df)
    
    st.markdown("---")
    
    # Fatores Analisados
    st.markdown("### 📋 Fatores Analisados na Avaliação")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Perfil do Paciente**
        - Gênero
        - Idade
        - Altura
        - Peso
        - Histórico familiar
        """)
    
    with col2:
        st.markdown("""
        **Hábitos Alimentares**
        - Consumo de alimentos calóricos
        - Frequência de vegetais
        - Número de refeições
        - Alimentação entre refeições
        - Consumo de água
        - Consumo de álcool
        """)
    
    with col3:
        st.markdown("""
        **Estilo de Vida**
        - Tabagismo
        - Monitoramento calórico
        - Atividade física
        - Tempo em telas
        - Meio de transporte
        """)
    
    st.markdown("---")
    
    # Equipe de Desenvolvimento
    st.markdown("### 👥 Desenvolvimento")
    
    st.markdown("""
    Este sistema foi desenvolvido como parte do **Tech Challenge 4** do programa de 
    **Pós-Graduação em Data Analytics** da **FIAP**.
    
    O projeto aplica técnicas avançadas de Machine Learning para criar uma solução 
    prática e eficiente para o ambiente clínico.
    """)
    
    st.markdown("#### Desenvolvedores")
    st.markdown("""
    - **Bianca Correa** - bianca.correa@fiap.com.br
    - **Daniele Andrino** - daniele.andrino@fiap.com.br
    """)
    
    st.markdown("---")
    
    # Aviso Legal
    st.warning("""
    **⚠️ Aviso Importante**
    
    Este sistema é uma **ferramenta de apoio à decisão clínica** e não substitui a avaliação 
    médica profissional. Os resultados devem ser interpretados por profissionais de saúde 
    qualificados, considerando o contexto clínico completo do paciente.
    """)

# Página de Previsão
elif menu == "🔬 Realizar Avaliação":
    st.markdown('<div class="main-header">🔬 Avaliação de Risco de Obesidade</div>', unsafe_allow_html=True)
    
    if components is None:
        st.error("⚠️ Sistema indisponível. Entre em contato com o suporte técnico.")
    else:
        st.markdown("Preencha os dados do paciente para realizar a avaliação de risco.")
        
        with st.form("prediction_form"):
            st.markdown("### 📝 Dados do Paciente")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Perfil")
                gender = st.selectbox("Gênero", ["Female", "Male"], format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
                age = st.number_input("Idade (anos)", min_value=10, max_value=100, value=30)
                height = st.number_input("Altura (metros)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
                weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
                family_history = st.selectbox("Histórico familiar de sobrepeso", ["yes", "no"], format_func=lambda x: "Sim" if x == "yes" else "Não")
            
            with col2:
                st.markdown("#### 🍽️ Alimentação")
                favc = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"], format_func=lambda x: "Sim" if x == "yes" else "Não")
                fcvc = st.slider("Frequência de consumo de vegetais", min_value=1.0, max_value=3.0, value=2.0, step=0.5, help="1 = Raramente, 3 = Sempre")
                ncp = st.slider("Número de refeições principais", min_value=1.0, max_value=4.0, value=3.0, step=0.5)
                caec = st.selectbox("Consumo de alimentos entre refeições", 
                                   ["no", "Sometimes", "Frequently", "Always"],
                                   format_func=lambda x: {"no": "Não", "Sometimes": "Às vezes", "Frequently": "Frequentemente", "Always": "Sempre"}[x])
                ch2o = st.slider("Consumo diário de água (litros)", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
                calc = st.selectbox("Consumo de bebidas alcoólicas", 
                                   ["no", "Sometimes", "Frequently", "Always"],
                                   format_func=lambda x: {"no": "Não", "Sometimes": "Às vezes", "Frequently": "Frequentemente", "Always": "Sempre"}[x])
            
            with col3:
                st.markdown("#### 🏃 Estilo de Vida")
                smoke = st.selectbox("Fumante", ["no", "yes"], format_func=lambda x: "Sim" if x == "yes" else "Não")
                scc = st.selectbox("Monitora calorias consumidas", ["no", "yes"], format_func=lambda x: "Sim" if x == "yes" else "Não")
                faf = st.slider("Atividade física (dias/semana)", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
                tue = st.slider("Tempo em telas (horas/dia)", min_value=0.0, max_value=2.0, value=1.0, step=0.5)
                mtrans = st.selectbox("Principal meio de transporte", 
                                     ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
                                     format_func=lambda x: {"Public_Transportation": "Transporte Público", 
                                                           "Automobile": "Automóvel", 
                                                           "Walking": "Caminhada", 
                                                           "Motorbike": "Motocicleta", 
                                                           "Bike": "Bicicleta"}[x])
            
            submitted = st.form_submit_button("🔍 Realizar Avaliação", use_container_width=True)
        
        if submitted:
            # Preparar dados para previsão
            dados_paciente = {
                'gender': gender,
                'age': age,
                'height': height,
                'weight': weight,
                'family_history': family_history,
                'favc': favc,
                'fcvc': fcvc,
                'ncp': ncp,
                'caec': caec,
                'smoke': smoke,
                'ch2o': ch2o,
                'scc': scc,
                'faf': faf,
                'tue': tue,
                'calc': calc,
                'mtrans': mtrans
            }
            
            # Fazer previsão
            resultado = fazer_previsao(dados_paciente, components)
            
            # Calcular IMC
            imc = weight / (height ** 2)
            
            st.markdown("---")
            
            # Resultado Principal
            st.markdown("### 📊 Resultado da Avaliação")
            
            classe_pt = CLASSES_PT[resultado['classe']]
            descricao = CLASSES_DESCRICAO[resultado['classe']]
            risco, tipo_risco = NIVEL_RISCO[resultado['classe']]
            
            # Layout do resultado - compacto e alinhado
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; align-items: stretch;">
                <div class="result-{tipo_risco}" style="flex: 3;">
                    <h3 style="margin:0; color: #1E3A5F; font-size: 1.4rem;">🎯 {classe_pt}</h3>
                    <p style="font-size: 0.95rem; margin: 0.3rem 0; color: #495057;">{descricao}</p>
                    <p style="font-size: 1rem; margin: 0.5rem 0 0 0;"><strong>Nível de Risco:</strong> {risco}</p>
                    <p style="font-size: 0.95rem; margin: 0.3rem 0 0 0;"><strong>Confiança da Avaliação:</strong> {resultado['confianca']*100:.2f}%</p>
                </div>
                <div style="flex: 2; display: flex; flex-direction: column; gap: 0.4rem;">
                    <p style="font-weight: 600; color: #1E3A5F; margin: 0; font-size: 0.9rem;">Dados Antropométricos</p>
                    <div class="patient-info-card">
                        <h4>Índice de Massa Corporal (IMC)</h4>
                        <h2>{imc:.1f} kg/m²</h2>
                    </div>
                    <div style="display: flex; gap: 0.4rem;">
                        <div class="patient-info-card" style="flex: 1;">
                            <h4>Peso</h4>
                            <h2>{weight:.1f} kg</h2>
                        </div>
                        <div class="patient-info-card" style="flex: 1;">
                            <h4>Altura</h4>
                            <h2>{height:.2f} m</h2>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Gráfico de probabilidades
            st.markdown("### 📈 Distribuição de Probabilidades")
            
            probs = resultado['probabilidades']
            classes_ordenadas = sorted(probs.keys(), key=lambda x: probs[x], reverse=True)
            
            # Cores para o gráfico
            cores = ['#28a745' if c == resultado['classe'] else '#1E3A5F' for c in classes_ordenadas]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[probs[c]*100 for c in classes_ordenadas],
                    y=[CLASSES_PT[c] for c in classes_ordenadas],
                    orientation='h',
                    marker_color=cores,
                    text=[f"{probs[c]*100:.2f}%" for c in classes_ordenadas],
                    textposition='outside',
                    textfont=dict(size=11)
                )
            ])
            
            fig.update_layout(
                title="",
                xaxis_title="Probabilidade (%)",
                yaxis_title="",
                height=300,
                showlegend=False,
                xaxis=dict(range=[0, 100]),
                margin=dict(l=0, r=50, t=10, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Recomendações Clínicas
            st.markdown("### 💡 Recomendações Clínicas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if resultado['classe'] in ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']:
                    st.markdown("""
                    **Condutas Recomendadas:**
                    - Encaminhamento para avaliação nutricional especializada
                    - Avaliação endocrinológica
                    - Investigação de comorbidades (diabetes, hipertensão, dislipidemia)
                    - Programa de exercícios supervisionados
                    - Avaliação para tratamento medicamentoso ou cirúrgico
                    """)
                elif resultado['classe'] in ['Overweight_Level_I', 'Overweight_Level_II']:
                    st.markdown("""
                    **Condutas Recomendadas:**
                    - Orientação nutricional
                    - Programa de atividade física
                    - Avaliação de fatores de risco cardiovascular
                    - Monitoramento de glicemia e perfil lipídico
                    - Estabelecimento de metas de perda de peso
                    """)
                elif resultado['classe'] == 'Insufficient_Weight':
                    st.markdown("""
                    **Condutas Recomendadas:**
                    - Investigação de causas do baixo peso
                    - Avaliação nutricional
                    - Investigação de distúrbios alimentares
                    - Monitoramento do estado nutricional
                    - Suplementação se necessário
                    """)
                else:
                    st.markdown("""
                    **Condutas Recomendadas:**
                    - Manutenção de hábitos alimentares saudáveis
                    - Prática regular de atividade física
                    - Check-ups preventivos regulares
                    - Monitoramento periódico do peso
                    """)
            
            with col2:
                # Classificação do IMC
                if imc < 18.5:
                    imc_class = "Abaixo do peso"
                    imc_cor = "#3498db"
                elif imc < 25:
                    imc_class = "Peso normal"
                    imc_cor = "#27ae60"
                elif imc < 30:
                    imc_class = "Sobrepeso"
                    imc_cor = "#f39c12"
                elif imc < 35:
                    imc_class = "Obesidade Grau I"
                    imc_cor = "#e67e22"
                elif imc < 40:
                    imc_class = "Obesidade Grau II"
                    imc_cor = "#e74c3c"
                else:
                    imc_class = "Obesidade Grau III"
                    imc_cor = "#c0392b"
                
                st.markdown(f"""
                **Classificação do IMC:** {imc_class}
                
                **Data da avaliação:** {datetime.now().strftime("%d/%m/%Y às %H:%M")}
                
                **Observações:**
                - O IMC é um indicador auxiliar
                - A classificação considera múltiplos fatores
                - Avaliação clínica completa é necessária
                """)
            
            st.markdown("---")
            
            # Aviso
            st.info("""
            **📋 Nota:** Esta avaliação é uma ferramenta de apoio à decisão clínica. 
            O diagnóstico final e o plano de tratamento devem ser definidos pelo médico 
            responsável, considerando a avaliação clínica completa do paciente.
            """)

# Footer
st.markdown("""
<div class="footer">
    © 2026 - Sistema de Avaliação de Risco de Obesidade | Tech Challenge 4 - FIAP<br>
    <small>Pós-Graduação em Data Analytics</small>
</div>
""", unsafe_allow_html=True)
