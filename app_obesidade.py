import streamlit as st
import pandas as pd
import joblib

# Carregar artefatos
modelo = joblib.load("modelo_obesidade.pkl")
label_encoder = joblib.load("label_encoder.pkl")
colunas_modelo = joblib.load("colunas_modelo.pkl")

st.title("🔍 Predição de Nível de Obesidade")

genero = st.selectbox("Gênero", ["Male", "Female"])
idade = st.slider("Idade", 10, 100, 25)
altura = st.slider("Altura (em metros)", 1.0, 2.2, 1.70, step=0.01)
peso = st.slider("Peso (em kg)", 30, 200, 70, step=1)
hist_familiar = st.selectbox("Histórico Familiar de Obesidade", ["yes", "no"])
freq_alimentos_caloricos = st.selectbox("Frequência de Alimentos Calóricos", ["no", "sometimes", "frequently"])
freq_consumo_vegetais = st.slider("Frequência de Vegetais (0-3)", 0.0, 3.0, 2.0, step=0.1)
qtd_refeicoes_dia = st.slider("Refeições por Dia", 1.0, 5.0, 3.0, step=0.5)
alimentos_entre_refeicoes = st.selectbox("Come entre Refeições?", ["no", "sometimes", "frequently", "always"])
flag_fumo = st.selectbox("Fumante?", ["yes", "no"])
qtd_dia_agua = st.slider("Litros de Água por Dia", 0.0, 4.0, 2.0, step=0.1)
freq_ativ_fisica = st.slider("Atividade Física (0-3)", 0.0, 3.0, 1.0, step=0.1)
freq_uso_tech = st.slider("Horas com Tecnologia/dia", 0.0, 16.0, 4.0, step=1.0)
freq_alcool = st.selectbox("Consumo de Álcool", ["no", "sometimes", "frequently"])
meio_transporte = st.selectbox("Meio de Transporte", ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"])

if st.button("📊 Estimar Nível de Obesidade"):
    entrada_dict = {
        'genero': genero,
        'idade': idade,
        'altura': altura,
        'peso': peso,
        'hist_familiar': hist_familiar,
        'freq_alimentos_caloricos': freq_alimentos_caloricos,
        'freq_consumo_vegetais': freq_consumo_vegetais,
        'qtd_refeicoes_dia': qtd_refeicoes_dia,
        'alimentos_entre_refeicoes': alimentos_entre_refeicoes,
        'flag_fumo': flag_fumo,
        'qtd_dia_agua': qtd_dia_agua,
        'freq_ativ_fisica': freq_ativ_fisica,
        'freq_uso_tech': freq_uso_tech,
        'freq_alcool': freq_alcool,
        'meio_transporte': meio_transporte
    }

    entrada_df = pd.DataFrame([entrada_dict])
    entrada_dummies = pd.get_dummies(entrada_df)

    for col in colunas_modelo:
        if col not in entrada_dummies.columns:
            entrada_dummies[col] = 0

    entrada_dummies = entrada_dummies[colunas_modelo]

    pred = modelo.predict(entrada_dummies)[0]
    nivel_obesidade = label_encoder.inverse_transform([pred])[0]

    st.subheader("🧠 Resultado:")
    st.write(f"**Nível estimado de obesidade:** `{nivel_obesidade}`")

    if "Obesity" in nivel_obesidade:
        st.warning("⚠️ Risco elevado! Recomenda-se acompanhamento médico.")
    elif "Overweight" in nivel_obesidade:
        st.info("ℹ️ Acima do peso ideal. Avaliação nutricional pode ser indicada.")
    else:
        st.success("✅ Peso dentro dos níveis considerados normais.")
