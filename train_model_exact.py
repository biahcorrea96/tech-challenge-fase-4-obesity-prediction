import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

print('=== TREINAMENTO DO MODELO LIGHTGBM ===\n')

# 1. Carregar dados
print('1. Carregando dados...')
df = pd.read_csv('Obesity.csv')
print(f'   Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas')

# 2. Separar features e target 
print('\n2. Separando features e target...')
# O arquivo CSV usa nomes de colunas diferentes do notebook
# Renomear para manter compatibilidade
df = df.rename(columns={
    'family_history_with_overweight': 'family_history',
    'NObeyesdad': 'Obesity'
})

X = df.drop('Obesity', axis=1)
y = df['Obesity']
print(f'   X (features): {X.shape}')
print(f'   y (target): {y.shape}')

# 3. Codificar variável alvo 
print('\n3. Codificando variável alvo...')
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f'   Classes: {list(le_target.classes_)}')

# 4. One-Hot Encoding 
print('\n4. Aplicando One-Hot Encoding...')
X_processed = pd.get_dummies(
    X,
    columns=['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC', 'MTRANS'],
    drop_first=False  # IMPORTANTE: Manter todas as colunas
)
print(f'   Features após One-Hot: {X_processed.shape[1]}')

# Salvar a ordem das colunas ANTES de qualquer transformação adicional
# Esta é a ordem que será usada para previsões
feature_columns = list(X_processed.columns)
print(f'   Colunas: {feature_columns}')

# 5. Ordinal Encoding para CAEC e CALC 
print('\n5. Aplicando Ordinal Encoding para CAEC e CALC...')
ordinal_encoder = OrdinalEncoder()
X_processed[['CAEC', 'CALC']] = ordinal_encoder.fit_transform(
    X_processed[['CAEC', 'CALC']]
).astype(int)
print(f'   Categorias CAEC: {ordinal_encoder.categories_[0]}')
print(f'   Categorias CALC: {ordinal_encoder.categories_[1]}')

# 6. MinMax Scaling (EXATAMENTE como no notebook)
print('\n6. Aplicando MinMax Scaling...')
minmax_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
minmax_scaler = MinMaxScaler()
X_processed[minmax_features] = minmax_scaler.fit_transform(X_processed[minmax_features])
print(f'   Features normalizadas: {minmax_features}')

# 7. Dividir dados (EXATAMENTE como no notebook: 80/20, random_state=42, stratify)
print('\n7. Dividindo dados (80% treino, 20% teste)...')
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f'   Treino: {X_train.shape[0]} amostras')
print(f'   Teste: {X_test.shape[0]} amostras')

# 8. Treinar modelo LightGBM 
print('\n8. Treinando modelo LightGBM...')
model = LGBMClassifier(
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# 9. Avaliar modelo
print('\n9. Avaliando modelo...')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'   Acurácia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)')

# 10. Testar com o cenário específico do notebook
print('\n10. Testando com cenário específico do notebook...')
test_sample = pd.DataFrame({
    'Age': [25],
    'Gender': ['Male'],
    'Height': [1.70],
    'Weight': [90],
    'family_history': ['yes'],
    'FAVC': ['yes'],
    'FCVC': [2.0],
    'NCP': [3],
    'CAEC': ['Sometimes'],
    'SMOKE': ['no'],
    'CH2O': [2.0],
    'SCC': ['no'],
    'FAF': [1.0],
    'TUE': [1.0],
    'CALC': ['Sometimes'],
    'MTRANS': ['Public_Transportation']
})

# Aplicar as mesmas transformações 
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
test_processed[['CAEC', 'CALC']] = ordinal_encoder.transform(test_processed[['CAEC', 'CALC']]).astype(int)

# Aplicar MinMax Scaling
test_processed[minmax_features] = minmax_scaler.transform(test_processed[minmax_features])

# Fazer predição
pred_class = model.predict(test_processed)[0]
pred_proba = model.predict_proba(test_processed)[0]

print(f'\n   Classe Predita: {le_target.classes_[pred_class]}')
print(f'   Confiança: {pred_proba[pred_class]*100:.2f}%')
print(f'\n   Probabilidades por classe:')
for i, class_name in enumerate(le_target.classes_):
    print(f'     {class_name:25s}: {pred_proba[i]*100:6.2f}%')

# 11. Salvar componentes
print('\n11. Salvando componentes do modelo...')
model_components = {
    'model': model,
    'le_target': le_target,
    'ordinal_encoder': ordinal_encoder,
    'minmax_scaler': minmax_scaler,
    'minmax_features': minmax_features,
    'feature_columns': feature_columns,
    'accuracy': accuracy
}

with open('model_components.pkl', 'wb') as f:
    pickle.dump(model_components, f)

print('   Arquivo salvo: model_components.pkl')

print('\n=== TREINAMENTO CONCLUÍDO ===')
print(f'\nResultado esperado do notebook:')
print(f'  Classe: Overweight_Level_II')
print(f'  Confiança: 88.94%')
print(f'\nResultado obtido:')
print(f'  Classe: {le_target.classes_[pred_class]}')
print(f'  Confiança: {pred_proba[pred_class]*100:.2f}%')
