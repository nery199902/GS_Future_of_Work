# 🚀 GS_Future_of_work  
## Classificação do Impacto da Inteligência Artificial com Rede Neural MLP

Projeto desenvolvido durante a **Global Solution – FIAP**, com foco na aplicação de **Redes Neurais Artificiais (MLP)** para prever e classificar o impacto da Inteligência Artificial no mercado de trabalho.

---

# 📌 Contexto

A Inteligência Artificial (IA) está transformando o mercado de trabalho globalmente, automatizando funções e redefinindo papéis profissionais.

Com base no dataset **"AI Impact on Job Market (2024–2030)"**, disponível no Kaggle, desenvolvemos um modelo preditivo utilizando uma **Rede Neural Artificial do tipo Multilayer Perceptron (MLP)** para classificar o impacto da IA nas profissões.

🔗 Dataset utilizado:  
https://www.kaggle.com/datasets/sahilislam007/ai-impact-on-job-market-20242030

---

# 🎯 Objetivo do Projeto

Construir um modelo de Machine Learning capaz de:

- Classificar o impacto da IA em diferentes profissões
- Aplicar técnicas de pré-processamento de dados
- Utilizar Rede Neural MLP com Scikit-learn
- Avaliar o desempenho do modelo com métricas estatísticas
- Interpretar os resultados e propor recomendações estratégicas

---

# 🧠 Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# 📊 Etapas do Projeto

## 1️⃣ Introdução
Contextualização do impacto da IA no mercado de trabalho e definição do problema de classificação.

---

## 2️⃣ Análise Exploratória de Dados (EDA)

- Análise estatística descritiva (`df.describe()`)
- Verificação da estrutura (`df.info()`)
- Visualizações gráficas (histogramas, boxplots e gráficos de barras)
- Identificação de padrões e insights relevantes

Exemplos de análises realizadas:
- Relação entre setor profissional e impacto da IA
- Influência da escolaridade na exposição à automação
- Correlação entre adoção de IA e crescimento/redução de vagas

---

## 3️⃣ Pré-processamento

- Tratamento de valores ausentes (remoção ou imputação)
- Codificação de variáveis categóricas com `pd.get_dummies()`
- Padronização com `StandardScaler()`
- Divisão treino/teste (80/20) utilizando `train_test_split()`

---

## 4️⃣ Modelagem com Rede Neural MLP

Configuração da rede neural:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=800,
    random_state=42
)
# 📊 5️⃣ Avaliação do Modelo

Após o treinamento, o modelo foi avaliado utilizando métricas clássicas de classificação:

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusão

## Código de Avaliação:

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1-Score:", f1_score(y_test, y_pred, average='macro'))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
plt.show()
```

## Análise da Matriz de Confusão

A matriz de confusão permitiu analisar:

- Verdadeiros Positivos
- Verdadeiros Negativos
- Falsos Positivos
- Falsos Negativos

Isso possibilitou uma visão mais detalhada do desempenho do modelo além da accuracy.

---

# 🧾 6️⃣ Conclusões e Recomendações

Com base nos resultados obtidos:

- O modelo conseguiu identificar padrões relevantes no impacto da IA sobre diferentes profissões.
- Variáveis como setor, nível de escolaridade e adoção tecnológica demonstraram influência significativa.
- A Rede Neural MLP apresentou boa capacidade de generalização nos dados de teste.

## 🔎 Limitações

- Dependência da qualidade e balanceamento do dataset
- Sensibilidade à escolha de hiperparâmetros
- Possível overfitting dependendo da configuração da rede

## 🚀 Melhorias Futuras

- Ajuste fino de hiperparâmetros com GridSearchCV
- Implementação de validação cruzada
- Testar outros algoritmos (Random Forest, XGBoost)
- Aplicar técnicas de balanceamento como SMOTE
- Explorar feature importance e interpretabilidade do modelo

---

# 📂 Estrutura do Projeto

```
GS_Future_of_work/
│
├── AI_Impact_MLP_NomeDoAluno.ipynb
├── README.md
└── dataset/
```

---

# 👨‍💻 Autor

Rodrigo Nery  
FIAP – Inteligência Artificial & Machine Learning

---

# ⭐ Considerações Finais

Este projeto demonstra a aplicação prática de Redes Neurais Artificiais na análise preditiva do impacto da Inteligência Artificial no mercado de trabalho, integrando conceitos de análise de dados, machine learning e transformação digital.
