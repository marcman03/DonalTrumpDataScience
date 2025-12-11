# 📊 Predicción de Viralidad de Tweets de Donald Trump

Este proyecto analiza tweets de Donald Trump (2015–2016) con el objetivo de predecir si un tweet será **viral** (“mucho”) o **no viral** (“poco”) usando técnicas de minería de datos y modelos de machine learning.

---

## 🗂️ Dataset

- Fuente: Kaggle (Donald Trump Tweets Dataset).  
- 8.716 tweets y 11 columnas.  
- Columnas útiles para el modelo:
  - `Tweet_Text`
  - `Likes`
  - `Retweets`

Se eliminan columnas irrelevantes o vacías (`Media_Type`, `Hashtags`, `Unnamed:*`, etc.).

---

## 🧹 Preprocesamiento

1. Limpieza del texto:
   - Eliminación de HTML, puntuación y *stopwords*.
   - Lematización/stemming.
   - Eliminación de palabras cortas y no alfabéticas.

2. Vectorización:
   - `CountVectorizer (min_df = 5)`
   - Resultado: **7375 tweets × 1477 palabras**

3. Definición del target:
   - Viral = interacción ≥ 1.05 × (mediana likes + mediana retweets)
   - Dataset balanceado:
     - mucho: 3602  
     - poco: 3773  

---

## 📐 Evaluación

- División estratificada **70% train – 30% test**
- **10-Fold Cross Validation**
- Métricas:
  - accuracy  
  - precision  
  - recall  
  - F1-score  
  - matriz de confusión  
- Cálculo de intervalos de confianza al 95%

---

## 🤖 Modelos Probados

### 🟦 Naïve Bayes (MultinomialNB)
- Mejor rendimiento y mayor estabilidad  
- Umbral optimizado ≈ 0.236  
- **Accuracy:** ~72%  
- **IC 95%:** (0.696, 0.733)

### 🟩 KNN
- Mejor con `weights='distance'` y `SelectKBest`  
- **Accuracy:** ~68%  
- **IC 95%:** (0.662, 0.701)

### 🟧 Decision Tree
- Criterio entropy, ajuste de impureza mínima  
- **Accuracy:** ~67%  
- **IC 95%:** (0.657, 0.696)

### 🟥 SVM
- Kernels probados: lineal, polinomial, RBF  
- Mejor: **kernel lineal (~64%)**  
- **IC 95%:** (0.509, 0.657)

### 🟪 Meta-Learning (Ensembles)
- Métodos: Voting, Bagging, Random Forest, AdaBoost  
- Mejor ensemble: **Random Forest (~71%)**  
- **IC 95% RF:** (0.636, 0.658)

---

## 🏁 Conclusiones

- **Mejor modelo global:** Naïve Bayes  
- **Mejor ensemble:** Random Forest  
- **Modelo más interpretable:** Decision Tree  
- SVM y KNN funcionan, pero con precisión inferior o problemas de escalabilidad  
- Los modelos probabilísticos y ensembles funcionan mejor en texto de alta dimensionalidad  

---


---

## ▶️ Ejecución

```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/models/naive_bayes.py

