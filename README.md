### Esercizi svolti
* [hiring decisions](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb)
* [mobile price](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb)
* [Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb)
* [hotel bookings (esame del 30-01-26)](exams/260130/hotel-bookings.ipynb)
* [bank marketing (esame del 08-01-26)](exams/260108/bank-marketing.ipynb)
* [loan default (esame del 18-02-25)](exams/250218/loan-default.ipynb)
* [diamonds (esame del 01-25)](exams/250110/diamonds.ipynb)
* [bank dropout (esame del 10-09-24)](exams/240910/bank-dropout.ipynb)

---

### Indice

1. [Analisi esplorativa (Parte 1)](#1-analisi-esplorativa-parte-1) — completezza, feature engineering, pd.cut/qcut, pivot, visualizzazioni
2. [Preprocessing (Parte 2)](#2-preprocessing-parte-2) — pulizia, OHE, LabelEncoder, split, scaler, KBins, imputer
3. [Modelli](#3-modelli) — DecisionTree, ExtraTree, KNN, RandomForest, SGD, Dummy
4. [Valutazione](#4-valutazione) — F1-score, confusion matrix, cross-validation, overfitting
5. [Hyperparameter tuning](#5-hyperparameter-tuning) — GridSearchCV
6. [Pipeline e ColumnTransformer](#6-pipeline-e-columntransformer) — Pipeline, ColumnTransformer, FeatureUnion
7. [Feature selection e dimensionality reduction](#7-feature-selection-e-dimensionality-reduction) — SelectKBest, TruncatedSVD, correlazione
8. [Feature importance e explainability](#8-feature-importance-e-explainability) — PFI, LOCO, feature_importances_
9. [Fairness](#9-fairness) — confronto per gruppo, fairlearn

---

### 1. Analisi esplorativa (Parte 1)

#### Completezza, bilanciamento, dimensione dataset
Controllare `df.index.size`, `df.isna().any().any()`, `df.groupby(target).count()`.
* [HB — P1 Q1](exams/260130/hotel-bookings.ipynb#p1-q1)
* [BM — P1 Q1](exams/260108/bank-marketing.ipynb#p1-q1)
* [LD — P1 Q1](exams/250218/loan-default.ipynb#p1-q1)
* [DI — P1 Q1](exams/250110/diamonds.ipynb#p1-q1)
* [HD — Q1](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q1)
* [BD — P1 Q1](exams/240910/bank-dropout.ipynb#p1-q1)

#### Feature engineering (nuove colonne calcolate)
Creare colonne derivate per l'analisi (es. somma, rapporto, volume).
* [HB — P1 Q2](exams/260130/hotel-bookings.ipynb#p1-q2) — `stays_nights = weekend + week`
* [LD — P1 Q2](exams/250218/loan-default.ipynb#p1-q2) — `ratio = loan_amount / property_value`
* [DI — P1 Q2](exams/250110/diamonds.ipynb#p1-q2) — `Volume = Length * Width * Height`
* [DI — P1 Q4](exams/250110/diamonds.ipynb#p1-q4) — `Price_Carat = Price / Carat`
* [BD — P2 Q4](exams/240910/bank-dropout.ipynb#p2-q4) — `engineered = (EstimatedSalary * Tenure + Balance) / 2`

#### Discretizzazione con `pd.cut` / `pd.qcut`
`pd.cut` divide in intervalli equispaziati, `pd.qcut` in quantili con uguale numerosita.
* [HB — P1 Q3](exams/260130/hotel-bookings.ipynb#p1-q3) — `pd.cut` su `lead_time`, 5 bin
* [BM — P1 Q3](exams/260108/bank-marketing.ipynb#p1-q3) — `pd.cut` su `hours.per.week`, bin manuali
* [LD — P1 Q3](exams/250218/loan-default.ipynb#p1-q3) — `pd.cut` su `Credit_Score` e `income`, 5 bin
* [DI — P1 Q3](exams/250110/diamonds.ipynb#p1-q3) — `pd.qcut` su `Carat Weight`, q=5
* [BD — P1 Q2](exams/240910/bank-dropout.ipynb#p1-q2) — `pd.qcut` su `Age`, q=5
* [BD — P1 Q3](exams/240910/bank-dropout.ipynb#p1-q3) — `pd.cut` su `EstimatedSalary`, 5 bin
* [BD — P2 Q4](exams/240910/bank-dropout.ipynb#p2-q4) — `pd.cut` su `engineered`, 10 bin

#### Tabelle pivot
`pd.pivot_table(data, index, columns, values, aggfunc)` per analisi incrociate.
* [HB — P1 Q3](exams/260130/hotel-bookings.ipynb#p1-q3)
* [BM — P1 Q3](exams/260108/bank-marketing.ipynb#p1-q3)
* [LD — P1 Q3](exams/250218/loan-default.ipynb#p1-q3)
* [DI — P1 Q3](exams/250110/diamonds.ipynb#p1-q3)
* [BD — P1 Q3](exams/240910/bank-dropout.ipynb#p1-q3)

#### Visualizzazioni
`lineplot`, `barplot`, `boxplot`, `regplot` di seaborn per esplorare relazioni.
* [HB — P1 Q2](exams/260130/hotel-bookings.ipynb#p1-q2) — `lineplot`
* [HB — P1 Q4](exams/260130/hotel-bookings.ipynb#p1-q4) — `barplot`
* [BM — P1 Q4](exams/260108/bank-marketing.ipynb#p1-q4) — `boxplot`
* [LD — P1 Q4](exams/250218/loan-default.ipynb#p1-q4) — `regplot`
* [DI — P1 Q2](exams/250110/diamonds.ipynb#p1-q2) — `regplot`
* [DI — P1 Q4](exams/250110/diamonds.ipynb#p1-q4) — `boxplot`
* [BD — P1 Q2](exams/240910/bank-dropout.ipynb#p1-q2) — `barplot`
* [BD — P1 Q4](exams/240910/bank-dropout.ipynb#p1-q4) — `histplot`

---

### 2. Preprocessing (Parte 2)

#### Pulizia dati
Drop colonne inutili, `dropna`, `drop_duplicates`, soglia 50% null (`dropna(axis=1, thresh=n/2)`).
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [HB — P2 Q6](exams/260130/hotel-bookings.ipynb#p2-q6) — reload + pulizia con soglia 50%
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1)
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1)
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1)
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1)

#### One-Hot Encoding
`pd.get_dummies` (fuori pipeline) oppure `OneHotEncoder` (dentro pipeline/ColumnTransformer).
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1) — `pd.get_dummies` + `OneHotEncoder` in pipeline
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1) — `pd.get_dummies`
* [BM — P2 Q5](exams/260108/bank-marketing.ipynb#p2-q5) — `OneHotEncoder` in pipeline
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1) — `pd.get_dummies`
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1) — `pd.get_dummies`
* [DI — P2 Q6](exams/250110/diamonds.ipynb#p2-q6) — `OneHotEncoder` in pipeline
* [TI — Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#titanic) — `OneHotEncoder` in pipeline
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1) — `pd.get_dummies`

#### LabelEncoder
Trasforma target categorico multiclasse in numeri interi.
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1) — `LabelEncoder` su colonna `Type`

#### Train/test split stratificato
`train_test_split(X, y, train_size=.75, stratify=y, random_state=42)`.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1)
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1)
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1)
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5)
* [TI — Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#titanic)
* [HD — Q2](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q2)
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1)

#### StandardScaler
Porta le feature a media 0 e varianza 1. Usato spesso con KNN e SGD.
* [HB — P2 Q4](exams/260130/hotel-bookings.ipynb#p2-q4) — in pipeline
* [BM — P2 Q5](exams/260108/bank-marketing.ipynb#p2-q5) — in pipeline
* [DI — P2 Q6](exams/250110/diamonds.ipynb#p2-q6) — in pipeline
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5) — in ColumnTransformer
* [HD — Q3](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q3) — in ColumnTransformer

#### MinMaxScaler
Scala nell'intervallo [0, 1].
* [BM — P2 Q4](exams/260108/bank-marketing.ipynb#p2-q4) — su `capital.gain`, `capital.loss`
* [DI — P2 Q5](exams/250110/diamonds.ipynb#p2-q5) — su `Price`
* [BD — P2 Q6](exams/240910/bank-dropout.ipynb#p2-q6) — su `Tenure`

#### KBinsDiscretizer
Discretizza feature continue in N bin. Parametri: `n_bins`, `encode`, `strategy`.
* [HB — P2 Q4](exams/260130/hotel-bookings.ipynb#p2-q4) — su `lead_time`, `adr`, 5 bin
* [BM — P2 Q4](exams/260108/bank-marketing.ipynb#p2-q4) — su `age`, `hours.per.week`
* [LD — P2 Q6](exams/250218/loan-default.ipynb#p2-q6) — su `income`, `loan_amount`, 7 bin
* [DI — P2 Q5](exams/250110/diamonds.ipynb#p2-q5) — su `Length`, `Width`, `Height`
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5) — su `mobile_wt`, `battery_power`, 5 bin
* [BD — P2 Q6](exams/240910/bank-dropout.ipynb#p2-q6) — su `Balance`, `EstimatedSalary`, 6 bin

#### SimpleImputer
Sostituisce valori nulli con strategia (`mean`, `median`, `most_frequent`).
* [HB — P2 Q6](exams/260130/hotel-bookings.ipynb#p2-q6) — `most_frequent`
* [DI — P2 Q6](exams/250110/diamonds.ipynb#p2-q6) — `mean` (default)
* [TI — Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#titanic) — `median` per numeriche, `most_frequent` per categoriche

---

### 3. Modelli

#### DecisionTreeClassifier
Albero di decisione. Parametri chiave: `max_depth`, `min_samples_leaf`, `criterion`.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1)
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5)
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1)

#### ExtraTreeClassifier
Albero con split casuali (meno overfitting del DecisionTree puro, ma comunque rischia senza `max_depth`).
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1)
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1)

#### KNeighborsClassifier
Classificatore basato sui K vicini. Parametri: `n_neighbors`, `weights`. Sensibile allo scaling.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1)
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1)
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1)
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1)

#### RandomForestClassifier
Ensemble di alberi. Parametri: `n_estimators`, `max_depth`.
* [TI — Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#titanic)

#### SGDClassifier
Classificatore lineare con Stochastic Gradient Descent. Richiede scaling delle feature.
* [HD — Q3](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q3)

#### DummyClassifier (baseline)
Classificatore che predice sempre la classe piu frequente. Serve come baseline per confronto.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1)
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1)
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1)
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1)

---

### 4. Valutazione

#### F1-score (binary vs weighted)
`f1_score(y_true, y_pred)` per binario, `f1_score(..., average='weighted')` per multiclasse.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1) — binary
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1) — binary
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1) — binary
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1) — weighted
* [HD — Q3](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q3) — binary
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1) — binary

#### Confusion Matrix / ConfusionMatrixDisplay
`confusion_matrix` per la matrice raw, `ConfusionMatrixDisplay.from_predictions` o `.from_estimator` per il plot.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1) — `confusion_matrix` + `ConfusionMatrixDisplay`
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1) — `confusion_matrix`
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1) — `confusion_matrix`
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1) — `ConfusionMatrixDisplay`
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5) — `ConfusionMatrixDisplay`
* [TI — Score](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#score-and-visualization) — `ConfusionMatrixDisplay`
* [HD — Q3](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q3) — `ConfusionMatrixDisplay`
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1) — `ConfusionMatrixDisplay`

#### Cross-validation (`cross_validate`)
Valutazione k-fold. `cross_validate(model, X, y, cv=10, scoring='f1')`.
* [LD — P2 Q2](exams/250218/loan-default.ipynb#p2-q2) — cv=10, scoring=`f1`
* [DI — P2 Q2](exams/250110/diamonds.ipynb#p2-q2) — cv=10, scoring=`f1_weighted`
* [BD — P2 Q2](exams/240910/bank-dropout.ipynb#p2-q2) — cv=10, scoring=`accuracy`

#### Confronto train vs test (overfitting)
Confrontare le metriche su train e test per diagnosticare overfitting.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [BM — P2 Q1](exams/260108/bank-marketing.ipynb#p2-q1)
* [LD — P2 Q1](exams/250218/loan-default.ipynb#p2-q1)
* [DI — P2 Q1](exams/250110/diamonds.ipynb#p2-q1)
* [BD — P2 Q1](exams/240910/bank-dropout.ipynb#p2-q1)

---

### 5. Hyperparameter tuning

#### GridSearchCV
`GridSearchCV(pipeline, param_grid, scoring, cv)`. I parametri nella pipeline usano la sintassi `step__param`.
* [HB — P2 Q2](exams/260130/hotel-bookings.ipynb#p2-q2) — `model__weights`, `model__n_neighbors`
* [BM — P2 Q3](exams/260108/bank-marketing.ipynb#p2-q3) — `criterion`, `max_depth`
* [LD — P2 Q5](exams/250218/loan-default.ipynb#p2-q5) — `criterion`, `max_depth`
* [DI — P2 Q4](exams/250110/diamonds.ipynb#p2-q4) — `weights`, `n_neighbors`
* [MP — Q6](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q6) — `selectkbest__k`, `n_bins`, `criterion`, `min_samples_leaf`
* [TI — GridSearch](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#gridsearch) — `strategy`, `n_estimators`, `max_depth`
* [BD — P2 Q5](exams/240910/bank-dropout.ipynb#p2-q5) — `criterion`, `max_depth`
* [BD — P2 Q7](exams/240910/bank-dropout.ipynb#p2-q7) — `svd__n_components`, `preprocessing__discretizer__n_bins`

---

### 6. Pipeline e ColumnTransformer

#### Pipeline base (preprocessing + model)
`Pipeline(steps=[('preprocessing', ...), ('model', ...)])`. Concatena trasformazioni e modello.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1)
* [BM — P2 Q4](exams/260108/bank-marketing.ipynb#p2-q4)
* [LD — P2 Q6](exams/250218/loan-default.ipynb#p2-q6)
* [DI — P2 Q5](exams/250110/diamonds.ipynb#p2-q5)
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5)
* [TI — Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#titanic)
* [HD — Q3](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q3)
* [BD — P2 Q6](exams/240910/bank-dropout.ipynb#p2-q6)

#### ColumnTransformer
Applica trasformazioni diverse a colonne diverse. `make_column_selector(dtype_include=...)` per selezionare automaticamente.
* [HB — P2 Q1](exams/260130/hotel-bookings.ipynb#p2-q1) — `OneHotEncoder` per categoriche, rest passthrough
* [HB — P2 Q4](exams/260130/hotel-bookings.ipynb#p2-q4) — `KBinsDiscretizer` + `StandardScaler` + `OneHotEncoder`
* [BM — P2 Q4](exams/260108/bank-marketing.ipynb#p2-q4) — `KBinsDiscretizer` + `MinMaxScaler`
* [BM — P2 Q5](exams/260108/bank-marketing.ipynb#p2-q5) — `OneHotEncoder` + `StandardScaler`
* [LD — P2 Q6](exams/250218/loan-default.ipynb#p2-q6) — `KBinsDiscretizer` + `StandardScaler`
* [DI — P2 Q5](exams/250110/diamonds.ipynb#p2-q5) — `KBinsDiscretizer` + `MinMaxScaler`
* [DI — P2 Q6](exams/250110/diamonds.ipynb#p2-q6) — `OneHotEncoder`, rest passthrough
* [MP — Q5](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q5) — `StandardScaler` + `KBinsDiscretizer`
* [TI — Titanic](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#titanic) — `SimpleImputer` + `OneHotEncoder`
* [HD — Q3](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q3) — `StandardScaler`, rest passthrough
* [BD — P2 Q6](exams/240910/bank-dropout.ipynb#p2-q6) — `KBinsDiscretizer` + `MinMaxScaler`

#### FeatureUnion (original + SVD)
`FeatureUnion` combina le feature originali (post-preprocessing) con le componenti SVD.
* [HB — P2 Q5](exams/260130/hotel-bookings.ipynb#p2-q5)
* [DI — P2 Q6](exams/250110/diamonds.ipynb#p2-q6)
* [MP — Q7](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q7)

---

### 7. Feature selection e dimensionality reduction

#### SelectKBest
Seleziona le K feature con il punteggio piu alto (default: `f_classif`).
* [BM — P2 Q6](exams/260108/bank-marketing.ipynb#p2-q6)
* [MP — Q6](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q6)

#### TruncatedSVD
Riduzione dimensionale, utile con matrici sparse (dopo OneHotEncoder). Si usa in `FeatureUnion` o standalone.
* [HB — P2 Q5](exams/260130/hotel-bookings.ipynb#p2-q5) — in FeatureUnion + GridSearchCV su `n_components`
* [LD — P2 Q7](exams/250218/loan-default.ipynb#p2-q7) — standalone + GridSearchCV
* [DI — P2 Q6](exams/250110/diamonds.ipynb#p2-q6) — in pipeline + GridSearchCV
* [MP — Q7](colab_2025/8-Esercitazioni/Simulazione%20d_esame/mobile-price.ipynb#q7) — in FeatureUnion + GridSearchCV
* [BD — P2 Q7](exams/240910/bank-dropout.ipynb#p2-q7) — standalone + GridSearchCV

#### Analisi correlazione con target
`data.corr()['target']` per trovare le feature piu correlate e usarle per un modello ridotto.
* [LD — P2 Q4](exams/250218/loan-default.ipynb#p2-q4) — top 2 positive + top 2 negative

---

### 8. Feature importance e explainability

#### Permutation Feature Importance (PFI)
`permutation_importance(model, X_test, y_test, n_repeats, scoring)`. Misura quanto peggiora lo score permutando una feature.
* [HB — P2 Q3](exams/260130/hotel-bookings.ipynb#p2-q3) — KNN, 5 permutazioni, f1
* [DI — P2 Q3](exams/250110/diamonds.ipynb#p2-q3) — KNN, 5 permutazioni, f1_weighted
* [HD — Q5](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q5) — SGDClassifier, 10 permutazioni, f1

#### Leave-One-Covariate-Out (LOCO)
Si rimuove una feature alla volta e si riaddestra il modello. Il delta di F1 indica l'importanza.
* [HD — Q6](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q6)

#### Feature importances da albero (`feature_importances_`)
Attributo degli alberi e dei forest. Restituisce l'importanza Gini/entropy di ogni feature.
* [TI — Score](colab_2025/7-Sklearn%20advanced/esercizi/7_home_exercise_titanic.ipynb#score-and-visualization) — RandomForestClassifier + barplot

---

### 9. Fairness

#### Confronto predizioni per gruppo protetto
Confrontare le predizioni del modello filtrate per un attributo sensibile (es. `sex`, `Gender`).
* [BM — P2 Q2](exams/260108/bank-marketing.ipynb#p2-q2) — ExtraTree, confronto uomini/donne su `income`, aware vs unaware
* [LD — P2 Q3](exams/250218/loan-default.ipynb#p2-q3) — ExtraTree e KNN, confronto `Gender`
* [BD — P2 Q3](exams/240910/bank-dropout.ipynb#p2-q3) — DecisionTree, confronto Male/Female su `Gender`, aware vs unaware

#### fairlearn (`MetricFrame`, `demographic_parity_ratio`, `selection_rate`)
Libreria per analisi di fairness. `MetricFrame` calcola metriche per ogni gruppo, `demographic_parity_ratio` verifica la parita.
* [HD — Q4](colab_2025/8-Esercitazioni/Fairness%20and%20explainability/hiring-decisions.ipynb#q4) — `MetricFrame`, `demographic_parity_ratio`, modello unaware
* [BD — P2 Q3](exams/240910/bank-dropout.ipynb#p2-q3) — `MetricFrame`, modello unaware
