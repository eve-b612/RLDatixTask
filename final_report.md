**Final report: RLDatix Data Scientist assignment**
**My approach**
The task involved two parts: binary classification to predict readmission within 30 days, and entity extraction and labelling from discharge notes. I first performed exploratory data analysis to understand the dataset, check for missing values, looked at correlations, and assessed class imbalance. For feature engineering, I handled categorical variables with one-hot encoding and imputation to convert them to numerical variables, and derived a new feature: medication-diagnosis combinations. I used XGBoost as the main classifier, tuning scale_pos_weight to account for class imbalance.

For the NER task, I used the d4data/biomedical-ner-all model from Hugging Face, with a manual mapping of biomedical entities to the categories provided in the task (Diagnosis, Treatment, Symptoms, Medications, Follow-up).

**Key Results**
XGBoost yielded an ROC AUC of 0.59 and an F1 score of 0.37 on the minority class. This is not ideal but was expected given the very limited sample size (200 rows). Feature importance plots suggested that variables like previous admissions and length of stay had modest influence, and the most influential variables were the diagnosis and medication. For the NER task, the model was able to extract relevant entities and I mapped those to task-specific labels using a dictionary. However, there were inconsistencies in labelling, in part because the model used different entity labels to the ones requested in this task, but also because the model lacks robust contextual understanding.

**Practical Implications**
For the classification task, a small dataset limits generalisability and reduces model performance. Real patient data is noisy and humans are complex, so even small shifts in behavior or treatment history can influence outcomes in unpredictable ways. This means models trained on small samples and a limited number of variables are more likely to underperform.

For the NER task, using general-purpose or even domain-specific models comes with risks of hallucination (generating "plausible" but incorrect labels), entity ambiguity (e.g. "pain" could be a symptom or a diagnosis), and inconsistent labelling. Models don’t always understand clinical differences which is critical in healthcare.

**If I Had More Time or Data**
With more time, I would tune the classifier further and try other models (grid search on XGBoost, RF, logistic regression). I would try to create more meaningful features (e.g. other variable interactions), and try oversampling or bootstrapping methods. On the NER side, I would try out more models, and aside from label mapping I would also include some additional prompting for contextual consideration when assigning labels.
If I had more data, there would probably be improvement using the classical binary classification algorithms (XGBoost, RF, logistic regression), but it would also allow for the use of more complex algorithms such as feedforward neural nets (e.g. MLP) and for the NER task try using RRNs (e.g. LSTM, BiLSTM).
