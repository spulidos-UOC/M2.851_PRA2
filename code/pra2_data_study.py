#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importar la biblioteca necesaria
import pandas as pd

# Carga el conjunto de datos
data = pd.read_csv('/kaggle/input/heart-attack-analysis-prediction-dataset/heart.csv')

# Muestra las primeras filas del conjunto de datos
data.head()


# In[3]:


# Verificar si hay elementos vacíos en el conjunto de datos
data.isnull().sum()


# In[4]:


# Verificar si hay ceros en el conjunto de datos
(data == 0).sum()


# In[5]:


# Calcular el IQR para cada columna
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Definir los límites superior e inferior para los valores extremos
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar los valores extremos
outliers = data[(data < lower_bound) | (data > upper_bound)].count()
print(outliers)


# In[6]:


# Importar la biblioteca necesaria
import numpy as np

# Calcular la mediana para cada columna
medians = data.median()

# Reemplazar los outliers por la mediana en cada columna
for column in ['trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'thall']:
    data[column] = np.where((data[column] < lower_bound[column]) | 
                            (data[column] > upper_bound[column]), 
                            medians[column], 
                            data[column])


# In[7]:


from scipy.stats import shapiro

# Realizar la prueba de Shapiro-Wilk en cada columna
for column in data.columns:
    _, p_value = shapiro(data[column])
    if p_value < 0.05:
        print(f"{column} no sigue una distribución normal")
    else:
        print(f"{column} sigue una distribución normal")


# In[10]:


import matplotlib.pyplot as plt
import scipy.stats as stats

variables_to_check = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 
                      'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

for var in variables_to_check:
    # Histograma
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data[var], bins='auto', alpha=0.7, rwidth=0.85)
    plt.title(f'Histograma de "{var}"')

    # Gráfico Q-Q
    plt.subplot(1, 2, 2)
    stats.probplot(data[var], dist="norm", plot=plt)
    plt.title(f'Gráfico Q-Q de "{var}"')
    plt.tight_layout()
    plt.show()


# In[11]:


from scipy.stats import levene

# Comprobando la homogeneidad de la varianza para las variables seleccionadas
variables_to_check = [('sex', 'output'), ('age', 'output'), 
                      ('cp', 'output'), ('trtbps', 'output')]

for var_pair in variables_to_check:
    _, p_value = levene(data[var_pair[0]], data[var_pair[1]])
    if p_value < 0.05:
        print(f"Las varianzas entre {var_pair[0]} y {var_pair[1]} no son homogéneas")
    else:
        print(f"Las varianzas entre {var_pair[0]} y {var_pair[1]} son homogéneas")


# In[12]:


from scipy.stats import chi2_contingency

# Prueba de Chi-Cuadrado
crosstab = pd.crosstab(data['sex'], data['output'])
_, p_value, _, _ = chi2_contingency(crosstab)
print(f"Chi-Cuadrado p-value: {p_value}")


# In[13]:


from scipy.stats import ttest_ind

# Prueba T de Student
group1 = data[data['output'] == 1]['age']
group2 = data[data['output'] == 0]['age']
_, p_value = ttest_ind(group1, group2)
print(f"T-Test p-value: {p_value}")


# In[14]:


from scipy.stats import pearsonr

# Análisis de correlación
_, p_value = pearsonr(data['age'], data['chol'])
print(f"Pearson Correlation p-value: {p_value}")


# In[15]:


# Generar una tabla de contingencia
contingency_table = pd.crosstab(data['sex'], data['output'])
print(contingency_table)


# In[17]:


import seaborn as sns

# Distribución de la enfermedad por género
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='sex', hue='output')
plt.title('Distribución de enfermedad cardíaca por género')
plt.show()


# In[18]:


# Comparar las edades entre los grupos con y sin enfermedad cardíaca
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='output', y='age')
plt.title('Comparación de la edad en función de enfermedad cardíaca')
plt.show()


# In[19]:


# Visualizar la relación entre edad y nivel de colesterol
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='age', y='chol')
plt.title('Relación entre la edad y el nivel de colesterol')
plt.show()


# In[20]:


# Comparar los niveles de colesterol entre los grupos con y sin enfermedad cardíaca
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='output', y='chol')
plt.title('Niveles de colesterol en función de la presencia de enfermedad cardíaca')
plt.show()

