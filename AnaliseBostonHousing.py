import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# Carregar o dataset
df = pd.read_csv('BostonHousing.csv')

# Visualizar as primeiras linhas do dataset
st.title('Análise do Dataset Boston Housing')
st.write('### Visualização das primeiras linhas do dataset')
st.write(df.head())

# Estatísticas descritivas
st.write('### Estatísticas descritivas')
st.write(df.describe())

# Verificar se há valores nulos
st.write('### Verificar se há valores nulos')
st.write(df.isnull().sum())

# Verificar a correlação entre as variáveis
st.write('### Verificar a correlação entre as variáveis')
st.write(df.corr())

# Visualizar a distribuição das variáveis
st.write('### Visualizar a distribuição das variáveis')
sns.histplot(df['medv'])
st.pyplot(plt.gcf())
plt.clf() # Limpa a figura para próximos plots

# Criar um gráfico de dispersão entre 'lstat' e 'medv'
st.write('### Gráfico de dispersão entre lstat e medv')
sns.scatterplot(x='lstat', y='medv', data=df)
st.pyplot(plt.gcf())
plt.clf() # Limpa a figura para próximos plots  

# Criar um gráfico de dispersão entre 'lstat' e 'medv'
st.write('### Gráfico de dispersão entre lstat e medv')
sns.scatterplot(x='lstat', y='medv', data=df)
st.pyplot(plt.gcf())
plt.clf() # Limpa a figura para próximos plots  



