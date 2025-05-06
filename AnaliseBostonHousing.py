import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

# Carregar o dataset
df = pd.read_csv('BostonHousing.csv')

# Visualizar as primeiras linhas do dataset
st.title('Análise do Dataset Boston Housing')
st.write('### Visualização das primeiras linhas do dataset')
st.write(df.head())

# Pré-processamento: Verificar e remover dados duplicados
st.write('### Verificar e remover dados duplicados')
st.write(f"Número de linhas duplicadas: {df.duplicated().sum()}")
df = df.drop_duplicates()
st.write(f"Número de linhas duplicadas após remoção: {df.duplicated().sum()}")

# Estatísticas descritivas
st.write('### Estatísticas descritivas')
st.write(df.describe())

# Verificar se há valores nulos
st.write('### Verificar se há valores nulos')
st.write(df.isnull().sum())
st.write("""
# As variaveis que escolhi para analise sao:
## lstat e medv
- lstat: percentual de nivel socioeconomico 
- medv: valor da casa
""")
# Definir x e y
x_col = 'lstat'
y_col = 'medv'

# Estatísticas para x (lstat) e y (medv)
st.write(f'### Estatísticas para {x_col} (x) e {y_col} (y)')

stats_x = pd.DataFrame({
    'Média': [df[x_col].mean()],
    'Variância': [df[x_col].var()],
    'Desvio Padrão': [df[x_col].std()],
    'Mediana': [df[x_col].median()]
}, index=[x_col])

stats_y = pd.DataFrame({
    'Média': [df[y_col].mean()],
    'Variância': [df[y_col].var()],
    'Desvio Padrão': [df[y_col].std()],
    'Mediana': [df[y_col].median()]
}, index=[y_col])

st.write(f"**Estatísticas para {x_col}:**")
st.write(stats_x)
st.write(f"**Estatísticas para {y_col}:**")
st.write(stats_y)

# Coeficiente de correlação entre x e y
correlation_xy = df[x_col].corr(df[y_col])
st.write(f'### Coeficiente de Correlação entre {x_col} e {y_col}')
st.write(f'{correlation_xy:.4f}')

# Visualizar a distribuição das variáveis (Histogramas e Densidade)
st.write('### Visualizar a distribuição das variáveis (Histogramas e Densidade)')

# Histograma e Densidade para x (lstat)
st.write(f'#### Histograma e Gráfico de Densidade para {x_col}')
sns.histplot(df[x_col], kde=True) # kde=True adiciona o gráfico de densidade
st.pyplot(plt.gcf())
plt.clf()

# Histograma e Densidade para y (medv)
st.write(f'#### Histograma e Gráfico de Densidade para {y_col}')
sns.histplot(df[y_col], kde=True)
st.pyplot(plt.gcf())
plt.clf()

# Boxplots
st.write('### Boxplots para x e y')

# Boxplot para x (lstat)
st.write(f'#### Boxplot para {x_col}')
sns.boxplot(x=df[x_col])
st.pyplot(plt.gcf())
plt.clf()

# Boxplot para y (medv)
st.write(f'#### Boxplot para {y_col}')
sns.boxplot(x=df[y_col])
st.pyplot(plt.gcf())
plt.clf()

# Teste de Normalidade (Shapiro-Wilk)
st.write('### Teste de Normalidade (Shapiro-Wilk)')

shapiro_x_stat, shapiro_x_p = stats.shapiro(df[x_col])
st.write(f'**Teste de Shapiro-Wilk para {x_col}:**')
st.write(f'Estatística={shapiro_x_stat:.4f}, p-valor={shapiro_x_p:.4f}')
if shapiro_x_p > 0.05:
    st.write(f'{x_col} parece ser normalmente distribuído (não se rejeita H0)')
else:
    st.write(f'{x_col} não parece ser normalmente distribuído (rejeita-se H0)')

shapiro_y_stat, shapiro_y_p = stats.shapiro(df[y_col])
st.write(f'**Teste de Shapiro-Wilk para {y_col}:**')
st.write(f'Estatística={shapiro_y_stat:.4f}, p-valor={shapiro_y_p:.4f}')
if shapiro_y_p > 0.05:
    st.write(f'{y_col} parece ser normalmente distribuído (não se rejeita H0)')
else:
    st.write(f'{y_col} não parece ser normalmente distribuído (rejeita-se H0)')

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



