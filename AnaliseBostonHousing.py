from email.charset import Charset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats


descricao_colunas = {   
    "CRIM" : "Taxa de criminalidade per capita por cidade  ",
    "ZN" : "Proporção de terrenos residenciais zoneados para lotes com mais de 25.000 pés²  ",
    "INDUS" : "Proporção de acres ocupados por comércios não varejistas por cidade  ",
    "CHAS" : "Variável fictícia do Rio Charset (1 se o trecho faz fronteira com o rio; 0 caso contrário)  ",
    "NOX" : "Concentração de óxidos nítricos (partes por 10 milhões)  ",
    "RM" : "Número médio de cômodos por habitação  ",
    "AGE" : "Proporção de unidades ocupadas pelos proprietários construídas antes de 1940  ",
    "DIS" : "Distâncias ponderadas até cinco centros de emprego em Boston  ",
    "RAD" : "Índice de acessibilidade às rodovias radiais  ",
    "TAX" : "Taxa de imposto predial de valor total por $10.000  ",
    "PTRATIO" : "Proporção de alunos por professor por cidade  ",
    "B" : "1000(Bk - 0,63)², onde Bk é a proporção de negros por cidade  ",
    "LSTAT" : "Percentual da população de menor status socioeconômico  ",
    "MEDV" : "Valor mediano das casas ocupadas pelos proprietários (em milhares de dólares)",
}


# Carregar o dataset
df = pd.read_csv('BostonHousing.csv')

# Visualizar as primeiras linhas do dataset
st.title('Análise do Dataset Boston Housing')
st.write('### Visualização das primeiras linhas do dataset')
st.write(df.head())

# Selecionar colunas para análise
st.sidebar.write('### Selecione as colunas para análise X e Y')
all_columns = df.columns.tolist()

# Define default indices safely
default_x_index = 0
if 'lstat' in all_columns:
    default_x_index = all_columns.index('lstat')

default_y_index = 0
if len(all_columns) > 1: # Default to the second column if available
    default_y_index = 1
if 'medv' in all_columns:
    default_y_index = all_columns.index('medv')

# Ensure default_y_index is different from default_x_index if possible and multiple columns exist
if len(all_columns) > 1 and default_x_index == default_y_index:
    default_y_index = (default_x_index + 1) % len(all_columns)
elif len(all_columns) == 1: # If only one column, x and y must be the same
    default_y_index = 0

x_col = st.sidebar.selectbox(
    'Selecione a coluna para X:',
    all_columns,
    index=default_x_index,
    key='x_col_select'
)
y_col = st.sidebar.selectbox(
    'Selecione a coluna para Y:',
    all_columns,
    index=default_y_index,
    key='y_col_select'
)

# Botão para iniciar a análise
run_analysis_button = st.sidebar.button("Realizar Análise")

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
st.write(f"""
# As variaveis que escolhi para analise sao:
## {x_col} e {y_col}
- {x_col}: {descricao_colunas[x_col.upper()]}
- {y_col}: {descricao_colunas[y_col.upper()]}
""")
st.sidebar.write(f"""
# X: {x_col} e Y: {y_col}
- {x_col}: {descricao_colunas[x_col.upper()]}
- {y_col}: {descricao_colunas[y_col.upper()]}
""")

if run_analysis_button:
    # Estatísticas para x e y selecionados
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

    st.write(f'**Estatísticas para {x_col}:**')
    st.write(stats_x)
    st.write(f'**Estatísticas para {y_col}:**')
    st.write(stats_y)

    # Coeficiente de correlação entre x e y
    correlation_xy = df[x_col].corr(df[y_col])
    st.write(f'### Coeficiente de Correlação entre {x_col} e {y_col}')
    st.write(f'{correlation_xy:.4f}')

    # Boxplots
    st.write('### Boxplots para x e y')

    # Boxplot para x
    st.write(f'#### Boxplot para {x_col}')
    sns.boxplot(x=df[x_col])
    st.pyplot(plt.gcf())
    plt.clf()

    # Boxplot para y
    st.write(f'#### Boxplot para {y_col}')
    sns.boxplot(x=df[y_col])
    st.pyplot(plt.gcf())
    plt.clf()

    # Teste de Normalidade (Shapiro-Wilk)
    st.write('### Teste de Normalidade (Shapiro-Wilk)')

    # Verifica se as colunas selecionadas são numéricas antes de aplicar o teste de Shapiro-Wilk
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        shapiro_x_stat, shapiro_x_p = stats.shapiro(df[x_col].dropna())
        st.write(f'**Teste de Shapiro-Wilk para {x_col}:**')
        st.write(f'Estatística={shapiro_x_stat:.4f}, p-valor={shapiro_x_p:.4f}')
        if shapiro_x_p > 0.05:
            st.write(f'{x_col} parece ser normalmente distribuído (não se rejeita H0)')
        else:
            st.write(f'{x_col} não parece ser normalmente distribuído (rejeita-se H0)')

        shapiro_y_stat, shapiro_y_p = stats.shapiro(df[y_col].dropna())
        st.write(f'**Teste de Shapiro-Wilk para {y_col}:**')
        st.write(f'Estatística={shapiro_y_stat:.4f}, p-valor={shapiro_y_p:.4f}')
        if shapiro_y_p > 0.05:
            st.write(f'{y_col} parece ser normalmente distribuído (não se rejeita H0)')
        else:
            st.write(f'{y_col} não parece ser normalmente distribuído (rejeita-se H0)')
    else:
        st.warning(f"Uma ou ambas as colunas ({x_col}, {y_col}) não são numéricas. O teste de Shapiro-Wilk não pode ser aplicado.")

    # Visualizar a distribuição das variáveis selecionadas
    st.write('### Visualizar a distribuição das variáveis selecionadas')
    st.write(f'#### Histograma para {x_col}')
    sns.histplot(df[x_col], kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

    st.write(f'#### Histograma para {y_col}')
    sns.histplot(df[y_col], kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

    # Criar um gráfico de dispersão entre as colunas selecionadas
    st.write(f'### Gráfico de dispersão entre {x_col} e {y_col}')
    sns.scatterplot(x=x_col, y=y_col, data=df)
    st.pyplot(plt.gcf())
    plt.clf()
else:
    st.info("Selecione as colunas X e Y na barra lateral e clique em 'Realizar Análise' para ver os resultados.")



