from email.charset import Charset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

descricao_colunas = {

    "BostonHousing.csv": {   
            "crim" : "Taxa de criminalidade per capita por cidade  ",
            "zn" : "Proporção de terrenos residenciais zoneados para lotes com mais de 25.000 pés²  ",
            "indus" : "Proporção de acres ocupados por comércios não varejistas por cidade  ",
            "chas" : "Variável fictícia do Rio Charset (1 se o trecho faz fronteira com o rio; 0 caso contrário)  ",
            "nox" : "Concentração de óxidos nítricos (partes por 10 milhões)  ",
            "rm" : "Número médio de cômodos por habitação  ",
            "age" : "Proporção de unidades ocupadas pelos proprietários construídas antes de 1940  ",
            "dis" : "Distâncias ponderadas até cinco centros de emprego em Boston  ",
            "rad" : "Índice de acessibilidade às rodovias radiais  ",
            "tax" : "Taxa de imposto predial de valor total por $10.000  ",
            "ptratio" : "Proporção de alunos por professor por cidade  ",
            "b" : "1000(Bk - 0,63)², onde Bk é a proporção de negros por cidade  ",
            "lstat" : "Percentual da população de menor status socioeconômico  ",
            "medv" : "Valor mediano das casas ocupadas pelos proprietários (em milhares de dólares)",
        },
    "winequality-red.csv": {
        "fixed acidity": "Ácidos não voláteis (principalmente tartárico, málico, cítrico e succínico) (g/dm³)",
        "volatile acidity": "Ácido acético (g/dm³)",
        "citric acid": "Ácido cítrico (g/dm³)",
        "residual sugar": "Açúcar restante após a fermentação (g/dm³)",
        "chlorides": "Quantidade de sal no vinho (g/dm³)",
        "free sulfur dioxide": "Dióxido de enxofre livre (SO₂) (mg/dm³)",
        "total sulfur dioxide": "Dióxido de enxofre total (SO₂) (mg/dm³)",
        "density": "Densidade do vinho (g/cm³)",
        "pH": "Nível de pH do vinho",
        "sulphates": "Sulfatos adicionados (ex: sulfato de potássio) (g/dm³)",
        "alcohol": "Teor alcoólico (% vol.)",
        "quality": "Qualidade do vinho (pontuação de 0 a 10, baseada em dados sensoriais)"
    },
    "water_potability.csv": {
        "ph": "Valor do pH da água. Importante para avaliar o equilíbrio ácido-base. Faixa recomendada pela OMS: 6.5 a 8.5.",
        "Hardness": "Dureza da água, causada principalmente por sais de cálcio e magnésio dissolvidos.",
        "Solids": "Sólidos totais dissolvidos (TDS) (mg/L). Mede minerais e sais dissolvidos. Limite desejável: 500, máximo: 1000.",
        "Chloramines": "Cloraminas (mg/L ou ppm). Desinfetante comum. Níveis até 4 mg/L considerados seguros.",
        "Sulfate": "Sulfato (mg/L). Substância natural. Comum em água doce: 3-30 mg/L.",
        "Conductivity": "Condutividade elétrica (μS/cm). Mede a capacidade da água de transmitir corrente. Padrão OMS: < 400 μS/cm.",
        "Organic_carbon": "Carbono Orgânico Total (TOC) (mg/L). Mede carbono em compostos orgânicos. Padrão US EPA: < 2 mg/L (tratada), < 4 mg/L (fonte).",
        "Trihalomethanes": "Trialometanos (THMs) (ppm). Formados no tratamento com cloro. Níveis até 80 ppm considerados seguros.",
        "Turbidity": "Turbidez (NTU). Mede matéria sólida suspensa. Recomendado OMS: < 5.00 NTU.",
        "Potability": "Potabilidade. Indica se a água é segura para consumo humano (1 = Potável, 0 = Não Potável)."
    },
    "vertebralcolumn-2C.csv": {
        "pelvic_incidence": "Incidência pélvica",
        "pelvic_tilt": "Inclinação pélvica",
        "lumbar_lordosis_angle": "Ângulo da lordose lombar",
        "sacral_slope": "Inclinação sacral",
        "pelvic_radius": "Raio pélvico",
        "degree_spondylolisthesis": "Grau de espondilolistese",
        "class": "Classe do paciente (Normal ou Anormal)"
    },
    "Vehicle.csv": {
        "Comp": "Compactness (Compacidade)",
        "Circ": "Circularity (Circularidade)",
        "D.Circ": "Distance Circularity (Circularidade da Distância)",
        "Rad.Ra": "Radius Ratio (Razão do Raio)",
        "Pr.Axis.Ra": "Pr.Axis Aspect Ratio (Razão de Aspecto do Eixo Principal)",
        "Max.L.Ra": "Max.Length Aspect Ratio (Razão de Aspecto do Comprimento Máximo)",
        "Scat.Ra": "Scatter Ratio (Razão de Dispersão)",
        "Elong": "Elongatedness (Alongamento)",
        "Pr.Axis.Rect": "Pr.Axis Rectangularity (Retangularidade do Eixo Principal)",
        "Max.L.Rect": "Max.Length Rectangularity (Retangularidade do Comprimento Máximo)",
        "Sc.Var.Maxis": "Scaled Variance along Major Axis (Variância Escalada - Eixo Maior)",
        "Sc.Var.maxis": "Scaled Variance along Minor Axis (Variância Escalada - Eixo Menor)",
        "Ra.Gyr": "Scaled Radius of Gyration (Raio de Giração Escalado)",
        "Skew.Maxis": "Skewness about Major Axis (Assimetria sobre Eixo Maior)",
        "Skew.maxis": "Skewness about Minor Axis (Assimetria sobre Eixo Menor)",
        "Kurt.maxis": "Kurtosis about Minor Axis (Curtose sobre Eixo Menor)",
        "Kurt.Maxis": "Kurtosis about Major Axis (Curtose sobre Eixo Maior)",
        "Holl.Ra": "Hollows Ratio (Razão de Cavidades)",
        "Class": "Classe do Veículo (van, saab, bus, opel)"
    },
    "spotify-2023.csv": {
        "track_name": "Nome da música.",
        "artist(s)_name": "Nome do(s) artista(s) ou banda(s).",
        "artist_count": "Número de artistas na música.",
        "released_year": "Ano de lançamento.",
        "released_month": "Mês de lançamento.",
        "released_day": "Dia de lançamento.",
        "in_spotify_playlists": "Número de playlists do Spotify em que a música está.",
        "in_spotify_charts": "Número de paradas do Spotify em que a música apareceu.",
        "streams": "Número total de streams no Spotify (até data específica).",
        "in_apple_playlists": "Número de playlists da Apple Music em que a música está.",
        "in_apple_charts": "Número de paradas da Apple Music em que a música apareceu.",
        "in_deezer_playlists": "Número de playlists do Deezer em que a música está.",
        "in_deezer_charts": "Número de paradas do Deezer em que a música apareceu.",
        "in_shazam_charts": "Número de paradas do Shazam em que a música apareceu.",
        "bpm": "Beats Per Minute (Batidas Por Minuto) - Tempo da música.",
        "key": "Tonalidade principal da música.",
        "mode": "Modalidade da música (Major/Minor).",
        "danceability_%": "Quão adequada para dançar (0-100%).",
        "valence_%": "Positividade musical (0-100%).",
        "energy_%": "Nível de energia percebido (0-100%).",
        "acousticness_%": "Quão acústica é a música (0-100%).",
        "instrumentalness_%": "Probabilidade de ser instrumental (0-100%).",
        "liveness_%": "Presença de audiência / ao vivo (0-100%).",
        "speechiness_%": "Presença de palavras faladas (0-100%)."
    },
    "prod_petroleo.csv": {
        "Ano": "Ano da observação",
        "PRODUÇÃO 1": "Produção de petróleo (nota 1)",
        "IMPORTAÇÃO 2": "Importação de petróleo (nota 2)",
        "EXPORTAÇÃO": "Exportação de petróleo",
        "VARIAÇÃO DE ESTOQUES": "Variação nos estoques de petróleo",
        "CONSUMO TOTAL": "Consumo total de petróleo",
        "TRANSFORMAÇÃO 2": "Volume de petróleo destinado à transformação/refino (nota 2)"
    },
    "opsd_germany_daily.csv": {
        "Date": "Data (AAAA-MM-DD)",
        "Consumption": "Consumo total de eletricidade (GWh)",
        "Wind": "Produção líquida de eletricidade eólica (GWh)",
        "Solar": "Produção líquida de eletricidade solar (GWh)",
        "Wind+Solar": "Soma da produção líquida eólica e solar (GWh)"
    },
    "mystocksn.csv": {
        "data": "Data da cotação (AAAA-MM-DD)",
        "IBOV": "Valor do índice IBOVESPA (B3)",
        "VALE3": "Preço da ação VALE3 (Vale S.A.)",
        "PETR4": "Preço da ação PETR4 (Petrobras PN)",
        "DOLAR": "Taxa de câmbio BRL/USD (Real por Dólar)"
    },
    "mtcars.csv": {
        "model": "Modelo do carro",
        "mpg": "Milhas por Galão (consumo)",
        "cyl": "Número de cilindros",
        "disp": "Cilindrada (polegadas cúbicas)",
        "hp": "Potência bruta (horsepower)",
        "drat": "Relação do eixo traseiro",
        "wt": "Peso (1000 lbs)",
        "qsec": "Tempo para 1/4 de milha (segundos)",
        "vs": "Motor (0 = V, 1 = reto)",
        "am": "Transmissão (0 = automática, 1 = manual)",
        "gear": "Número de marchas",
        "carb": "Número de carburadores"
    },
    "dados_banco.csv": {
        "ID_Linha": "ID sequencial da linha",
        "Cliente": "ID do cliente",
        "Sexo": "Sexo do cliente (M/F)",
        "Idade": "Idade do cliente (anos)",
        "Empresa": "Tipo de vínculo empregatício (Privada, Autônomo, Pública)",
        "Salario": "Salário do cliente (R$)",
        "Saldo_cc": "Saldo em conta corrente (R$)",
        "Saldo_poupança": "Saldo em conta poupança (R$)",
        "Saldo_investimento": "Saldo em investimentos (R$)",
        "Devedor_cartao": "Valor devido no cartão de crédito (R$)",
        "Inadimplente": "Status de inadimplência (0 = Não, 1 = Sim)"
    },
    "energia_socioeconomia.csv": {
        "Ano": "Ano da observação",
        "OFERTA INTERNA DE ENERGIA - OIE": "Oferta Interna de Energia (10⁶ tep)",
        "OFERTA INT. ENERGIA ELÉTRICA-OIEE": "Oferta Interna de Energia Elétrica (GWh)",
        "PRODUTO INTERNO BRUTO - PIB": "Produto Interno Bruto (10⁹ US$ PPC 2010)",
        "POPULAÇÃO RESIDENTE - POP": "População residente (10⁶ habitantes)",
        "OIE/PIB": "Intensidade Energética (OIE / PIB) (tep/10³ US$)",
        "OIE/POP": "Consumo de Energia per capita (OIE / POP) (tep/hab)",
        "OIEE/POP": "Consumo de Energia Elétrica per capita (OIEE / POP) (KWh/hab)"
    },
    "houses_to_rent_v2.csv": {
        "city": "Cidade onde o imóvel está localizado",
        "area": "Área do imóvel (m²)",
        "rooms": "Número de quartos",
        "bathroom": "Número de banheiros",
        "parking spaces": "Número de vagas de estacionamento",
        "floor": "Andar do imóvel ('-' se não aplicável)",
        "animal": "Aceita animais (acept/not acept)",
        "furniture": "Mobiliado (furnished/not furnished)",
        "hoa (R$)": "Valor do condomínio (R$)",
        "rent amount (R$)": "Valor do aluguel (R$)",
        "property tax (R$)": "Valor do IPTU (R$)",
        "fire insurance (R$)": "Valor do seguro incêndio (R$)",
        "total (R$)": "Valor total mensal (Aluguel + Condomínio + IPTU + Seguro) (R$)"
    },
    "concrete_data.csv": {
        "cement": "Cimento (kg/m³)",
        "blast_furnace_slag": "Escória de alto-forno (kg/m³)",
        "fly_ash": "Cinza volante (kg/m³)",
        "water": "Água (kg/m³)",
        "superplasticizer": "Superplastificante (kg/m³)",
        "coarse_aggregate": "Agregado graúdo (kg/m³)",
        "fine_aggregate ": "Agregado fino (kg/m³)",
        "age": "Idade do concreto (dias)",
        "concrete_compressive_strength": "Resistência à compressão do concreto (MPa)"
    },
    "CompanhiaMB.csv": {
        "funcionario": "ID do funcionário",
        "estado_civil": "Estado civil do funcionário",
        "instrucao": "Nível de instrução/escolaridade",
        "nfilhos": "Número de filhos (NA se não aplicável)",
        "salario": "Salário (unidade não especificada)",
        "idade_anos": "Idade em anos",
        "idade_meses": "Meses adicionais da idade",
        "regiao": "Região de origem/trabalho (interior, capital, outro)"
    },
    "BreastCancer.csv": {
        "Id": "Número de identificação da amostra",
        "Cl.thickness": "Espessura do Aglomerado (1-10)",
        "Cell.size": "Uniformidade do Tamanho da Célula (1-10)",
        "Cell.shape": "Uniformidade da Forma da Célula (1-10)",
        "Marg.adhesion": "Adesão Marginal (1-10)",
        "Epith.c.size": "Tamanho da Célula Epitelial Única (1-10)",
        "Bare.nuclei": "Núcleos Nus (1-10, pode ter NA)",
        "Bl.cromatin": "Cromatina Branda (1-10)",
        "Normal.nucleoli": "Nucléolos Normais (1-10)",
        "Mitoses": "Mitoses (1-10)",
        "Class": "Classe do tumor (0 = benigno, 1 = maligno)"
    }
}

Planilhas = os.listdir(path='./Planilhas')

Planilha = st.sidebar.selectbox('Selecione a planilha', Planilhas)

# Carregar o dataset com tratamento de encoding e separador
file_path = os.path.join('./Planilhas', Planilha)
df = None
error_reading = False
used_encoding = None
used_separator = ',' # Assume vírgula por padrão

try:
    # Tenta ler com UTF-8 e separador vírgula
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    used_encoding = 'utf-8'
    # Verifica se interpretou como uma única coluna (indício de separador errado)
    if len(df.columns) == 1:
        st.warning(f"Arquivo \'{Planilha}\' lido com UTF-8 resultou em uma coluna. Tentando separador \';\'...")
        try:
            df_semicolon = pd.read_csv(file_path, encoding='utf-8', sep=';')
            # Só atualiza se a leitura com ; resultou em mais de uma coluna
            if len(df_semicolon.columns) > 1:
                df = df_semicolon
                used_separator = ';'
                # st.success(f"Arquivo \'{Planilha}\' lido com sucesso usando UTF-8 e separador \';\'.")
            else:
                pass
                #  st.warning(f"Leitura com separador \';\' também resultou em uma coluna. Mantendo leitura original.")
        except Exception as e_sep:
            # st.warning(f"Falha ao tentar ler \'{Planilha}\' com separador \';\' (UTF-8): {e_sep}. Mantendo leitura original.")
            pass

except UnicodeDecodeError:
    # st.warning(f"Falha ao decodificar \'{Planilha}\' com UTF-8. Tentando com Latin-1...")
    try:
        # Tenta ler com Latin-1 e separador vírgula
        df = pd.read_csv(file_path, encoding='latin-1', sep=',')
        used_encoding = 'latin-1'
        # st.success(f"Arquivo \'{Planilha}\' lido preliminarmente com Latin-1 e separador \',\'.")
        # Verifica se interpretou como uma única coluna
        if len(df.columns) == 1:
            # st.warning(f"Arquivo \'{Planilha}\' lido com Latin-1 resultou em uma coluna. Tentando separador \';\'...")
            try:
                df_semicolon = pd.read_csv(file_path, encoding='latin-1', sep=';')
                if len(df_semicolon.columns) > 1:
                    df = df_semicolon
                    used_separator = ';'
                    # st.success(f"Arquivo \'{Planilha}\' lido com sucesso usando Latin-1 e separador \';\'.")
                else:
                    pass
                    # st.warning(f"Leitura com separador \';\' também resultou em uma coluna. Mantendo leitura original.")
            except Exception as e_sep:
                pass
                # st.warning(f"Falha ao tentar ler \'{Planilha}\' com separador \';\' (Latin-1): {e_sep}. Mantendo leitura original.")

    except Exception as e:
        pass
        # st.error(f"Falha ao ler o arquivo \'{Planilha}\' com Latin-1. Erro: {e}")
        error_reading = True
except Exception as e:
    st.error(f"Erro inesperado ao ler o arquivo \'{Planilha}\': {e}")
    error_reading = True

# Se houve erro na leitura ou df não foi carregado, para a execução
if error_reading or df is None:
    st.error(f"Não foi possível carregar os dados do arquivo \'{Planilha}\'. Verifique o arquivo e tente novamente.")
    # st.stop() # Garante que o script pare se o df não for carregado
else:
    st.info(f"Arquivo \'{Planilha}\' carregado inicialmente com encoding \'{used_encoding}\' e separador \'{used_separator}\'.")

    # Tratamento específico para arquivos como 'energia_socioeconomia.csv'
    # que têm uma linha de unidades logo após o cabeçalho.
    if Planilha == "energia_socioeconomia.csv":
        # st.info(f"Aplicando tratamento específico para '{Planilha}': pulando linha de unidades e ajustando nome da primeira coluna.")
        try:
            # Reler o arquivo, pulando a segunda linha (índice 1 do arquivo original) que contém as unidades.
            # A primeira linha do arquivo (índice 0) é usada como cabeçalho.
            df_adjusted = pd.read_csv(file_path, encoding=used_encoding, sep=used_separator, skiprows=[1])
            
            # Se a primeira coluna do DataFrame lido (após pular a linha de unidades)
            # tiver um nome genérico como "Unnamed: 0" (devido ao ';' inicial no cabeçalho do CSV)
            # e houver colunas, tentar renomeá-la para o nome esperado do dicionário.
            if not df_adjusted.empty and len(df_adjusted.columns) > 0 and df_adjusted.columns[0].startswith('Unnamed:'):
                current_columns = df_adjusted.columns.tolist()
                expected_first_col_name = None
                # Busca o primeiro nome de coluna definido em descricao_colunas para esta planilha
                if Planilha in descricao_colunas and descricao_colunas[Planilha]:
                    expected_col_names_list = list(descricao_colunas[Planilha].keys())
                    if expected_col_names_list:
                        expected_first_col_name = expected_col_names_list[0]
                
                if expected_first_col_name:
                    # st.info(f"Para '{Planilha}', renomeando primeira coluna de '{current_columns[0]}' para '{expected_first_col_name}'.")
                    current_columns[0] = expected_first_col_name
                    df_adjusted.columns = current_columns
            
            df = df_adjusted # Atualiza o df principal com a versão ajustada
            st.success(f"Arquivo '{Planilha}' reprocessado com sucesso para pular linha de unidades e ajustar cabeçalho.")
        except Exception as e_specific_treat:
            st.warning(f"Falha ao aplicar tratamento específico para '{Planilha}': {e_specific_treat}. O DataFrame carregado inicialmente será usado, mas pode conter uma linha de unidades como dados ou ter cabeçalho incorreto.")

# Visualizar as primeiras linhas do dataset
st.title(f'Análise do Dataset: {Planilha}')
st.write('### Visualização das primeiras linhas do dataset')
st.write(df.head())
st.write('### Visualização dos tipos de dados das colunas')
st.write(df.dtypes)
# Selecionar colunas para análise
st.sidebar.write('### Selecione as colunas para análise X e Y')
all_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

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



# Botão para iniciar a análise

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
x_col = st.sidebar.selectbox(
    'Selecione a coluna para X:',
    all_columns,
    index=default_x_index,
    key='x_col_select',
    placeholder='Selecione a coluna para X',
    help='Selecione a coluna para X',
    
)
y_col = st.sidebar.selectbox(
    'Selecione a coluna para Y:',
    all_columns,
    index=default_y_index,
    key='y_col_select',
    placeholder='Selecione a coluna para Y',
    help='Selecione a coluna para Y'
)
run_analysis_button = st.sidebar.button("Realizar Análise")

st.write(f"""
# As variaveis que escolhi para analise sao:
## {x_col} e {y_col}
- {x_col}: {descricao_colunas[Planilha][x_col.strip()]}
- {y_col}: {descricao_colunas[Planilha][y_col.strip()]}
""")
st.sidebar.write(f"""
# X: {x_col} e Y: {y_col}
- {x_col}: {descricao_colunas[Planilha][x_col.strip()]}
- {y_col}: {descricao_colunas[Planilha][y_col.strip()]}
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
    st.write(f'Correlação pelo numpy: {np.corrcoef(df[x_col], df[y_col])[0, 1]:.4f}')

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



