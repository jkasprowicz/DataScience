import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as st

#display all columns
pd.set_option("Display.max_columns", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#acessando o arquivo e explorando os dados, estatistica.
file = pd.read_csv("/Users/joaokasprowicz/Desktop/dados.csv")

#Apenas os 5 primeiras linhas, incluindo o header.
print("Head".center(70,"~"))
print(file.head())


#descobrir padrões
#Utilizando o metodo describe do pandas, permite acesso a uma boa parte da estatistica descritiva.
print("Describe".center(70,"~"))
print(file.describe())


#data types
print("Data Types".center(70,"~"))
print(file.info())

#detectar anomalias
#Nas exploração do conjunto de dados, é importante lidar com os missing values.
#bem como, os valores errados.....
print("Missing values".center(70,"~"))
file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']]=file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']].replace(0,np.NaN)
print(file.isna().sum())

#DISTRUIBUIÇÃO DOS DADOS.
#Conseguimos verificar visualizando os graficos que temos, valores zerados, e alguns valores errados.
#por isso é importante antes de olhar o dataset, criar uma familiaridade com os dados.
fig, ax = plt.subplots( 4, 2, figsize=(14,15))
sns.histplot(file.Age,kde=True,ax=ax[0,0])
sns.histplot(file.Pregnancies,kde=True,  ax=ax[0,1])
sns.histplot(file.Glucose,kde=True,  ax=ax[1,0])
sns.histplot(file.BloodPressure,kde=True,  ax=ax[1,1])
sns.histplot(file.SkinThickness,kde=True,  ax=ax[2,0])
sns.histplot(file.Insulin,kde=True,  ax=ax[2,1])
sns.histplot(file.DiabetesPedigreeFunction,kde=True, ax=ax[3,0])
sns.histplot(file.BMI,kde=True, ax=ax[3,1])
plt.show()

#alterando os missing values (NaN values) com a media.
file['Glucose'].fillna(file['Glucose'].mean(), inplace=True)
file['BloodPressure'].fillna(file['BloodPressure'].mean(), inplace=True)
file['SkinThickness'].fillna(file['SkinThickness'].mean(), inplace=True)
file['Insulin'].fillna(file['Insulin'].mean(), inplace=True)
file['BMI'].fillna(file['BMI'].mean(), inplace=True)

print("PORCENTAGEM POR RESULTADO".center(70,"~"))
print(file['Outcome'].value_counts(normalize=True))
print("MEDIA AGRUPADO POR RESULTADO".center(70,"~"))
print(file.groupby(['Outcome']).mean())

print("correlaçao de variaveis".center(70,"~"))
file_cor = file[['Insulin','Glucose','Age','Pregnancies','BloodPressure','BMI']].dropna().corr()
print(file_cor)

#Analisando correlação de variaveis com auxilio de um heatmap
sns.heatmap(file_cor, annot=True)
plt.show()

#testar hipoteses e verificar suposições
#teste de hipotese
#primeiro passo, determinar a hipotese.
#segundo passo, escolha do teste estatistico apropriado.
#terceiro passo, Calcular o p valor.
#Determinar a significância estatística.

#Já que um dos datasets (amostras) não tem distrubuição normalizada, devemos escolher um teste não parametrico.

#O teste de Kruskal-Wallis por postos, teste H de Kruskal-Wallis ou análise de variância de um fator em postos é um método não paramétrico para testar se amostras se originam da mesma distribuição.
#É usado para comparar duas ou mais amostras independentes de tamanhos iguais ou diferentes.


#Hipoteses
# A media de glicose muda com o numero de gravidez se mulher é diabetica. (H0)
# A media de glicose não muda de acordo com o numero de gravidez se a mulher é diabetico. (Alternativa)

print("Inferência estatística".center(70,"~"))
#dividindo os grupos positivos e negativo para inferencia estatistica
positive_df=file[file['Outcome']==1]
negative_df=file[file['Outcome']==0]

#dividindo as amostras populacionais de pessoa diabeticas de acordo com o numero de vezes gravidas
preg_0=positive_df[positive_df['Pregnancies']==0]
preg_1=positive_df[positive_df['Pregnancies']==1]
preg_2=positive_df[positive_df['Pregnancies']==2]
preg_3=positive_df[positive_df['Pregnancies']==3]
preg_4=positive_df[positive_df['Pregnancies']==4]
preg_5=positive_df[positive_df['Pregnancies']==5]
preg_6=positive_df[positive_df['Pregnancies']==6]
preg_7=positive_df[positive_df['Pregnancies']==7]
preg_8=positive_df[positive_df['Pregnancies']==8]
preg_9=positive_df[positive_df['Pregnancies']==9]
preg_10=positive_df[positive_df['Pregnancies']==10]
preg_11=positive_df[positive_df['Pregnancies']==11]
preg_12=positive_df[positive_df['Pregnancies']==12]
preg_13=positive_df[positive_df['Pregnancies']==13]
preg_14=positive_df[positive_df['Pregnancies']==14]
preg_15=positive_df[positive_df['Pregnancies']==15]
preg_16=positive_df[positive_df['Pregnancies']==16]
preg_17=positive_df[positive_df['Pregnancies']==17]

#loop que intera com cada amostra, verificando a distribuição.
for i in range(18):
    p=positive_df[positive_df['Pregnancies']==i]['Glucose']
    if st.shapiro(p)[1]>0.95 or st.shapiro(p)[1]<0.05:
        print(st.shapiro(p))
        break

# executando o teste estatistico escolhido, uma vez que os dados demonstraram ser não normalizados, de acordo com:
# o resultado do teste de Shapiro-Wilk < 0,05.

#uma vez que, temos dados não normalizados, devemos escolher testes estatisticos não parametricos.
#para verificar a hipotese levantada, escolhi Kruskal-wallis

print(st.kruskal(preg_0['Glucose'],preg_1['Glucose'],preg_2['Glucose'],preg_3['Glucose'],preg_4['Glucose'],preg_5['Glucose'],preg_6['Glucose'],preg_7['Glucose'],preg_8['Glucose'],preg_9['Glucose'],preg_10['Glucose'],preg_11['Glucose'],preg_12['Glucose'],preg_13['Glucose'],preg_14['Glucose'],preg_15['Glucose'],preg_17['Glucose']))

#Since pvalue>0.05 test stastic falls in the non-rejection region.
#Conclusion: Average glucose levels does not change with pregnancies if a woman is diabetic

# O resultado do p valor maior que > 0.05 resultado na não rejeição da hipotese.
# concluido que os niveis medios de glicose não demonstram mudanças estatisticas significantes com gravidez se a mulher é diabetica.

plt.figure(figsize=(10,10))
ax=sns.lineplot(x='Pregnancies',y='Glucose',data=file,hue='Outcome',style='Outcome',markers=True)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
plt.title('Effect of Pregnancies on Glucose')
plt.show()

#O grafico mostra niveis mais altos de glicose em mulheres diabeticas, mas não existe padrão visivel para gravidez.

