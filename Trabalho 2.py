#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np #Bibliotecas necessárias
import matplotlib.pyplot as plt
import pandas as pd


# In[251]:


#Importado do jupyter notebook

serie = pd.read_csv('AirPassengers.csv') #Dataset dos passageiros mensais de linhas aéreas nos EUA
df = np.diff(np.log(serie['#Passengers']).rolling(2).mean()) #Transforma a série em estacionária
df = df[1:] #Retira o primeiro valor que foi descartado na derivada da série
vet = df
lin = np.linspace(0,0.2, 142) #Cria os valores no eixo x

##Cálculo dos Kernels##

def kernel(x,h): #Função que calcula a função kernel para cada ponto da série
    matrix = np.zeros((len(x) + 1,len(lin))) #Cria matriz para armazenar todos os kernels
    for k in np.arange(len(lin)): #Guarda os valores do eixo x na linha 1
        matrix[0][k] = lin[k] 
    col = 1 #Variável para iterar nas colunas da matriz
    for j in x:
        row_lin = 0 #Variável para iterar nas linhas da matriz
        for i in lin: #Calcula os valores dentro da janela para o kernel
            u = 0
            u = (i - j)/h #Filtro na janela do kernel
            if abs(u) <= 1: #Retorna Epanechnikov se u >= 1
                matrix[col][row_lin] = ((3/4)*(1 - u*u)/(h*
                                       len(lin)))
            row_lin += 1
        col += 1
    return matrix #Retorna a matriz com todas as funções kernel
janela = 0.04 #Tamanho da janela do kernel

##Plota dos kernels independentes##

#for i in np.arange(1,len(vet) + 1): #Plota os kernels para cada ponto
    #plt.plot(kernel(vet, janela)[0], kernel(vet, janela)[i])
#    plt.plot(kernel(vet, janela)[0], kernel(vet, janela)[i])
#plt.plot(kernel(vet, janela)[0], kernel(vet, janela)[1],label="K(x=3)")
#plt.plot(kernel(vet, janela)[0], kernel(vet, janela)[2],label="K(x=4)")
#plt.plot(kernel(vet, janela)[0], kernel(vet, janela)[3],label="K(x=4)")

kern = kernel(vet, janela) #Matriz que guarda os kernels
plot_sum = 0

for k in np.arange(1, len(vet)): #Soma todos os kernel para gerar a distribuição
    plot_sum += kern[k] + kern[k + 1]

##Plot da soma dos kernels##    

plt.plot(kernel(vet, janela)[0], plot_sum, label='Densidade') #Plota a distribuição
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")
plt.legend()
plt.xlim(0,0.2)
plt.savefig("Densidade_Epanechnikov.png", dpi=150)


# In[257]:


##Plot da série temporal##

#plt.plot(df, label="Série temporal estacionária")
#plt.xlabel("Eixo x")
#plt.ylabel("Eixo y")
#plt.legend()
#plt.ylim(-.18,0.18)
#plt.savefig("Serie_temporal.png", dpi=150)


# In[275]:


serie = pd.read_csv('AirPassengers.csv') #Dataset dos passageiros mensais de linhas aéreas nos EUA
df = np.diff(np.log(serie['#Passengers']).rolling(2).mean()) #Transforma a série em estacionária
df = df[1:] #Retira o primeiro valor que foi descartado na derivada da série
x = np.linspace(1,len(df),142) #Cria os valores no eixo x

y_wind = [] #Multiplica a série pela janela de Hann
for i in np.arange(len(x)):
    y_wind.append(df[i]*np.power(np.sin(i*len(x)/(len(x) - 1)),2)) #Multiplicação pela janela de Hann
y_fft = list(abs(np.fft.fft(y_wind))) #Calcula a DFT e guarda os valores na lista

div = int((np.fft.fftfreq(len(df),d=1)*2*np.pi).size/2) #Operação para deixar os valores calculados pela classe de DFT em ordem
axis = list(np.fft.fftfreq(len(df),d=1)*2*np.pi)

for i in np.arange(div): #Coloca os valore em ordem
    y_fft.append(y_fft[i])
    axis.append(axis[i])
plt.plot(axis[div:],y_fft[div:]) #Plota da DFT e as configurações do plot
plt.xticks(np.arange(-3,3.5,0.5))
plt.xlabel("Frequência(Hz)")
plt.ylabel("Amplitude(F(w))")
#plt.savefig("Transformada.png", dpi=150)


# In[ ]:




