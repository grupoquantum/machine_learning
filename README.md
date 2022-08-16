<div align="justify">
<p align="justify">
Machine Learning (ML) ou Aprendizado de Máquina consiste em uma série de técnicas matemáticas e algoritmos computacionais que tem como objetivo a construção e aplicação do conhecimento através do reconhecimento de padrões. Os algoritmos de Aprendizado de Máquina tem caráter analítico e preditivo, o que quer dizer que eles podem ser usados para descrever ou prever dados. São utilizados principalmente em Data Science (DS) ou Ciência de Dados, na Estatística, Probabilidade, Economia e demais áreas do conhecimento que tenham interesse particular na previsão de cenários futuros com base em históricos passados.  Os algoritmos de Machine Learning pertencem a classe de algoritmos de Inteligência Artificial (IA), porém diferente do Deep Learning (DL) ou Aprendizado Profundo eles não se utilizam de estruturas matemáticas baseadas na anatomia neuronal, mas sim em cálculos puramente estatísticos. Podem ser divididos em três grupos principais separados por tipo de predição, são eles: Classificação, Regressão e Agrupamento/Clusterização.
</p>
<br>
<p align="justify">
Classificação: são algoritmos que tem a proposta de realizar predições dentro de um número limitado de alternativas, ou seja, as respostas que serão retornadas como resultado serão sempre uma das respostas já contidas no histórico de dados. Confira a seguir um exemplo desse tipo de algoritmo, nesse caso uma classificação por Árvore de Decisão (Decision Tree).
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.decision_tree import DecisionTree
decision_tree = DecisionTree()
''' classificação de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
decision_tree.fit(
	inputs=inputs, # entradas de exemplo
	outputs=outputs, # saídas de exemplo
	extra_trees=False # se True acrescenta árvores extras para aumentar o nível de precisão, porém o processamento ficará mais lento
)
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]] # novas entradas para predizer
new_outputs = decision_tree.predict(inputs=new_inputs) # método de predição
print(new_outputs) # exibe as novas saídas com base nas novas entradas 
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]  
  </code>
</pre>
<br>
<p align="justify">
Na Árvore de Decisão os dados de treinamento (fase da assimilação do conhecimento) serão organizados internamente em uma estrutura em forma de uma árvore de alternativas semelhante por exemplo às utilizadas na representação de hierarquias de cargo dentro de uma empresa onde temos os presidente executivo no topo, os gerentes abaixo e os gerenciados em um nível inferior da árvore. Uma variação deste algoritmo é a Floresta Aleatória ou Random Forest que ao invés de utilizar uma única árvore trabalha com um conjunto delas onde cada uma dessas árvores será composta por combinações diferentes dos dados de entrada escolhidos aleatoriamente.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.random_forest import RandomForest
random_forest = RandomForest()
''' classificação de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
random_forest.fit(
	inputs=inputs, # entradas de exemplo
	outputs=outputs, # saídas de exemplo
	number_of_trees=3 # número relativo a quantidade de árvores na floresta
)
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]] # novas entradas para predizer
new_outputs = random_forest.predict(inputs=new_inputs) # método de predição
print(new_outputs) # exibe as novas saídas com base nas novas entradas  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]  
  </code>
</pre>
<br>
<p align="justify">
No algoritmo de Floresta Aleatória teremos um grau de confiabilidade maior nos resultados porém como utilizamos várias árvores trabalhando em conjunto a performance poderá ser sacrificada o tornando mais lento do que um algoritmo de Árvore de Decisão comum.<br>
Também existem algoritmos de Aprendizado de Máquina capazes de emitir resultados classificativos probabilísticos, ou seja, resultados que ao invés de retornarem o rótulo da classe de resposta propriamente dito retornam a probabilidade de uma determinada entrada pertencer a uma determinada classe. A seguir um exemplo desse tipo de algoritmo com o Naive Bayes:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.naive_bayes import NaiveBayes
naive_bayes = NaiveBayes()
''' classificação de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
naive_bayes.fit(
	inputs=inputs, # entradas de exemplo
	outputs=outputs, # saídas de exemplo
	classify=False # se definido como True irá retornar a classe com o maior percentual probabilístico
)
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]] # novas entradas para predizer
new_outputs = naive_bayes.predict(inputs=new_inputs) # método de predição
print(new_outputs) # exibe as novas saídas com base nas novas entradas  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[{'unidades': 1.0, 'dezenas': 0.0, 'centenas': 0.0, 'classify': 'unidades'}, {'unidades': 0.0, 'dezenas': 1.0, 'centenas': 0.0, 'classify': 'dezenas'}, {'unidades': 0.0, 'dezenas': 0.0, 'centenas': 1.0, 'classify': 'centenas'}, {'unidades': 1.0, 'dezenas': 0.0, 'centenas': 0.0, 'classify': 'unidades'}, {'unidades': 0.0, 'dezenas': 1.0, 'centenas': 0.0, 'classify': 'dezenas'}, {'unidades': 0.0, 'dezenas': 0.0, 'centenas': 1.0, 'classify': 'centenas'}, {'unidades': 1.0, 'dezenas': 0.0, 'centenas': 0.0, 'classify': 'unidades'}, {'unidades': 0.0, 'dezenas': 1.0, 'centenas': 0.0, 'classify': 'dezenas'}, {'unidades': 0.0, 'dezenas': 0.0, 'centenas': 1.0, 'classify': 'centenas'}]  
  </code>
</pre>
<br>
<p align="justify">
O Naive Bayes ou (Bayes Ingênuos) que recebe o nome de ingênuo por assumir previamente que existe uma forte correlação entre os dados, foi criado com base no famoso teorema do matemático britânico Thomas Bayes que é conhecido como Teorema de Bayes. Na fórmula fazemos um cálculo simples que calcula a probabilidade de uma entrada pertencer a uma determinada classe simplesmente calculando a probabilidade dessa mesma classe pertencer a essa mesma entrada multiplicada pela probabilidade da entrada pertencer a qualquer uma das classes, dividindo esse produto pela probabilidade da classe pertencer a qualquer uma das entradas. Veja abaixo a fórmula do teorema:
</p>
<p align="center"><img src="https://github.com/aiquantumneuro/machine_learning/blob/main/teorema_de_bayes.jpg"></p>
<p align="justify">
Uma variação desse tipo de resultado pode ser conseguida através do algoritmo K-Nearest Neighbors Probabilístico que calcula a distância euclidiana do KNN através de respostas probabilísticas.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.probabilistic_knn import ProbabilisticKNN
probabilistic_knn = ProbabilisticKNN()
''' classificação de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
probabilistic_knn.fit(
	inputs=inputs, # entradas de exemplo
	outputs=outputs, # saídas de exemplo
	k=0 # número referente a quantidade de vizinhos mais próximos
)
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]] # novas entradas para predizer
new_outputs = probabilistic_knn.predict(inputs=new_inputs) # método de predição
print(new_outputs) # exibe as novas saídas com base nas novas entradas  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[{'unidades': 0.7142493576959376, 'dezenas': 0.25610890262210656, 'centenas': 0.02964173968195583}, {'dezenas': 0.5552164261931187, 'unidades': 0.24569922308546058, 'centenas': 0.19908435072142064}, {'centenas': 0.668614619570311, 'dezenas': 0.29588118222364335, 'unidades': 0.03550419820604564}, {'unidades': 0.6210398212171975, 'dezenas': 0.3355409371134971, 'centenas': 0.043419241669305296}, {'dezenas': 0.560831795471936, 'centenas': 0.30301120763379397, 'unidades': 0.13615699689427016}, {'centenas': 0.7875877715976531, 'dezenas': 0.19120810666796956, 'unidades': 0.021204121734377447}, {'unidades': 0.5583278677310599, 'dezenas': 0.3865635162821316, 'centenas': 0.05510861598680854}, {'dezenas': 0.536910446017243, 'centenas': 0.371734964232535, 'unidades': 0.09135458975022197}, {'centenas': 0.8414340668336251, 'dezenas': 0.14316887396707126, 'unidades': 0.015397059199303604}]  
  </code>
</pre>
<br>
<p align="justify">
Outra forma de se conseguir resultados probabilísticos porém com base na probabilidade total e não na probabilidade comparativa como o algoritmo anterior é através do Processo Gaussiano ou Gaussian Process que aplica uma distribuição gaussiana aos dados para se ter uma resposta probabilística. Veja um exemplo a seguir:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.gaussian_processes import GaussianProcesses
gaussian_processes = GaussianProcesses()
''' classificação de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
gaussian_processes.fit(inputs=inputs, outputs=outputs) # método de treinamento
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]] # novas entradas para predizer
new_outputs = gaussian_processes.predict(inputs=new_inputs) # método de predição
print(new_outputs) # exibe as novas saídas com base nas novas entradas  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[{'classification': ['unidades'], 'probability': 0.5833333333333334}, {'classification': ['dezenas'], 'probability': 0.5833333333333334}, {'classification': ['centenas'], 'probability': 0.5833333333333334}, {'classification': ['unidades'], 'probability': 0.775}, {'classification': ['dezenas'], 'probability': 0.775}, {'classification': ['centenas'], 'probability': 0.775}, {'classification': ['unidades'], 'probability': 0.8452380952380952}, {'classification': ['dezenas'], 'probability': 0.8452380952380952}, {'classification': ['centenas'], 'probability': 0.8452380952380952}]  
  </code>
</pre>
<br>
<p align="justify">
Também conseguimos emitir resultados com mais de um rótulo por classe como acontece com o algoritmo K-Nearest Neighbors Multilabel ou Multilabel KNN. Esse algoritmo simplesmente aplica o KNN retornando os rótulos mais possíveis para uma determinada entrada, dessa forma poderemos ter entradas pertencentes a mais de um rótulo.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.multilabel_knn import MultilabelKNN
multilabel_knn = MultilabelKNN()
''' classificação de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2, 3], [10, 20, 30], [100, 200, 300], [2, 3, 4], [20, 30, 40], [200, 300, 400], [5, 6, 7], [50, 60, 70], [500, 600, 700]]
outputs = ['unidades', 'dezenas', 'centenas', 'unidades', 'dezenas', 'centenas', 'unidades', 'dezenas', 'centenas']
multilabel_knn.fit(
	inputs=inputs, # entradas de exemplo
	outputs=outputs, # saídas de exemplo
	k=0, # número referente a quantidade de vizinhos mais próximos
	precision=.75 # percentual relativo ao nível de intersecção dos rótulos
)
new_inputs = [[1, 2, 30], [10, 200, 300], [1, 20, 300]] # novas entradas para predizer
new_outputs = multilabel_knn.predict(inputs=new_inputs) # método de predição
print(new_outputs) # exibe as novas saídas com base nas novas entradas  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[['unidades', 'dezenas'], ['dezenas', 'centenas'], ['unidades', 'dezenas', 'centenas']]  
  </code>
</pre>
<br>
<p align="justify">
Clusterização ou Agrupamento: na Clusterização/Agrupamento iremos classificar dados para os quais não conhecemos previamente as classes, dessa forma a classificação é feita através da separação desses dados em conjuntos com características semelhantes. Confira a seguir um exemplo desse tipo de algoritmo com o K-Means:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.AutonomousLearning.k_means import KMeans
k_means = KMeans()
''' clusterização/agrupamento de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400]]
k_means.fit(inputs=inputs) # método de treinamento
clusters = k_means.predict(k=3) # método de predição para k=3 (3 grupos distintos)
for cluster in clusters: print(cluster) # exibe um grupo por linha  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[1, 2], [3, 4]]
[[10, 20], [30, 40]]
[[100, 200], [300, 400]]  
  </code>
</pre>
<br>
<p align="justify">
O algoritmo K-Means ou Médias K separa os dados baseando-se no centro de distribuição dos mesmos que é ocupada por eles quando estes estão dispostos em um gráfico de pontos, esse centro que representa o ponto central de aglomeração dos pontos é chamado de centroide e representa a média de cada conjunto definido pela variável K.
</p>
<p align="justify">
Regressão: na Regressão nós sabemos que tipo de resposta queremos mas não temos esse valor de forma exata como na classificação. Nesse tipo de algoritmo as respostas serão adaptativos de modo a se adaptarem as entradas recebidas fazendo com que possam emitir resultados iguais mas também diferentes dos contidos no histórico de dados. Na Regressão Simples usamos apenas uma entrada com uma única saída, veja um exemplo a seguir:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import SimpleRegression
simple_regression = SimpleRegression()
''' padrão regressivo simples de exemplo onde as saídas são as entradas multiplicadas por dez '''
inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
outputs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
simple_regression.fit(inputs=inputs, outputs=outputs, regression_type='linear') # método de treinamento do tipo linear
new_inputs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
new_outputs = simple_regression.predict(inputs=new_inputs)
print(new_outputs)  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]  
  </code>
</pre>
<br>
<p align="justify">
Na Regressão Multivariada que também é conhecida como Regressão Múltipla poderemos ter uma ou mais entradas porém cada lista de entrada terá somente uma única saída. No código abaixo temos um exemplo de Regressão  Multivariada/Múltipla:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import MultivariateRegression
multivariate_regression = MultivariateRegression()
''' padrão regressivo de exemplo para regressão multivariada/múltipla onde as saídas são as somas das entradas '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
multivariate_regression.fit(inputs=inputs, outputs=outputs, regression_type='linear') # método de treinamento do tipo linear
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = multivariate_regression.predict(inputs=new_inputs)
print(new_outputs)  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[5, 9, 13, 17, 21, 25, 29, 33, 37, 41]  
  </code>
</pre>
<br>
<p align="justify">
Já na Regressão Multivariável  nós temos a possibilidade de usar dados múltiplos tanto na entrada quanto na saída, apesar deste ser um método mais completo os seus resultados poderão não ser tão precisos. A seguir um exemplo de  Regressão Multivariável:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import MultivariableRegression
multivariable_regression = MultivariableRegression()
''' padrão regressivo de exemplo para regressão multivariável onde as saídas são os dobros das entradas '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20], [22, 24], [26, 28], [30, 32], [34, 36], [38, 40]]
multivariable_regression.fit(inputs=inputs, outputs=outputs, regression_type='linear') # método de treinamento do tipo linear
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = multivariable_regression.predict(inputs=new_inputs)
print(new_outputs)  
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[4, 6], [8, 10], [12, 14], [16, 18], [20, 22], [24, 26], [28, 30], [32, 34], [36, 38], [40, 42]]  
  </code>
</pre>
<br>
<p align="justify">
Dessa forma podemos realizar predições através de algoritmos com construções diferentes que emitem resultados também com diferentes níveis de precisão e confiabilidade. Não existe um algoritmo perfeito para cada tipo de caso ou um que seja melhor do que o outro, você deverá realizar os seus testes e comparar os resultados entre eles para decidir qual algoritmo aplicar.
</p>
</div>
