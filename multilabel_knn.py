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