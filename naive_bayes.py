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