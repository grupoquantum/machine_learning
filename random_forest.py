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