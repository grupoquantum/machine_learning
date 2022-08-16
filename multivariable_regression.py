from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import MultivariableRegression
multivariable_regression = MultivariableRegression()
''' padrão regressivo de exemplo para regressão multivariável onde as saídas são os dobros das entradas '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20], [22, 24], [26, 28], [30, 32], [34, 36], [38, 40]]
multivariable_regression.fit(inputs=inputs, outputs=outputs, regression_type='linear') # método de treinamento do tipo linear
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = multivariable_regression.predict(inputs=new_inputs)
print(new_outputs)