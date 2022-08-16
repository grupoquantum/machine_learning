from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import MultivariateRegression
multivariate_regression = MultivariateRegression()
''' padrão regressivo de exemplo para regressão multivariada/múltipla onde as saídas são as somas das entradas '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
multivariate_regression.fit(inputs=inputs, outputs=outputs, regression_type='linear') # método de treinamento do tipo linear
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = multivariate_regression.predict(inputs=new_inputs)
print(new_outputs)