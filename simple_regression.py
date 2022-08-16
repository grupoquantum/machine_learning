from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import SimpleRegression
simple_regression = SimpleRegression()
''' padrão regressivo simples de exemplo onde as saídas são as entradas multiplicadas por dez '''
inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
outputs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
simple_regression.fit(inputs=inputs, outputs=outputs, regression_type='linear') # método de treinamento do tipo linear
new_inputs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
new_outputs = simple_regression.predict(inputs=new_inputs)
print(new_outputs)