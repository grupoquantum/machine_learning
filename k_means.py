from Neuraline.ArtificialIntelligence.MachineLearning.AutonomousLearning.k_means import KMeans
k_means = KMeans()
''' clusterização/agrupamento de exemplo entre unidades, dezenas e centenas '''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400]]
k_means.fit(inputs=inputs) # método de treinamento
clusters = k_means.predict(k=3) # método de predição para k=3 (3 grupos distintos)
for cluster in clusters: print(cluster) # exibe um grupo por linha