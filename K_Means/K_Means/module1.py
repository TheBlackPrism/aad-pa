import matplotlib as plt

km = K_Means(3)
km.fit(X)

colors = 10*["r","g","c","b","k"]

for centroid in km.centroids:
    plt.scatter(backend_managers.centroids[centroid][0], km.centroids[centroid][1],s = 130, marker = "x")

for classification in km.classes:
    color = colors[classification]
    for features in km.classes[classification]:
        plt.scatter(features[0], features[1], color = color,s = 30)

plt.show()


