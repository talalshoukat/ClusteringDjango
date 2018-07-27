from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


def get_cluster_kmeans(tfidf_matrix, num_clusters):
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    cluster_list = km.labels_.tolist()
    return (cluster_list, km)

def kmeans_cluster_analysis(tfidf_matrix):
    # k means determine k
    sse = {}
    K = range(2,120)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(tfidf_matrix)
        sse[k] = kmeanModel.inertia_
        label = kmeanModel.labels_
        sil_coeff = silhouette_score(tfidf_matrix, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))

    # Plot the elbow
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

def get_top_terms_per_cluster(model, vectorizer, total_clusters):
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(total_clusters):
        print ('Cluster: '+ str(i)),
        for ind in order_centroids[i, :3]:
            print (terms[ind],)
        print
def get_dbscan_cluster(tfidf_matrix, epsilon):
    db = DBSCAN(eps= epsilon, min_samples= 3).fit(tfidf_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return labels


def multidim_scaling(similarity_matrix, n_components):
    one_min_sim = 1 - similarity_matrix
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=4)
    pos = mds.fit_transform(one_min_sim)  # shape (n_components, n_samples)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def pca_reduction(similarity_matrix, n_components):
    one_min_sim = 1 - similarity_matrix
    pca = PCA(n_components=10)
    pos = pca.fit_transform(one_min_sim)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def tsne_reduction(similarity_matrix):
    one_min_sim = 1 - similarity_matrix
    tsne = TSNE(learning_rate=1000).fit_transform(one_min_sim)
    x_pos, y_pos = tsne[:, 0], tsne[:, 1]
    return (x_pos, y_pos)
