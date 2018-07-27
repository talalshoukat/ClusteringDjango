import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.cluster.hierarchy import ward,single,complete,average,weighted, dendrogram, fcluster, to_tree, cut_tree, leaders,ClusterNode
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
import sys
import os

def scatter_clusters(x_pos, y_pos, clusters, titles):
    cluster_colors = {0: '#cc0000',
                      1: '#006600',
                      2: '#002699',
                      3: '#ffff33',
                      4: '#ffa64d',
                      5: '#000000'}
    # As many as items
    cluster_names = {0: '',
                 1: '',  
                 2: '', 
                 3: '',
                 4: '',
                 5: ''}
                 
    df = pd.DataFrame(dict(x= x_pos, y= y_pos, label= clusters, title= titles)) 
    groups = df.groupby('label')
    fig, ax = plt.subplots(figsize=(17, 9))  # Set size
    #ax.set_axis_bgcolor('#e6f7ff')
    # Iterate through groups to layer the plot
    for name, group in groups:
        ax.plot(group.x, group.y, marker='D', linestyle='solid', ms=15, 
                label=cluster_names[name], color=cluster_colors[name], mec='black')
        ax.set_aspect('auto')
        ax.tick_params(axis= 'x', which='both', labelbottom='off')
        ax.tick_params(axis= 'y', which='both', labelleft='off')
    ax.legend(numpoints=1)

    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size= 15)  
    #plt.show() # Show the plot

def get_nodes_at_depth(tree, depth):
    nodes = []
    #Waqas code
    #if (depth == 0 or tree.get_count() < 50):
    if (tree.get_count() < 50 or (tree.left is None and tree.right is None)):
        nodes.append(tree)
        return nodes
    if tree.left is not None:
        left_nodes = get_nodes_at_depth(tree.left,depth-1)
        # Waqas code
        #for i, node in enumerate(left_nodes):
        #    nodes.append(node)
        #comment for normal flow
        nodes.append(left_nodes)
    if tree.right is not None:
        right_nodes = get_nodes_at_depth(tree.right,depth-1)
        # Waqas code
        #for i, node in enumerate(right_nodes):
        #        nodes.append(node)
        # comment for normal flow
        nodes.append(right_nodes)
    return nodes


def get_nodes_at_depth_(tree):
    nodes = []
    if (tree.get_count() < 50):
        nodes.append(tree)
        return nodes
    if tree.left is not None:
        left_nodes = get_nodes_at_depth(tree.left)
        for i, node in enumerate(left_nodes):
            nodes.append(node)
    if tree.right is not None:
        right_nodes = get_nodes_at_depth(tree.right)
        for i, node in enumerate(right_nodes):
                nodes.append(node)
    return nodes


def get_clusters_with_hierarchy(tree):
    #comment for normal flow
    nodes = get_nodes_at_depth(tree, 1)
    #Waqas code1
    #nodes = get_nodes_at_depth(tree, 5)
    # comment for normal flow
    clusters=get_clusters(nodes)
    # Waqas code2
    '''clusters = [None] * len(nodes)
    for i, node in enumerate(nodes):
        n = get_nodes_at_depth(node,3)
        clusters[i] = [None] * len(n)
        for j, val in enumerate(n):
            clusters[i][j] = val.pre_order(lambda x:x.id)
    '''
    return clusters


def get_clusters(nodes):
    # nodes = get_nodes_at_depth(tree, 1)
    clusters=[]
    if(len(nodes)==1):
        for j, val in enumerate(nodes):
            clusters.append(val.pre_order(lambda x: x.id))
        return clusters
    for i, node in enumerate(nodes):
        clusters.append(get_clusters(node))
    return clusters



def ward_clustering(similarity_matrix):
    similarity_matrix = 1 - similarity_matrix
    ward = AgglomerativeClustering(n_clusters=120, linkage='ward').fit(similarity_matrix)
    label = ward.labels_
    return (ward,label)

def nmf(tfidf, vectorizer):
    nmf = NMF(n_components=30, random_state=1).fit(tfidf)
    feature_names = vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-2 - 1:-1]]))
        print()

def ward_dendogram(similarity_matrix, book_names):
    linkage_matrix = ward(similarity_matrix) #Define the linkage_matrix using ward clustering pre-computed distances
    assignments = fcluster(linkage_matrix,3,depth=5)
    clusters = get_clusters_with_hierarchy(to_tree(linkage_matrix))
    return [assignments, clusters]

def single_dendogram(similarity_matrix, book_names):
    linkage_matrix = single(similarity_matrix)  # Define the linkage_matrix using ward clustering pre-computed distances
    assignments = fcluster(linkage_matrix, 3, depth=5)
    clusters = get_clusters_with_hierarchy(to_tree(linkage_matrix))
    return [assignments, clusters]

def complete_dendogram(similarity_matrix, book_names):
    linkage_matrix = complete(similarity_matrix)  # Define the linkage_matrix using ward clustering pre-computed distances
    assignments = fcluster(linkage_matrix, 3, depth=5)
    clusters = get_clusters_with_hierarchy(to_tree(linkage_matrix))
    return [assignments, clusters]

def average_dendogram(similarity_matrix, book_names):
    linkage_matrix = average(similarity_matrix)  # Define the linkage_matrix using ward clustering pre-computed distances
    assignments = fcluster(linkage_matrix, 3, depth=5)
    clusters = get_clusters_with_hierarchy(to_tree(linkage_matrix))
    return [assignments, clusters]

    #mpl.rcParams['lines.linewidth'] = 5

    #fig, ax = plt.subplots(figsize=(150, 200)) # Set size
    ##sys.setrecursionlimit(30000)
    ##ax = dendrogram(linkage_matrix, orientation="right", labels=book_names);

    #fig.subplots_adjust(bottom=0)
    #fig.subplots_adjust(top=200)
    #fig.subplots_adjust(left=0)
    #fig.subplots_adjust(right=200)
    #plt.tick_params(\
    #    axis= 'x',
    #    which='both',
    #    bottom='off',
    #    top='off',
    #    labelbottom='off',
    #    length = 25)
    #plt.tick_params(\
    #    axis= 'y',
    #    which='both',
    #    bottom='off',
    #    top='off',
    #    labelbottom='off',
    #    labelsize = 20)
    ##plt.tick_params(width=50, length = 10)
    #plt.tight_layout() # Show plot with tight layout
    ##plt.show()
    ##plt.savefig('plt.png',dpi=200, format='png',bbox_inches='tight')
