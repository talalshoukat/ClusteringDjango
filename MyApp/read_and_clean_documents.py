import glob
from gensim import models,corpora
from html.parser import HTMLParser
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk.corpus.reader.wordnet
from json import JSONEncoder
import re
import os
import copy
import pandas as pd
from MyApp.text_processing import get_similarity_matrix, get_feature_list,tokenize_and_lemmatize
from sklearn.decomposition import NMF, LatentDirichletAllocation


directory = '\\\\?\\D:\\Repositories\\PGConfizBotmiscellaneous\\Machine Learning\\cluster-code\\data\\cleaned_data\\'
cachedict = {}

def read_authors_book_names():
    with open(directory + 'authors.txt', "r") as f:
        authors = []
        for line in f:
            line_str = line.strip()
            authors.append(line_str)
    f.close()

    with open(directory + 'books.txt', "r") as f:
        books = []
        for line in f:
            line_str = line.strip()
            books.append(line_str)
    f.close()
    return (books, authors)

def read_incident_numbers():
    with open(directory + 'cleaned_incidents.txt', "r") as f:
        incidents = []
        for line in f:
            line_str = line.strip()
            incidents.append(line_str)
    f.close()
    return incidents

def read_incident_data():
    with open(directory + 'incident_data.txt', "r") as f:
        incidents = []
        descriptions = []
        for line in f:
            line_str = line.strip()
            arr = line_str.split(',',1);
            if(len(arr)==2):
                incidents.append(arr[0])
                descriptions.append(arr[1].split())
    f.close()
    return (incidents,descriptions)

def write_list_to_file(file_name, content):
    f = open(directory + file_name, 'a')
    for list in content:
        for item in list:
            f.write(str(item))
            f.write(' ')
        f.write('\n')
    f.close()
def write_content_to_file(file_name, content):
    f = open(directory + file_name, 'a')
    for item in content:
        f.write(str(item))
        f.write('\n')
    f.close()
def write_content_to_file_in_quotes(file_name, content):
    f = open(directory + file_name, 'a')
    for item in content:
        f.write("'"+str(item)+"',")
        f.write('\n')
    f.close()
def read_from_cleaned_file(file_name):
    with open(directory + file_name, "r") as f:
        content_as_list = []
        content_as_str = []
        for line in f:
            line_str = line.strip()
            line_list = line_str.split()
            content_as_list.append(line_list)
            content_as_str.append(line_str)
    f.close()
    return (content_as_list, content_as_str)

def safe_make_folder(i,index, cluster_corpas):
    '''Makes a folder (and its parents) if not present'''
    try:  
        os.makedirs(directory + i+str(index))
        index=index+1
        cluster_corpas.append("")
        return index,cluster_corpas
    except ValueError as e:
        return index,cluster_corpas

def save_hierarchy(clusters, incidents, cleaned_content_as_str) :
    cluster_corpas = []
    counterCluster=0
    for index, cluster in enumerate(clusters):
        cluster_incident_description = []
        folderCreated=False
        l=-1
        sub_cluster_corpas = []
        for j , c in enumerate(cluster):
            if len(c)<5:
                continue
            if not folderCreated:
                counterCluster, cluster_corpas = safe_make_folder("hierarchy3\\cluster", counterCluster, cluster_corpas)
                folderCreated=True
            incident_list = []
            incident_descriptions = []
            l=l+1
            sub_cluster_corpas.append("")
            for k, incident in enumerate(c):
                incident_list.append(incidents[incident])
                sub_cluster_corpas[l]=sub_cluster_corpas[l]+" "+cleaned_content_as_str[incident]
                cluster_corpas[counterCluster-1]=cluster_corpas[counterCluster-1]+" "+cleaned_content_as_str[incident]
                incident_descriptions.append(cleaned_content_as_str[incident])
                cluster_incident_description.append(cleaned_content_as_str[incident])
            #feature_list = get_feature_list(cluster_corpas)
            sub_nmf_topic, sub_lda_topic,df = cluster_labeling(sub_cluster_corpas, 1, 1)


            df.to_csv(directory +"hierarchy3\\cluster"+str(counterCluster-1)+"\\DataFrame-"+str(l)+".csv", sep=',', encoding='utf-8')
            #write_content_to_file("hierarchy3\\cluster"+str(index)+"\\DataFrame-"+sub_nmf_topic[0]+"-"+sub_lda_topic[0] +"-"+str(j)+".txt",df)
            write_content_to_file("hierarchy3\\cluster"+str(counterCluster-1)+"\\subcluster-"+str(l)+".txt",incident_descriptions)
            #write_content_to_file("hierarchy3\\cluster"+str(index)+"\\subcluster-features-"+sub_nmf_topic[0]+"-"+sub_lda_topic[0] +"-"+str(j)+".txt",feature_list)
        #feature_list = get_feature_list(cluster_incident_description)

        #Cluster name using LDA technique
        if len(sub_cluster_corpas) > 0:
            for x in range(l+1):
                sub_nmf_topic, sub_lda_topic, df = cluster_labeling(sub_cluster_corpas, 1, 1)
                #label=df.iloc[:, x].argmax()
                labels=list(df.nlargest(5, list(df.columns.values)[x]).index)
                title=""
                for a,label in enumerate(labels):
                    title=title+"-"+label
                os.rename(directory +"hierarchy3\\cluster"+str(counterCluster-1)+"\\DataFrame-"+str(x)+".csv", directory +"hierarchy3\\cluster"+str(counterCluster-1)+"\\DataFrame-"+title +"-"+str(x)+".csv")
                os.rename(directory +"hierarchy3\\cluster"+str(counterCluster-1)+"\\subcluster-"+str(x)+".txt", directory +"hierarchy3\\cluster"+str(counterCluster-1)+"\\subcluster-"+title +"-"+str(x)+".txt")

            nmf_topic, lda_topic, dataframe = cluster_labeling(sub_cluster_corpas, 1, 1)
            df.to_csv(directory + "hierarchy3\\cluster-feature-DataFrame " + str(index) + "-" + lda_topic[0] + "-" + nmf_topic[0] + ".csv", sep=',', encoding='utf-8')

    if len(cluster_corpas) > 0:
        for x in range(counterCluster):
            nmf_topic, lda_topic, df = cluster_labeling(cluster_corpas, 1, 1)
            # label=df.iloc[:, x].argmax()
            labels = list(df.nlargest(5, list(df.columns.values)[x]).index)
            title = ""
            for a, label in enumerate(labels):
                title = title + "-" + label
            os.rename(directory + "hierarchy3\\cluster" + str(x) ,directory + "hierarchy3\\cluster" + str(x) + title)
            #os.rename(directory + "hierarchy3\\cluster" + str(x) + "\\subcluster-" + str(x) + ".txt",directory + "hierarchy3\\cluster" + str(index) + "\\subcluster-" + title + "-" + str(x) + ".txt")

        #nmf_topic, lda_topic, dataframe = cluster_labeling(sub_cluster_corpas, 1, 1)
        #df.to_csv(directory + "hierarchy3\\cluster-feature-DataFrame " + str(index) + "-" + lda_topic[0] + "-" + nmf_topic[0] + ".csv", sep=',', encoding='utf-8')

        nmf_topic, lda_topic, dataframe = cluster_labeling(cluster_corpas, 1, 1)

        #if len(cluster_incident_description)>0:
        #    ldamodel = lda_topic_modeling(cluster_incident_description, 1)
        #   write_content_to_file("hierarchy3\\cluster-feature " + ldamodel.show_topics(formatted=False, num_words=1) + ".txt",feature_list)
        #write_content_to_file("hierarchy3\\cluster-feature "+str(index)+".txt",feature_list)



def cluster_labeling(cluster_incident_description,no_topics,no_top_words):




        no_features = 1000

        # NMF is able to use tf-idf
        #tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tfidf_vectorizer = TfidfVectorizer(max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(cluster_incident_description)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer( max_features=no_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(cluster_incident_description)
        tf_feature_names = tf_vectorizer.get_feature_names()

        #no_topics = 1

        # Run NMF
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        corpus_index = [n for n in cluster_incident_description]

        df = pd.DataFrame(tfidf.T.todense(), index=tfidf_feature_names, columns=corpus_index)
        # Run LDA
        lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(tf)

        #no_top_words = 1
        nmf_topic=display_topics(nmf, tfidf_feature_names, no_top_words)
        lda_topic=display_topics(lda, tf_feature_names, no_top_words)
        return nmf_topic, lda_topic,df



def save_all_hierarchy(clusters, incidents, cleaned_content_as_str) :
    cluster_corpas = []
    safe_make_folder("AllClusters\\cluster", 1, cluster_corpas)
    save_cluster("AllClusters\\cluster1",clusters, incidents, cleaned_content_as_str)

class Incidents:
    incident_data=[]
    LDA_title=""
    NMF_title = ""
    def __init__(self, Name="NoTitle"):
        self.LDA_title = Name
        self.NMF_title=Name
    def jsonable(self):
        return self.__dict__

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = Incidents()

    def jsonable(self):
        return self.__dict__

def ComplexHandler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()

class MyEncoder(JSONEncoder):
    def default(self, o):
        o.data.__dict__
        return o.__dict__

def json_cluster(cluster,tree, incidents, cleaned_content_as_str):
    incident_descriptions=[]
    #Base case for recursive call
    #Return  desciption at leaf level and inc object
    if len(cluster)==1:
        inc = Incidents()
        for k, incident in enumerate(cluster[0]):
            incident_descriptions.append(cleaned_content_as_str[incident])
            #tree.incident_data.append(cleaned_content_as_str[incident])
            #get title from lda and nmf
        sub_nmf_topic, sub_lda_topic, df = cluster_labeling(incident_descriptions, 1, 1)
            #pass whole object rather than incident list
        #incidents_list.Incidents=incident_descriptions
        inc.LDA_title=sub_lda_topic
        inc.NMF_title = sub_nmf_topic
        inc.incident_data=incident_descriptions
        tree.data=inc
        return incident_descriptions,inc


    for index, cluster in enumerate(cluster):
        inc = Incidents()
        #Check if node is left or right
        #Call json_cluster method recurcively to get left and right node data
        if index==0:
            tree.left=Tree()
            inc_des,inc_list=json_cluster(cluster ,tree.left, incidents, cleaned_content_as_str)
        else:
            tree.right = Tree()
            inc_des, inc_list = json_cluster(cluster, tree.right, incidents, cleaned_content_as_str)\
        #assign description from note children to current node
        for  desc in inc_des:
            incident_descriptions.append(desc)
    #Get labels of current cluster
    sub_nmf_topic, sub_lda_topic, df = cluster_labeling(incident_descriptions, 1, 1)
    inc.incident_data=incident_descriptions
    inc.LDA_title = sub_lda_topic
    inc.NMF_title = sub_nmf_topic
    tree.data=inc
    return incident_descriptions,tree



def save_cluster(path,cluster, incidents, cleaned_content_as_str):
    incident_descriptions=[]

    if len(cluster)==1:
        for k, incident in enumerate(cluster[0]):
            incident_descriptions.append(cleaned_content_as_str[incident])
        sub_nmf_topic, sub_lda_topic, df = cluster_labeling(incident_descriptions, 1, 1)

        write_content_to_file(path + "\\cluster.txt",incident_descriptions)
        os.rename(directory + path,
                  directory + path+"-"+sub_nmf_topic[0]+"-"+sub_lda_topic[0])
        return incident_descriptions

    for index, cluster in enumerate(cluster):
        safe_make_folder(path+"\\cluster", index,cleaned_content_as_str)
        for  desc in save_cluster(path+"\\cluster"+str(index), cluster, incidents, cleaned_content_as_str):
            incident_descriptions.append(desc)
    sub_nmf_topic, sub_lda_topic, df = cluster_labeling(incident_descriptions, 1, 1)
    os.rename(directory +path,directory + path+str(index) + sub_nmf_topic[0]+"-"+sub_lda_topic[0])
    return incident_descriptions

def save_json_hierarchy(clusters, incidents, cleaned_content_as_str) :
    #Tree node added
    root = Tree()
    #Return desciption of incidents and tree with topic assigned
    return json_cluster(clusters,root,incidents, cleaned_content_as_str)


def save_ward_hierarchy(clusters, incidents, cleaned_content_as_str) :
    cluster_corpas = []
    safe_make_folder("ClusterLinkage\\Ward\\cluster", 1, cluster_corpas)
    clu=save_cluster("ClusterLinkage\\Ward\\cluster1",clusters, incidents, cleaned_content_as_str)
    #save_Json(clusters, incidents, cleaned_content_as_str)

def save_single_hierarchy(clusters, incidents, cleaned_content_as_str) :
    cluster_corpas = []
    safe_make_folder("ClusterLinkage\\single\\cluster", 1, cluster_corpas)
    save_cluster("ClusterLinkage\\single\\cluster1",clusters, incidents, cleaned_content_as_str)

def save_complete_hierarchy(clusters, incidents, cleaned_content_as_str) :
    cluster_corpas = []
    safe_make_folder("ClusterLinkage\\complete\\cluster", 1, cluster_corpas)
    save_cluster("ClusterLinkage\\complete\\cluster1",clusters, incidents, cleaned_content_as_str)

def save_average_hierarchy(clusters, incidents, cleaned_content_as_str) :
    cluster_corpas = []
    safe_make_folder("ClusterLinkage\\average\\cluster", 1, cluster_corpas)
    save_cluster("ClusterLinkage\\average\\cluster1",clusters, incidents, cleaned_content_as_str)



def generateCorpus(cluster_incident_description):
    tokenizer = RegexpTokenizer(r'\w+')
    # load nltk's English stopwords as variable called 'stopwords'
    en_stop = stopwords.words('english')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    texts = []
    # loop through document list
    for i in cluster_incident_description:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]


def tfidf(content_as_str):
    words_to_remove = ['user', 'able', 'ap', 'eu', 'na', 'la', 'th', 'emp', 'hp', 'dc', 'cn', 'gs3', 'wki', 'wiki',
                       'ct', 'ind', 'want', 'need']
    for index, str in enumerate(content_as_str):
        for r in words_to_remove:
            content_as_str[index] = content_as_str[index].replace(r, '')
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000, min_df=0.034,
                                       stop_words='english', use_idf=True,
                                       tokenizer=tokenize_and_lemmatize, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_as_str)  # fit the vectorizer to synopses
    return  (tfidf_matrix, tfidf_vectorizer)
def lda_topic_modeling(content, topic_count):
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(content)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in content]

    # generate LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topic_count, id2word=dictionary, passes=20)

    return ldamodel

def read_word_list(file_name):
    word_list = []
    with open(directory + file_name, "r") as f:
        content_as_list = []
        content_as_str = []
        for line in f:
            line_str = line.strip()
            line_list = line_str.split()
            content_as_list.append(line_list)
            content_as_str.append(line_str)
    f.close()
    return (content_as_list, content_as_str)

def normalize_words_return_list2(str_list):
    rep = {'(^| |\n)((ping id)|(ping )|(ping$))': ' pingid ', '(^| |\n)(one drive)( |$|\n)' : 'onedrive', '(^| |\n)(set up)( |$|\n)' : 'setup'}
    list = []
    for line in str_list:
        line_str = line.strip()
        line_str = strrep(line_str,rep)
        line_str = line_str.strip()
        line_list = line_str.split()
        list.append(line_list)
    return list

def strrep(orig, repdict):
    for k,v in repdict.items():
        orig = re.sub(k,v, orig)
    return orig
class MyHTMLParser(HTMLParser):  # To parse html files
    html_plain_document = []
    def handle_data(self, data):  # For every html data
        self.html_plain_document.append(data)  # Append the data to the list


def get_file_names(data_folder):
    """
    Creates a list of list containing folder and file names

    Param_1: Data folder name as string
    Output_1: List of lists containng folder and file names
    """
    folder_list = glob.glob(data_folder+"*")  # Get folder and file names names in a list
    folder_file_list = []
    for index,folder in enumerate(folder_list):
        folder_file_list.append([])
        folder_name = [ folder[folder.rfind('/') + 1 : ] ]  # Get folder name
        file_names = glob.glob(folder+ '/*.html')  # Get html file names in a list
        file_names = sorted(file_names)
        folder_file_list[index].append(folder_name)
        folder_file_list[index].append(file_names)
    return folder_file_list


def get_cleaned_html_documents(folder_file_list):
    """
    Reads the files in folders
    Returns a list of list containing cleaned words/data in the html files

    Param_1: Output of get_file_names, list of folders and file names
    Output_1: List of lists containing string, which are words in the html
    """
    html_parser = MyHTMLParser()
    cleaned_documents = []
    for item in folder_file_list:
        for file_name in item[1]:
            f = open(file_name, "r")  # Open the file
            x = f.read()  # Read file
            html_parser.feed(x)  # Feed the file to get rid of html elements
            f.close()  # Close the file
        plain_html_file = html_parser.html_plain_document  # Get the list of words
        cleaned_html_file = clean_list(plain_html_file)  # Clean the list
        cleaned_documents.append(cleaned_html_file)
        html_parser.html_plain_document = []  # Empty the html document
    return cleaned_documents

def get_cleaned_descriptions(list_descriptions, incidents):
    """
    Function to clean the list of description
    Removes any non-alphanumeric characters
    Stems words
    Gets rid of any empty elements in the list

    Param_1: List, containing List of strings
    Output_1: List, containing list of cleaned strings
    """
    for index,item in enumerate(list_descriptions):
        list_descriptions[index] = clean_list(list_descriptions[index]);
    return (list_descriptions, incidents)

def normalize_words_return_list(list_to_clean):
    """
    Function to clean a list
    Removes any non-alphanumeric characters
    Stems words
    Gets rid of any empty elements in the list

    Param_1: List, containing strings
    Output_1: List, containing cleaned strings
    """
    stemmer = PorterStemmer()
    lmtzr = WordNetLemmatizer()
    items_to_clean = set(list(stopwords.words('english')) + ['\n','\n\n','\n\n\n','\n\n\n\n','ocroutput','',' ','user','able','ap','eu','na','la','th', 'emp','hp','dc','cn','gs3','wiki','ct', 'ind', 'need', 'want'])
    # Items to clean
    regex_non_alphanumeric = re.compile('[^0-9a-zA-Z]')  # REGEX for non alphanumeric chars
    cleaned_list = []
    for i, str in enumerate(list_to_clean):
        lst = str.split()
        tagged_list = pos_tag(lst)
        for index,item in enumerate(lst):
            item = regex_non_alphanumeric.sub(' ', item)  # Filter text, remove non alphanumeric chars
            item = item.lower()  # Lowercase the text
            #item = stemmer.stem(item)  # Stem the text
            pos_of_word = get_pos_for_lemmatization( tagged_list[index][1])
            item = lmtzr.lemmatize(item,pos_of_word)
            if len(item) < 3:  # If the length of item is lower than 3, remove item
                item = ''
            lst[index] = item  # Put item back to the list
        cleaned_string = [elem.strip() for index,elem in enumerate(lst) if elem.strip() not in items_to_clean]
        cs_full = '';
        for cs in cleaned_string:
            cs_full += cs + ' '
        cleaned_list.append(cs_full.strip())
    # Remove empty items from the list
    return cleaned_list

def get_pos_for_lemmatization(pos):
    if pos.startswith('NN'):
        return 'n'
    if pos.startswith('VB'):
        return 'v'
    if 'JJ' in pos:
        return 'a'
    return 'n'
def remove_frequent_items(book_word_list, percentage):
    """
    Remove frequently occured words

    Param_1: List of list containing strings
    Param_2: Above x percentage of occurance will be removed
    Output_1: Cleaned list
    """
    treshold = int(len(book_word_list) * percentage / 100)
    DF = defaultdict(int)
    for cleaned_list in book_word_list:
        for word in set(cleaned_list):
                DF[word] += 1
    words_to_remove = {k:v for k,v in DF.items() if v > treshold }
    # A new dictionary of items that only has count above treshold
    words_to_remove_as_list = set(words_to_remove.keys())
    freq_items_removed_book_word_list = []
    for book in book_word_list:
        freq_items_removed_list = [word for word in book if word not in words_to_remove_as_list]
        freq_items_removed_book_word_list.append(freq_items_removed_list)
    return freq_items_removed_book_word_list


def convert_list(content):
    converted_content = []
    for index, str_list in enumerate(content):
        converted_content.append('')
        for item in str_list:
            converted_content[index] = converted_content[index] + ' ' + item
    return converted_content
