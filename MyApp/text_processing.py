from gensim import corpora, models
import gensim
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.decomposition import NMF

stemmer = SnowballStemmer("english")

def read_positive_and_negative_word(data_folder):
    """
    Reads positive and negative adjectives from flat files

    Param_1: File location
    Output_1: Positive words in a list of strings
    Output_2: Negative words in a list of strings
    """
    positiveWords = []
    negativeWords = []
    with open(data_folder+'positive.txt', 'r') as readFile:
        for line in readFile:
            line = line.replace('\n','')
            positiveWords.append(line)

    with open(data_folder+'negative.txt', 'r') as readFile:
        for line in readFile:
            line = line.replace('\n','')
            negativeWords.append(line)
    return (positiveWords, negativeWords)

def get_pos_for_lemmatization(pos):
    if pos.startswith('NN'):
        return 'n'
    if pos.startswith('VB'):
        return 'v'
    if 'JJ' in pos:
        return 'a'
    return 'n'

def filter_positive_negative(book_word_list, positive_words, negative_words):
    """
    Generates a list of positive and negative filtered lists
    
    Param_1: List, containing all words in the book 
    Param_2: List of positive words
    Param_3: List of negative words
    Output_1: List of positive filtered words from all words in the book
    Output_2: List of negative filtered words from all words in the book
    """
    positive_filtered_list = []
    negative_filtered_list = []
    for book in book_word_list:
        current_book_positive_list = []
        current_book_negative_list = []
        for word in book:
            if word in positive_words:
                current_book_positive_list.append(word)
            if word in negative_words:
                current_book_negative_list.append(word)
        positive_filtered_list.append(current_book_positive_list)
        negative_filtered_list.append(current_book_negative_list)
    return (positive_filtered_list, negative_filtered_list)


def lda_topic_modeling(content, topic_count):
    dictionary = corpora.Dictionary(content)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in content]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_count, id2word = dictionary, passes=20)
    #print(ldamodel.print_topics(num_topics=5, num_words=4))
    return lda_model


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
    
def tokenize_and_lemmatize(text):
    words_to_remove = ['user','able','ap','eu','na','la','th', 'emp','hp','dc','cn','gs3','wiki','ct', 'ind']
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.lower() not in words_to_remove]

    tagged_list = pos_tag(tokens);    
    lemmatizes = [WordNetLemmatizer().lemmatize(t,get_pos_for_lemmatization( tagged_list[i][1])) for i,t in enumerate(tokens)]
    return lemmatizes

def get_similarity_matrix(content_as_str):
    words_to_remove = ['user','able','ap','eu','na','la','th', 'emp','hp','dc','cn','gs3','wki','wiki','ct', 'ind', 'want', 'need']
    for index,str in enumerate(content_as_str):
        for r in words_to_remove:
            content_as_str[index] = content_as_str[index].replace(r,'')
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000, min_df=0.034,
                                       stop_words='english',use_idf=True,
                                       tokenizer=tokenize_and_lemmatize, ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_as_str) #fit the vectorizer to synopses
    feature_names = tfidf_vectorizer.get_feature_names()
    similarity_matrix = cosine_similarity(tfidf_matrix)
    #vectorizer = CountVectorizer()
    #BOW=vectorizer.fit_transform(content_as_str).todense()
    return (similarity_matrix, tfidf_matrix, tfidf_vectorizer)

def nmf_features(tfidf, vectorizer):
    nmf = NMF(n_components=5, random_state=1).fit(tfidf)
    feature_names = vectorizer.get_feature_names()
    feature_list = []
    for topic_idx, topic in enumerate(nmf.components_):
        feature_list.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-2 - 1:-1]]))
    return feature_list

def get_feature_list(content_as_str):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.75, max_features=200000, min_df=0.15,
                                       stop_words='english',use_idf=True,
                                       tokenizer=tokenize_and_lemmatize, ngram_range=(1,2))
    #fit the vectorizer to synopses
    try:
         tfidf_matrix = tfidf_vectorizer.fit_transform(content_as_str)
        #fit the vectorizer to synopses
    except:
        try:
            tfidf_vectorizer = TfidfVectorizer(max_df=.85, max_features=200000, min_df=0.10,
                                stop_words='english',use_idf=True,
                                tokenizer=tokenize_and_lemmatize, ngram_range=(1,2))
            tfidf_matrix = tfidf_vectorizer.fit_transform(content_as_str)
        except:
            return []
    feature_names = tfidf_vectorizer.get_feature_names()
    feature_names.extend(nmf_features(tfidf_matrix, tfidf_vectorizer))
    return feature_names