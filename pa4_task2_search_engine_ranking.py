#####begin put all your imports
from __future__ import division
import nltk
import pymongo
from pymongo import MongoClient
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import linalg as LA
import urllib
import operator
import os
#####end put all your imports 


mongodb_client = None
mongodb_db = None
document_frequency = defaultdict(int)
total_number_of_docs = 0


tokenizer = RegexpTokenizer(r'\w+')
stemmer=PorterStemmer()
lmtzr = WordNetLemmatizer()


def setup_mongodb():
    #####################Task t2a: your code #######################
    # Connect to mongodb
    global mongodb_client, mongodb_db
    mongodb_client = MongoClient()
    mongodb_db = mongodb_client['uta-edu-corpus']
    #####################Task t2a: your code #######################


# This function processes the entire document corpus
def process_document_corpus(file_name):
    #####################Task t2b: your code #######################
    #### The input is a file where in each line you have two information
    #   filename and url separated by |
    # Process the file line by line
    #   and for each line call the function process_document with file name and url and index
    #   first file should have an index of 0, second has 1 and so on
    #Remember to set total_number_of_docs to number of documents
    #####################Task t2b: your code #######################
    global total_number_of_docs
    with open(file_name) as doc_corpus:
        index = 0
        for line in doc_corpus:
            #print line
            process_document(line.split('|')[0], line.split('|')[1], index)
            index += 1
        total_number_of_docs = index


#This function processes a single web page and inserts it into mongodb
def process_document(file_name, url, index):
    #Do not change 
    f = open(file_name)
    file_contents = f.read()
    f.close()
    
    soup = BeautifulSoup(file_contents)

    #####################Task t2c: your code #######################
    #Using the functions that you will write (below), convert the document
    #   into the following structure:
    # title_processed: a string that contains the title of the string after processing
    # hx_processed: an array of strings where each element is a processed string
    #   for eg, if the document has two h1 tags, then the array h1_processed will have two elements
    #   one for each h1 tag and contains its contentent after processing
    # a_processed: same for a tags
    # body_processed: a string that contains body of the document after processing
    
    title_processed = process_text(soup.title.string)
    h1_processed = process_array([header.text.encode('ascii', 'ignore').decode('ascii') for header in soup.find_all('h1')])
    h2_processed = process_array([header.text.encode('ascii', 'ignore').decode('ascii') for header in soup.find_all('h2')])
    h3_processed = process_array([header.text.encode('ascii', 'ignore').decode('ascii') for header in soup.find_all('h3')])
    h4_processed = process_array([header.text.encode('ascii', 'ignore').decode('ascii') for header in soup.find_all('h4')])
    h5_processed = process_array([header.text.encode('ascii', 'ignore').decode('ascii') for header in soup.find_all('h5')])
    h6_processed = process_array([header.text.encode('ascii', 'ignore').decode('ascii') for header in soup.find_all('h6')])
    a_processed = []
    for link in soup.find_all('a'):
        if(link.has_attr("href")):
            a_processed.append(link['href'])
    #a_processed = [link['href'] for link in soup.find_all('a')]
    body_processed = process_text(soup.body.get_text(" "))

    #Insert the processed document into mongodb
    #Do not change 
    webpages = mongodb_db.webpages
    document_to_insert = {
        "url": url,
        "title": title_processed,
        "h1": h1_processed,
        "h2": h2_processed,
        "h3": h3_processed,
        "h4": h4_processed,
        "h5": h5_processed,
        "h6": h6_processed,
        "a": a_processed,
        "body": body_processed,
        "filename": file_name,
        "index": index
    }
    #print document_to_insert
    #webpage_id = webpages.insert_one(document_to_insert)
    webpage_id = webpages.insert(document_to_insert)
    #####################Task t2c: your code #######################

    #Do not change below
    #Write the processed document
    new_file_name = file_name.replace("downloads/", "processed/")
    f = open("processedFileNamesToUUID.txt", "a")
    f.write(new_file_name + '|' + url)
    f.flush()
    f.close()

    f = open(new_file_name, "w")
    f.write(body_processed)
    f.close()


#helper function for h tags and a tags
# use if needed
def process_array(array):
    processed_array = [process_text(element) for element in array]
    return processed_array

#This function does the necessary text processing of the text
def process_text(text):
    #####################Task t2d: your code #######################
    #Given the text, do the following:
    #   convert it to lower case
    #   remove all stop words (English)
    #   remove all punctuation
    #   stem them using Porter Stemmer
    #   Lemmatize it
    #####################Task t2d: your code #######################
    processed_text = text.lower()
    #print processed_text
    tokens = word_tokenize(processed_text)
    #print tokens
    stop_words_removed = [w for w in tokens if not w in stopwords.words('english')]
    #print stop_words_removed
    sentence = ""
    for word in stop_words_removed:
        sentence = sentence+" "+word
    punctuation_removed = tokenizer.tokenize(sentence)
    sentence = ""
    for word in punctuation_removed:
        sentence = sentence+" "+lmtzr.lemmatize(stemmer.stem(word).encode('ascii', 'ignore').decode('ascii'))
    return sentence


#This function determines the vocabulary after processing
def find_corpus_vocabulary(file_name):
    vocabulary = None
    top_5000_words = None
    #Document frequency is a dictionary
    #   given a word, it will tell you how many documents this word was present it
    # use the variable document_frequency 
    document_frequency = defaultdict(int)
    #####################Task t2e: your code #######################
    # The input is the file name with url and processed file names
    # for each file:
    #   get all the words and compute its frequency (over the entire corpus)
    # return the 5000 words with highest frequency
    # Hint: check out Counter class in Python
    #####################Task t2e: your code #######################

    myWordCount = Counter()
    f1 = open(file_name)
    for line1 in f1:
        filename = line1.split('|')[0]
        f2 = open(filename,"r")
        for line2 in f2:
            arr = str.split(line2)
            for word in arr:
                myWordCount[word] += 1
        f2.close()
    f1.close()

    top_5000_words = [word[0] for word in myWordCount.most_common(5000)]
    #print top_5000_words
    #calculate document frequency of top 5000 words
    f1 = open(file_name)
    for line1 in f1:
        #each of the files, check if word present in it
        for word in top_5000_words:
            f2 = open(line1.split('|')[0])
            for line2 in f2:
                if word in line2:
                    document_frequency[word] += 1
                    break
            f2.close()

    f = open("vocabulary.txt", "w")
    for word in top_5000_words:
        f.write(word + "," + str(document_frequency[word]) + "\n")
    f.close()

    return top_5000_words


def corpus_to_document_vec(vocabulary_file_name, file_name, output_file_name):
    #####################Task t2f: your code #######################
    # The input is the file names of vocabulary, and 
    #   the file  with url, processed file names  and the output file name
    #   the output is a file with tf-idf vector for each document
    #Pseudocode:
    # for each file:
    #   call the function text_to_vec with document body
    #   write the vector into output_file_name one line at a time
    #   into output_file_name
    #       ie document i will be in the i-th line
    #####################Task t2f: your code #######################
    f1 = open(file_name)
    f3 = open(output_file_name,'a')
    for line in f1:
        with open(line.split("|")[0],'r') as f2:
            tfidf = text_to_vec(vocabulary, f2.read())
            for item in tfidf:
                f3.write("%s " %item)
            f3.write('\n')

def text_to_vec(vocabulary, text):
    #####################Task t2g: your code #######################
    # The input are vocabulary and text
    #   compute its tf-idf vector (ignore all words not in vocabulary)
    #Remember to use the variable document_frequency for computing idf
    #####################Task t2g: your code #######################
    tfidf = []
    text1 = text.split()
    wordCount = Counter()
    for word in text1:
        wordCount[word] += 1
    totalwords = len(text1)
    #calculate  tf-idf
    for word in text1:
        if word in vocabulary:
            #add 1 to divisor to avoid "divide by zero"
            tfidf.append((wordCount[word] / totalwords) * (total_number_of_docs / (1 + document_frequency[word])))

    tfidf = tfidf/LA.norm(tfidf)
    return tfidf


def query_document_similarity(query_vec, document_vec):
    #####################Task t2h: your code #######################
    #   Given a query and document vector
    #   compute their cosine similarity
    cosine_similarity = None
    #####################Task t2h: your code #######################
    query_vec = query_vec.split()
    document_vec = document_vec.split()
    
    cosine_similarity_square = 0
    for i in range(len(query_vec)):
        cosine_similarity_square += (float(query_vec[i]) - float(document_vec[i])) ** 2
    
    cosine_similarity = cosine_similarity_square ** (0.5)
    
    return cosine_similarity


def rank_documents_tf_idf(query, k=10):
    #####################Task t2i: your code #######################

    #convert query to document using text_to_vec function
    query_as_document = None
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   for each matching document:
    #       retrieve its tf-idf vector (use the file_name and index fields from mongodb)
    #   compute the tf-idf score and sort them accordingly 
    # return top-k documents
    #####################Task t2i: your code #######################
    
    # Transforming query using process_text
    query = process_text(query)
    
    # Creating set of unique words from query
    words_in_query = set(query.split())
    
    # Creating query vector
    query_vector = ""
    #for each query calculate the tfidf
    tfidf = text_to_vec(vocabulary, query)
    for item in tfidf:
        query_vector += str(item) + " "
    
    similarity_candidates = []
    doc_counter = 0

    # Issuing the transformed query to mongodb
    webpages = mongodb_db.webpages
    cursor = webpages.find()
    
    # Getting matching documents - by the end of this loop, similarity_candidates will have indices of matching documents
    for row in cursor:
        for key, value in row.iteritems():
            found = False
            if not found and type(value) is list:
                for v in value:
                    words_in_v = set(v.split())
                    if len(words_in_query.intersection(words_in_v)) > 0:
                        similarity_candidates.append(doc_counter)
                        found = True
                        break
            if not found and (type(value) is str or type(value) is unicode):
                words_in_value = set(value.split())
                if len(words_in_query.intersection(words_in_value)) > 0:
                    similarity_candidates.append(doc_counter)
                    break
        doc_counter += 1
    
    # For each matching document, finding the cosine similarity with the query vector
    line_counter = 0
    document_similarity = {}
    with open('tf_idf_vector.txt', 'r') as vector_file:
        lines = vector_file.readlines()
        
    for line in lines:
        document_similarity[line_counter] = 0
        if line_counter in similarity_candidates:
            document_similarity[line_counter] = query_document_similarity(query_vector, line)
        line_counter += 1
    
    # Sort the similarity_candidats according to their similarity values
    sorted_vectors = sorted(document_similarity.items(), key=operator.itemgetter(1))[::-1]
    
    # From the list sorted according to their similarities, find the document indices
    ranked_documents = []
    for rdoc in sorted_vectors:
        ranked_documents.append(rdoc[0])
    
    # Returning top k documents
    return ranked_documents[:k]


def rank_documents_zone_scoring(query, k=10):
    #####################Task t2j: your code #######################

    #convert query to document using text_to_vec function
    query_as_document = None
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   for each matching document compute its score as following:
    #       score = 0
    #       for each word in query:
    #           find which "zone" the word fell in and give appropriate score
    #           title = 0.3, h1 = 0.2, h2=0.1, h3=h4=h5=h6=0.05,a: 0.1, body: 0.1
    #   so if a query keyword occured in title, h1 and body, its score is 0.6
    #       compute this score for all keywords
    #       score of the document is the score of all keywords
    # return top-k documents
    #####################Task t2j: your code #######################
    
    # Transforming query using process_text
    query = process_text(query)
    
    # Creating set of unique words from query
    words_in_query = set(query.split())
    
    # Creating query vector
    query_vector = ""
    tfidf = text_to_vec(vocabulary, query)
    for item in tfidf:
        query_vector += str(item) + " "
    
    document_scores = {}
    doc_counter = 0

    # Issuing the transformed query to mongodb
    webpages = mongodb_db.webpages
    cursor = webpages.find()
    
    # Getting matching documents - by the end of this loop, similarity_candidates will have indices of matching documents
    for row in cursor:
        score = 0
        for key, value in row.iteritems():
            if key == 'title':
                words_in_value = set(value.split())
                if len(words_in_query.intersection(words_in_value)) > 0:
                    score += 0.3
            elif key == 'h1':
                for v in value:
                    words_in_v = set(v.split())
                    if len(words_in_query.intersection(words_in_v)) > 0:
                        score += 0.2
                        break
            elif key == 'h2':
                for v in value:
                    words_in_v = set(v.split())
                    if len(words_in_query.intersection(words_in_v)) > 0:
                        score += 0.1
                        break
            elif key in ['h3', 'h4', 'h5', 'h6']:
                for v in value:
                    words_in_v = set(v.split())
                    if len(words_in_query.intersection(words_in_v)) > 0:
                        score += 0.05
                        break
            elif key == 'a':
                for v in value:
                    words_in_v = set(v.split())
                    if len(words_in_query.intersection(words_in_v)) > 0:
                        score += 0.1
                        break
            elif key == 'body':
                words_in_value = set(value.split())
                if len(words_in_query.intersection(words_in_value)) > 0:
                    score += 0.1
            
        document_scores[doc_counter] = score
        doc_counter += 1
        
    # Sort the similarity_candidats according to their similarity values
    sorted_documents = sorted(document_scores.items(), key=operator.itemgetter(1))[::-1]
    
    # From the list sorted according to their similarities, find the document indices
    ranked_documents = []
    for rdoc in sorted_documents:
        ranked_documents.append(rdoc[0])
    
    return ranked_documents[:k]


def rank_documents_pagerank(query, k=10):
    #####################Task t2k: your code #######################

    #convert query to document using text_to_vec function
    query_as_document = None
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   order the documents based on their pagerank score (computed in task 3)
    # return top-k documents
    #####################Task t2k: your code #######################
    
    # Transforming query using process_text
    query = process_text(query)
    
    # Creating set of unique words from query
    words_in_query = set(query.split())
    
    similarity_candidates = []
    doc_counter = 0

    # Issuing the transformed query to mongodb
    webpages = mongodb_db.webpages
    cursor = webpages.find()
    
    # Getting matching documents - by the end of this loop, similarity_candidates will have indices of matching documents
    for row in cursor:
        for key, value in row.iteritems():
            found = False
            if not found and type(value) is list:
                for v in value:
                    words_in_v = set(v.split())
                    if len(words_in_query.intersection(words_in_v)) > 0:
                        similarity_candidates.append(doc_counter)
                        found = True
                        break
            if not found and (type(value) is str or type(value) is unicode):
                words_in_value = set(value.split())
                if len(words_in_query.intersection(words_in_value)) > 0:
                    similarity_candidates.append(doc_counter)
                    break
        doc_counter += 1
    
    with open('page_rank.txt') as f:
        lines = f.readlines()
        
    document_pagerank_similarities = {}
    doc_counter = 0
    for line in lines:
        score = 0
        if doc_counter in similarity_candidates:
            score = float(line.split()[1].split(']')[0])
        document_pagerank_similarities[doc_counter] = score
        doc_counter += 1
    
    # Sort the similarity_candidats according to their similarity values
    sorted_documents = sorted(document_pagerank_similarities.items(), key=operator.itemgetter(1))[::-1]
    
    # From the list sorted according to their similarities, find the document indices
    ranked_documents = []
    for rdoc in sorted_documents:
        ranked_documents.append(rdoc[0])
    
    return ranked_documents[:k]


#Do not change below
def rank_documents(query):
    print "Ranking documents for query:", query
    print "Top-k for TF-IDF"
    print rank_documents_tf_idf(query)
    print "Top-k for Zone Score"
    print rank_documents_zone_scoring(query)
    print "Top-k for Page Rank"
    print rank_documents_pagerank(query)


setup_mongodb()
#####Uncomment the following functions as needed
process_document_corpus("fileNamesToUUID.txt")
vocabulary = find_corpus_vocabulary("processedFileNamesToUUID.txt")
corpus_to_document_vec("vocabulary.txt", "processedFileNamesToUUID.txt", "tf_idf_vector.txt")