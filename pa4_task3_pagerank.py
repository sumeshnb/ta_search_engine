import numpy as np
from  scipy import sparse

import time
import networkx as nx

import os.path
import collections
import urlparse
from bs4 import BeautifulSoup



#Pagerank code from Samuel
###############Do not change below
def compute_PageRank(G, beta=0.85, epsilon=10**-4):
    '''
    Efficient computation of the PageRank values using a sparse adjacency 
    matrix and the iterative power method.
    
    Parameters
    ----------
    G : boolean adjacency matrix. np.bool8
        If the element j,i is True, means that there is a link from i to j.
    beta: 1-teleportation probability.
    epsilon: stop condition. Minimum allowed amount of change in the PageRanks
        between iterations.

    Returns
    -------
    output : tuple
        PageRank array normalized top one.
        Number of iterations.

    '''    
    #Test adjacency matrix is OK
    n,_ = G.shape
    assert(G.shape==(n,n))
    #Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T/beta #vector
    #Initialize
    ranks = np.ones((n,1))/n #vector
    time = 0
    flag = True
    while flag:        
        time +=1
        with np.errstate(divide='ignore'): # Ignore division by 0 on ranks/deg_out_beta
            new_ranks = G.dot((ranks/deg_out_beta)) #vector
        #Leaked PageRank
        new_ranks += (1-new_ranks.sum())/n
        #Stop condition
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False        
        ranks = new_ranks
    return(ranks, time)


################In this task, you will evaluate Pagerank on the UTA graph


def construct_digraph(file_name, output_file_name):
    #Your input file will have a structure similar to fileNamesToUUID.txt
    #   ie each line has two columns separated by | (i.e. filename, url)
    # Analyze the file in two passes:
    #   in the first pass, assign node ids to each url/file
    #       for eg, web page i on line i has node index i
    #       be careful though: it is possible that same url occurs multiple times in the document
    #       in that case treat them as two different pages
    #   in the second pass
    #       Create a directed graph as follows:
    #           each web page is a node
    #           parse the a tag in the document  and add an edge to where it points
    #           if the link is not in our corpus, ignore it
    #           if there a link from url X to url Y
    #           and url Y occurs multiple times in the file say in lines a,b,c
    #           and edge from X=>a, X=>b and X=>c
    #  DO NOT CONSTRUCT A DIRECTED GRAPH USING ADJACENCY MATRIX
    #   your computer will not have enough RAM
    # So instead store the data in the edge list format in output_file_name
    #   without explicitly constructing the graph
    # Write the following in output_file_name
    #   first line says the number of nodes in the graph
    #   for each edge (u,v) in the graph:
    #       write u,v in the output_file_name one line for each edge
    #####################Task t3a: your code below#######################
    #####################Task t3a: your code below#######################
    myfile = open('fileNamesToUUID.txt','r')
	lines = myfile.readlines()
	myfile.close()
	
	urlList = list()
	for line in lines:
		urlSplit = line.split('|')[1].split('\n')[0]
		urlList.append(urlSplit)
	
	keys=urlList
	values= range(len(urlList))
	urlHash = collections.OrderedDict(zip(keys, values))

	G=nx.DiGraph()
	for key in urlHash.keys():
		G.add_node(urlHash[key])
		
	
	for filename in os.listdir(os.getcwd() + "/downloads"):
		soup = BeautifulSoup(open(os.getcwd() + "/downloads/" + filename))
		links = soup.find_all('a')
		fullLink = list()
		for link in links:
			fullLink.append(link.get('href'))
		for link in fullLink:
			#if not link == None and link.endswith((".html", ".htm", ".php")) and not link.startswith("htt"):
			if not link == None:
				# if link.endswith((".html", ".htm", ".php")) and not link.startswith("htt"):
				link = urlparse.urljoin(key,link)
			for key in urlHash.keys():
				if link in urlHash.keys():
					G.add_edge(urlHash[key],urlHash[link])
	
	number_of_nodes = len(G.nodes())
	filename = os.getcwd() + "/output_di_graph.txt"
	f = open(filename,"w")
	f.write(str(number_of_nodes) + "\n")
	f.close()  
	f = open(filename,"a")
	for edges in G.edges():
		f.write(str(edges).strip('()') + "\n") 
	f.close()                                                                                

def construct_sparse_graph_dictionary_of_keys(graph_file_name):
    #If you create a graph for UTA using traditional methods
    #   such as adjacency matrix (which we will need for pagerank)
    #   it might take tens of GB
    # So we will represent the graph as a sparse matrix
    # In this code, you will read the input file (graph that you wrote in construct_digraph
    #   and convert it to a sparse matrix with dictionary of keys (DoK) encoding
    #   you might want to read https://scipy-lectures.github.io/advanced/scipy_sparse/storage_schemes.html
    #       or http://scipy-lectures.github.io/advanced/scipy_sparse/
    #####################Task t3b: your code below#######################
	with open(graph_file_name,'r') as f:
		number_of_nodes = int(f.readline())
		dok_mtx = sparse.dok_matrix((number_of_nodes,number_of_nodes),dtype=np.float64)
		# dok_mtx = sparse.dok_matrix((number_of_nodes,number_of_nodes),dtype=np.bool)
		for line in f.readlines():
			start, end = (int(x) for x in line.split(','))
			dok_mtx[start,end] = 1.0
			# dok_mtx[start,end] = True
	
	# G = dok_mtx.todense()
	G = dok_mtx
	f.close()
    #####################Task t3b: your code below#######################
    # print "Graph size through DoK is ", G.nbytes
    return G


def construct_sparse_graph_compressed_sparse_row(graph_file_name):
    #If you create a graph for UTA using traditional methods
    #   such as adjacency matrix (which we will need for pagerank)
    #   it might take tens of GB
    # So we will represent the graph as a sparse matrix
    # In this code, you will read the input file (graph that you wrote in construct_digraph
    #   and convert it to a sparse matrix with compressed sparse row (CSR) format
    #   you might want to read https://scipy-lectures.github.io/advanced/scipy_sparse/storage_schemes.html
    #       or http://scipy-lectures.github.io/advanced/scipy_sparse/
    #####################Task t3c: your code below#######################
	row = []
	col = []
	number_of_edges = 0
	with open(graph_file_name,'r') as f:
		number_of_nodes = int(f.readline())
		for line in f.readlines():
			number_of_edges = number_of_edges+1
			start, end = (int(x) for x in line.split(','))
			row.append(start)
			col.append(end)

	G = sparse.csr_matrix(([1.0]*number_of_edges,(row,col)),shape=(number_of_nodes,number_of_nodes))
	# G = G.todense()
	f.close()
    #####################Task t3c: your code below#######################
    # print "Graph size through CSR is ", G.nbytes
	return G 

def construct_sparse_graph_networkx(graph_file_name):
    #In this task, we will compare our method with NetworkX
    #   one of the state of the art graph analytics platorm
    # Networkx has a from_edgelist function that accepts an array of edges
    #   and construct a graph from it.
    #Read the input file and popular the variable edge_list
    #####################Task t3d: your code below#######################
    edge_list = None
    #####################Task t3d: your code below#######################
    with open(graph_file_name,'r') as f:
		edge_list = [
		tuple(int(x) for x in line.split(','))
        for line in f.readlines()[1:]
    ] 
	
    G = nx.from_edgelist(edge_list, create_using=nx.DiGraph())
	# print G.edges()[:10]
    return G 


def persist_pagerank(graph_file_name, output_file_name):
    #####################Task t3e: your code below#######################
    #Construct the graph using some construct algorithm
    # Compute the pagerank
    #For each web page compute its pagerank (score) and write it in output_file_name
    #####################Task t3e: your code below#######################
    # pass 
	G = construct_sparse_graph_dictionary_of_keys(graph_file_name)
	filename = os.getcwd() + "/" + output_file_name
	f = open(filename,"w")
	f.write(compute_PageRank(G) + "\n")
	f.close()
	

def compare_pagerank_algorithms(graph_file_name):
    algo_name = ["PageRank-DOK", "PageRank-CSR", "PageRank-NetworkX"]
    algo_fns = [construct_sparse_graph_dictionary_of_keys, construct_sparse_graph_compressed_sparse_row, construct_sparse_graph_networkx]

    for i in range(len(algo_name)):
        print "Testing:", algo_name[i]

        start_time = time.time()
        G = algo_fns[i](graph_file_name)
        end_time = time.time()

        time_for_graph_construction = end_time - start_time

        start_time = time.time()
        if algo_name[i] == "PageRank-NetworkX":
            nx.pagerank(G)
        else:
            compute_PageRank(G)

        end_time = time.time()
        time_for_pagerank_computation = end_time - start_time
        total_time = time_for_graph_construction + time_for_pagerank_computation


        print "Time for graph, page rank and total", time_for_graph_construction, time_for_pagerank_computation, total_time
