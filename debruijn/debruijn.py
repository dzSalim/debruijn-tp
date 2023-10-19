#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
import networkx as nx
import matplotlib
from operator import itemgetter
import random
random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from collections import Counter

__author__ = "Agsous Salim"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Agsous Salim"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Agsous Salim"
__email__ = "salim.agsous99@gmail.comr"
__status__ = "Developpement"

def isfile(path): # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file
    
    :raises ArgumentTypeError: If file doesn't exist
    
    :return: (str) Path 
    """
    if not os.path.isfile(path):
        if os.path.isdir(path):
            msg = "{0} is a directory".format(path)
        else:
            msg = "{0} does not exist.".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', dest='fastq_file', type=isfile,
                        required=True, help="Fastq file")
    parser.add_argument('-k', dest='kmer_size', type=int,
                        default=22, help="k-mer size (default 22)")
    parser.add_argument('-o', dest='output_file', type=str,
                        default=os.curdir + os.sep + "contigs.fasta",
                        help="Output contigs in fasta file (default contigs.fasta)")
    parser.add_argument('-f', dest='graphimg_file', type=str,
                        help="Save graph as an image (png)")
    return parser.parse_args()


def read_fastq(fastq_file):
    """Extract reads from fastq files.

    :param fastq_file: (str) Path to the fastq file.
    :return: A generator object that iterate the read sequences. 
    """
    with open(fastq_file, 'r') as fasta:
        for line in fasta:
            yield next(fasta).strip("\n")
            next(fasta)
            next(fasta)


def cut_kmer(read, kmer_size):
    """Cut read into kmers of size kmer_size.
    
    :param read: (str) Sequence of a read.
    :return: A generator object that iterate the kmers of size kmer_size.
    """
    for i in range(0, len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]


def build_kmer_dict(fastq_file, kmer_size):
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    sequences = read_fastq(fastq_file)
    kmer_dict = Counter()
    for sequence in sequences:
        kmers = cut_kmer(sequence.strip(), kmer_size)
        kmer_dict.update(kmers)

    return kmer_dict


def build_graph(kmer_dict):
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    digraph = nx.DiGraph()
    for keys in kmer_dict.items():
        digraph.add_edge(keys[0][0:-1], keys[0][1:], weight=keys[1])

    return digraph


def remove_paths(graph, path_list, delete_entry_node, delete_sink_node):
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path 
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if (type(path_list[0]) != int):
        for path in path_list:
            if (delete_entry_node and delete_sink_node):
                graph.remove_nodes_from(path)
            elif (delete_entry_node):
                graph.remove_nodes_from(path[:-1])
            elif (delete_sink_node):
                graph.remove_nodes_from(path[1:])
            else:
                graph.remove_nodes_from(path[1:-1])
    else:
        if (delete_entry_node and delete_sink_node):
            graph.remove_nodes_from(path_list)
        elif (delete_entry_node):
            graph.remove_nodes_from(path_list[:-1])
        elif (delete_sink_node):
            graph.remove_nodes_from(path_list[1:])
        else:
            graph.remove_nodes_from(path_list[1:-1])
    return graph


def select_best_path(graph, path_list, path_length, weight_avg_list, 
                     delete_entry_node=False, delete_sink_node=False):
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path 
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if (statistics.stdev(weight_avg_list) > 0):
        ind = weight_avg_list.index(max(weight_avg_list))
        #remove the weight from the more recurrent path
        path_list.pop(ind)
        graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    elif (statistics.stdev(path_length) > 0):
        ind = path_length.index(max(path_length))
        #remove the longest path
        path_list.pop(ind)
        graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    else:
        random.seed(9001)
        r = random.randint(0, len(path_list))
        # remove randomly a path
        path_list.pop(r)
        graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)

    return graph

def path_average_weight(graph, path):
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean([d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)])

def solve_bubble(graph, ancestor_node, descendant_node):
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph 
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    list_of_path = list(nx.all_simple_paths(graph, ancestor_node, descendant_node))
    len_path = []
    weight_avg_list = []
    for i in list_of_path:
        w = []
        len_path.append(len(i))
        weight_l = list(graph.subgraph(i).edges(data=True))
        for j in weight_l:
            # take d where the value of weight is
            w.append(j[2]['weight'])
        weight_avg_list.append(statistics.mean(w))
    # select the better path
    graph = select_best_path(graph,list_of_path,len_path,weight_avg_list,
                             delete_entry_node=False,delete_sink_node=False)

    return graph
def simplify_bubbles(graph):
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for i in graph.nodes:
        node = i
        list_prd = list(graph.predecessors(i))
        if len(list_prd) > 1:
            for j in list_prd:
                for pre in list_prd:
                    node_ancestor = nx.lowest_common_ancestor(graph, pre, j)
                    if node_ancestor != None:
                        bubble = True
                        break
        if bubble:
            break
    # La simplification ayant pour conséquence de supprimer des noeuds du hash
    # Une approche récursive est nécessaire avec networkx
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, node_ancestor, node))

    return graph


def solve_entry_tips(graph, starting_nodes):
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    start_node_delete = []
    for start_node in starting_nodes:
        list_predecessors = list(graph.predecessors(start_node))
        list_successors = list(graph.successors(start_node))

        if len(list_predecessors) > 0:
            start_node_delete.append(start_node)
        elif len(list_successors) > 0:
            start_node_delete.append(start_node)
        else:
            return graph

    for start_node in start_node_delete:
        graph.remove_node(start_node)
    return graph


def solve_out_tips(graph, ending_nodes):
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    end_node_delete = []
    for end_node in ending_nodes:
        list_predecessors = list(graph.predecessors(end_node))
        list_successors = list(graph.successors(end_node))

        if len(list_predecessors) > 0:
            end_node_delete.append(end_node)
        elif len(list_successors) > 0:
            end_node_delete.append(end_node)
        else:
            return graph

    for end_node in end_node_delete:
        graph.remove_node(end_node)
    return graph

def get_starting_nodes(graph):
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    first_nodes = []
    for i in graph.nodes:
        prd = list(graph.predecessors(i))
        if len(prd) == 0:
            first_nodes.append(i)

    return first_nodes

def get_sink_nodes(graph):
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    last_nodes = []
    for i in graph.nodes:
        scs = list(graph.successors(i))
        if len(scs) == 0:
            last_nodes.append(i)

    return last_nodes

def get_contigs(graph, starting_nodes, ending_nodes):
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object 
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    list_contigs = []
    seq = ""
    for i in starting_nodes:
        for j in ending_nodes:
            if (nx.has_path(graph, i, j)):
                for k in nx.all_simple_paths(graph, source=i, target=j):
                    seq += i
                    for kmer in k[1:]:
                        seq += kmer[-1]
                    list_contigs.append((seq, len(seq)))
                    seq = ""
    return list_contigs

def save_contigs(contigs_list, output_file):
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (str) Path to the output file
    """
    file = open(output_file + ".fasta", "w")
    count = 0
    for contig in contigs_list:
        file.write(textwrap.fill(">contig_" + str(count) + " len=" + str(contig[1]), width=80))
        file.write("\n")
        file.write(textwrap.fill(contig[0], width=80))
        file.write("\n")
    file.close()



def draw_graph(graph, graphimg_file): # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (str) Path to the output file
    """                                   
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 3]
    #print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 3]
    #print(elarge)
    # Draw the graph with networkx
    #pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=6, alpha=0.5, 
                           edge_color='b', style='dashed')
    #nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file)


#==============================================================
# Main program
#==============================================================
def main(): # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    dictio_kmer = build_kmer_dict(args.fastq_file, args.kmer_size)
    print('============================================================ Etape 1 ============================================================')
    graph = build_graph(dictio_kmer)
    print('============================================================ Etape 2 ============================================================')
    start_node = get_starting_nodes(graph)
    sink_node = get_sink_nodes(graph)
    print('============================================================ Etape 3 ============================================================')
    graph = simplify_bubbles(graph)
    print('============================================================ Etape 4 ============================================================')
    graph = solve_entry_tips(graph,start_node)
    print('============================================================ Etape 5 ============================================================')
    graph = solve_out_tips(graph, sink_node)
    print('============================================================ Etape 6 ============================================================')
    contigs = get_contigs(graph,get_starting_nodes(graph),get_sink_nodes(graph))
    print('============================================================ Etape 7 ============================================================')
    save_contigs(contigs,args.output_file)
    print('============================================================ Etape 8 ============================================================')

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit 
    # graphe
    # Plot the graph
    #if args.graphimg_file:
    #    draw_graph(graph, args.graphimg_file)


if __name__ == '__main__': # pragma: no cover
    main()
