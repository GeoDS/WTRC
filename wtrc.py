import argparse
import copy
import datetime
import importlib
import math
import multiprocessing
import os
import random
import sys
import time
import timeit
import uuid
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Manager, Pool, Process, Queue, Value, cpu_count
from random import choice, sample

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def output_path(args, kind, ext="npy"):
    """Path for one of a scan's output files.

    The name is built from the settings that define the scan, so a run can be
    identified from its filename alone: network type, district, and timestep
    range. `kind` separates the files a single scan writes -- "scan" for the
    rich club coefficients, "max_t" for the starting timestep of the strongest
    window, "geoids" for the club members, "m_s" for the per-window matrices.
    """
    stem = (f"{args.network_type}_rich_club_dis{args.district}"
            f"_t{args.ti}-{args.t}_{kind}")
    return f"{args.path_prefix}{stem}.{ext}"


def filter_flows_to_district(flows_df, district):
    # load district boundaries
    district_boundaries = gpd.read_file("wi_cong_adopted_2022/POLYGON.shp")
    district_boundaries = district_boundaries.to_crs("EPSG:3071")
    district_boundaries['NAME'] = district_boundaries['NAME'].astype(int) 
    district_polygon = district_boundaries[district_boundaries['NAME'] == district].geometry.iloc[0]
    # load census tracts
    # cts = './wi_ct_boundaries_2020/wi_t_2020_bound.shp'
    cts = ("zip://wi_ct_boundaries_2018/WI_CensusCBF_Tracts_2018.zip"
           "!WI_CensusCBF_Tracts_2018/cb_2018_55_tract_500k.shp")
    cts = gpd.read_file(cts)
    cts = cts.to_crs("EPSG:3071")
    cts['GEOID'] = cts['GEOID'].astype(int)
    # filter census tracts by spatial intersect (mostly within district boundary)
    buffered_polygon = district_polygon.buffer(0)
    # Step 3: Filter gdf to include geometries with more than 70% of their area within the buffered polygon
    filtered_cts = cts[cts.geometry.apply(lambda x: x.intersection(buffered_polygon).area / x.area > 0.5)] # did .6 for dis 2 and 3 analysis, but .5 for other level
    # Step 4: Plot filtered geometries
    ax = filtered_cts.plot(figsize=(6, 6), color='lightblue', edgecolor='black')
    # Step 5: Plot the boundary of the buffered polygon
    boundary = gpd.GeoSeries([buffered_polygon.boundary])
    boundary.plot(ax=ax, color='red')
    # Optional: Set axis off
    plt.axis('off')
    plt.show()
    
    # filter flows_df to geoids in within boundary
    valid_geoids = set(filtered_cts['GEOID'])
    # Filter flows_df for rows where both geoid_o and geoid_d are in valid_geoids
    flows_df_filtered = flows_df[flows_df['geoid_o'].isin(valid_geoids) & flows_df['geoid_d'].isin(valid_geoids)]
    
    print(f'Count of CTs within boundary: {len(valid_geoids)}')
    # flows_ct_count = set(filtered_flows_df.geoid_o,filtered_flows_df.geoid_d)
    
    geoids_set = np.union1d(flows_df_filtered.geoid_o, flows_df_filtered.geoid_d) # this could be less than valid_geoids
    node_geoid_dict = {k: geoids_set[k] for k in range(len(geoids_set))}
    geoid_node_dict = {geoids_set[k]: k for k in range(len(geoids_set))}
    print(f'Count of geometries in filtered_flows_df: {len(geoids_set)}')
    flows_df_filtered = flows_df_filtered.drop(columns=['Unnamed: 0'], errors='ignore')
    flows_df_filtered['i'] = flows_df_filtered['geoid_o'].map(geoid_node_dict)
    flows_df_filtered['j'] = flows_df_filtered['geoid_d'].map(geoid_node_dict)
    
    return flows_df_filtered, node_geoid_dict, boundary, filtered_cts


def plot_rich_club_scan(ks, deltas, RC_norm, args):
    # Replace infinities and NaNs
    RC_norm = np.nan_to_num(RC_norm, nan=-99)
    RC_norm_masked = np.ma.masked_where(RC_norm == -99, RC_norm)

    # Create a custom colormap and set the 'bad' value color
    cmap = cm.PiYG.copy()
    cmap.set_bad('grey')

    # Normalization
    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.2)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjusted figsize
    im = ax.imshow(RC_norm_masked, cmap=cmap, norm=norm, aspect='auto')  # Using 'auto' aspect

    # Setting the x and y ticks to the borders of each cell
    ax.set_xticks(np.arange(len(ks)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(deltas)) - 0.5, minor=True)

    # Grid lines based on minor ticks
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.33)
    ax.tick_params(which="minor", size=0)

    # Setting the x and y major tick labels
    ax.set_xticks(np.arange(len(ks)))
    ax.set_yticks(np.arange(len(deltas)))
    ax.set_xticklabels(ks, rotation=45)
    ax.set_yticklabels(deltas)

    # Loop over data dimensions and create text annotations for cell values
    for i in range(len(deltas)):
        for j in range(len(ks)):
            ax.text(j, i, f'{RC_norm_masked[i, j]:.2f}', 
                    ha="center", va="center", color="black")

    ax.set_ylabel(r'$\Delta$ (time-lag over which TRC is present)')
    ax.set_xlabel('Temporal Edge Count (Richness Sequence)')  

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)

    # Create the colorbar in the new axes
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1.0, 2.2])
    cbar.ax.set_yticklabels(['0', '1.0', '2.2'])
    cbar.ax.set_ylabel(r'$M(k,\Delta)_{norm}$', rotation=-90, va="bottom")

    # Save and show the plot
    path = output_path(args, "scan", ext="png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_congressional_districts():
    gdf = gpd.read_file("./wi_cong_adopted_2022/POLYGON.shp")
    # gdf = gdf.to_crs("EPSG:32616")
    gdf = gdf.to_crs("EPSG:3071")
    gdf['centroid'] = gdf.geometry.centroid
    plt.figure(figsize=(3, 2))
    ax = gdf.plot('NAME', edgecolor='black')
    # Annotate each polygon with its name
    for idx, row in gdf.iterrows():
        plt.annotate(text=row['NAME'], xy=(row['centroid'].x, row['centroid'].y),
                     horizontalalignment='center')
     # Remove axes
    ax.set_axis_off()
    plt.show()
    return gdf


# Some links
# Randomizations github: https://github.com/mgenois/RandTempNet/blob/master/randomisations.py

# Randomizations paper: https://arxiv.org/pdf/1806.04032.pdf

# https://github.com/mgenois/RandTempNet/blob/059c8ec1ed4e18dba2ded3eeb0def2036b2ee637/classes.py#L207
# link_timeline.display() returns a list of tuples ((i,j),[t...])


def load_randomized_graphs(graphs_list, args):
    # read in one randomized graph
    # the csvs in the directory listed below have already been randomized with P__w_t, and so we just load them here
        # Step 1: Identify the CSV files
    # directory = '/media/raid/jkruse/Temporal-Rich-Club/Human_Mobility_Flows/airport_wtrc/randomized_dfs'
    if args.save_file == 'airports':
        directory = '/media/raid/jkruse/Temporal-Rich-Club/US Air Traffic TN/'
        print(f'directory = {directory}')
    if 'count' in args.save_file:
        directory = '/media/raid/jkruse/Temporal-Rich-Club/Human_Mobility_Flows/us_counties/us_counties_weekly_flows_52/'
        print(f'directory = {directory}')
    # pattern = 'tij_airport_2012_2020_RANDOMIZED_*.csv'
    # num_graphs = len(graphs_list)
    # print(f'len(graphs_list) = {len(graphs_list)}')
    file_list = [f for f in os.listdir(directory)]
    if args.save_file == 'airports':
        file_list = [f for f in file_list if 'SEGMENT' in f and 'pass5' not in f]
        # file_list = [f for f in file_list if 'SEGMENT' in f and 'pass5' in f]
        # print('using pass5')
    # print(f'file_list[0:5] = {file_list[0:5]}')
    # Step 2: Randomly sample num_graphs many of these files
#   this should always be 1, because topotemporal randomization is called args.shuffle number of times
    sampled_files = random.sample(file_list, 1)
    print(f'sampled_files = ',sampled_files)
    graphs_randomized = []
    # Step 3: Process each sampled file
    for file_name in sampled_files:
        # Read the file into a DataFrame
        df = pd.read_csv(os.path.join(directory, file_name))
        # Add a weight column with all values set to 1
        df['weight'] = 1
        df['geoid_o'] = -1
        df['geoid_d'] = -1

        # Step 4: Call build_graphs on the DataFrame
        # Assuming build_graphs is a function that accepts a DataFrame and returns a graph object
        # Replace x, nodes, iis, jjs with actual values or variables
        iis=np.unique(df['i'])
        jjs=np.unique(df['j'])
        # nodes=np.union1d(iis,jjs)
        combined_unique = np.unique(np.concatenate((iis, jjs)))
        # Count the number of unique values
        nodes = len(combined_unique)
        x = max(df['t'])

        graphs_list_dc, AGG = build_graphs(df, x, nodes, iis, jjs)
        # graphs_randomized.append(graphs_list_dc)

    return graphs_list_dc



def randomize_preserving_strength(i,G):
    # https://github.com/jeffalstott/richclub/blob/master/richclub.py
    # - assuming bc most of the processes are launched at the same second
    seed = uuid.uuid4().int & (1<<32)-1
    np.random.seed(seed) # this seed is almost guaranteed to be unique
    G_copy = deepcopy(G)
    ### NODES ARE NOT ITERATED THROUGH SEQUENTIALLY
    nodes = sorted(G_copy.nodes())
    degrees = np.array([G_copy.degree(n) for n in nodes])
    strengths = np.array([G_copy.degree(n, weight='weight') for n in nodes])
    # print(f'len of nodes, degrees, strengths: {len(nodes)}, {len(degrees)}, {len(strengths)}')
    # print(f'nodes = {nodes}')

    # Permute strengths
    for degree in np.unique(degrees):
        if degree != 0:
            indices = np.where(degrees == degree)[0]
            shuffled_strengths = np.random.permutation(strengths[indices])
            strengths[indices] = shuffled_strengths
    
    # Recalculate weights
    # Filter out zero values from degrees and strengths
    filtered_degrees = degrees[degrees != 0]
    filtered_strengths = strengths[strengths != 0]

    # Calculate mean of filtered degrees and strengths
    mean_degree = np.mean(filtered_degrees)
    mean_strength = np.mean(filtered_strengths)

    for u, v, data in G_copy.edges(data=True): # should only find existing edges
        try:
            data['weight'] = (mean_degree / mean_strength) * strengths[u] * strengths[v] / (degrees[u] * degrees[v])
        except (IndexError, KeyError):
            # the edge keeps its original weight, so the randomization for this
            # edge is not applied -- report it rather than failing silently
            print(f'FAILED: u,v,data = {u},{v},{data}, {degrees[u]}, {degrees[v]}')
    return G_copy



def randomize_series_preserving_strength(graphs_list):
    # https://github.com/jeffalstott/richclub/blob/master/richclub.py
    new_graphs = []
    for i, G in enumerate(graphs_list):
        G_shuffled_w = randomize_preserving_strength(i,G)
        new_graphs.append(G_shuffled_w)

    return new_graphs

def double_edge_swap(G, nswap=1, max_tries=100, seed=None):
    #https://networkx.org/documentation/stable/_modules/networkx/algorithms/swap.html#connected_double_edge_swap
    # -> Dani says this will converge faster, and is the right sampler
    seed_n = uuid.uuid4().int & (1<<32)-1
    np.random.seed(seed_n) # this seed is almost guaranteed to be unique
    
    if G.is_directed():
        raise nx.NetworkXError(
            "double_edge_swap() not defined for directed graphs. Use directed_edge_swap instead."
        )
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise nx.NetworkXError("Graph has fewer than 2 edges")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        v = np.random.choice(list(G[u]), size=1)[0]
        y = np.random.choice(list(G[x]), size=1)[0]
        if v == y:
            continue  # same target, skip
        if (x not in G[u]) and (y not in G[v]):  # don't create parallel edges
            G.add_edge(u, x, weight=1)
            G.add_edge(v, y, weight=1)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            swapcount += 1
        if n >= max_tries:
            e = (
                f"Maximum number of swap attempts ({n}) exceeded "
                f"before desired swaps achieved ({nswap})."
            )
            raise nx.NetworkXAlgorithmError(e)
        n += 1
    return G

def sample_degree_sequence(graphs_list):
    new_graphs = []
    # initialize swap numbers based on first graph
    Q = 1
    G = graphs_list[0]
    E = G.number_of_edges() 
    number_of_swaps = int(Q * E / 1)
    max_attempts = number_of_swaps*5000
    
    for G in graphs_list: 
        G = double_edge_swap(G, nswap=number_of_swaps, max_tries=max_attempts)
        new_graphs.append(G)
    return new_graphs



def shuffle_sequence(graphs_list):
    # P[pT (Γ)]: P__pGamma from https://arxiv.org/pdf/1806.04032v3.pdf
    # https://github.com/mgenois/RandTempNet/blob/master/randomisations.py
    """
    Randomly shuffle the order of graphs in a list.
        Paper describing use: https://arxiv.org/pdf/1806.04032v3.pdf

    Parameters:
        graphs_list (list): A list of graphs.

    Returns:
        list: A new list with the order of graphs shuffled.
    """
    #Make a deep copy of the list of graphs
    shuffled_graphs = copy.deepcopy(graphs_list)
    
    # Randomly shuffle the order of graphs
    random.shuffle(shuffled_graphs)
    
    return shuffled_graphs

def randomize_topology_and_time(input_list, dt=1):
    print(f'starting randomize_topology_and_time...')
    graphs_list, AGG, args = input_list
    
    if args.randomize == 'pwt':
        print('Using pwt for randomization')
        graphs_list = load_randomized_graphs(graphs_list, args)
        return graphs_list, AGG 
    
    graphs_list = shuffle_sequence(graphs_list) # shuffle temporal order
    print('Finished shuffle_sequence(graphs_list)')
    
    if args.just_sequence == "True":
        print('Using just sequence shuffling for randomization')
        print('Only doing shuffle_sequence() for randomization')
        return graphs_list, AGG
    
    
    if args.network_type == 'topological' or args.network_type == 'trc':
        print('Using edge swapping for randomization')
        graphs_list = sample_degree_sequence(graphs_list)

    elif args.network_type == 'weighted':
        print('Using weight decorrelation for randomization')
        graphs_list = randomize_series_preserving_strength(graphs_list)
    
    # we don't use the randomized agg graphs, so just return the original (return to keep functions happy)
        # -> richness sequence is calculated with the original AGG each time
    return graphs_list, AGG


########################### defining functions ###################
def max_node_index(df):
    i_max = df.i.max()
    j_max = df.j.max()
    max_ind = 0
    if (i_max > j_max):
        max_ind = i_max
    else:
        max_ind = j_max
    return max_ind


def build_graphs(df, x, nodes, iis, jjs):
    start = time.time()
    graphs_list = []
    # t already starts from 0, so need to add 1
    x = int(x)
    for t in range(x+1): # add each temporal snapshot to the graph
        # print(f't step being added to graphs list: {t}')
        fr=df[df['t']==t][['i', 'j', 'weight', 't', 'geoid_o','geoid_d']].copy(deep=True)
        # 'i' and 'j': These are the names of the columns in fr that represent the nodes at either end of each edge. So, every row in fr is expected to have an edge between the nodes named in its 'i' and 'j' columns.
        fr['weight'] = pd.to_numeric(fr['weight'], downcast='integer', errors='coerce').fillna(0).astype(int)
        fr['t'] = pd.to_numeric(fr['t'], downcast='integer', errors='coerce').fillna(-99).astype(int)
        g = nx.from_pandas_edgelist(fr,'i','j', create_using=nx.Graph, edge_attr=['weight','t'])


        # Prepare dictionaries for node attributes
        node_attributes_i = pd.Series(fr.geoid_o.values,index=fr.i).to_dict()
        node_attributes_j = pd.Series(fr.geoid_d.values,index=fr.j).to_dict()
        # Update node_id and geoid for 'i' nodes
        nx.set_node_attributes(g, node_attributes_i, 'geoid')
        # Since you're setting 'node_id' to the index itself, it's redundant if the index is already the node_id
        # However, if you still need to set it explicitly
        node_id_i = pd.Series(fr.i.values,index=fr.i).to_dict()
        nx.set_node_attributes(g, node_id_i, 'node_id')
        # Update node_id and geoid for 'j' nodes
        node_attributes_j = pd.Series(fr.geoid_d.values,index=fr.j).to_dict()
        nx.set_node_attributes(g, node_attributes_j, 'geoid')
        node_id_j = pd.Series(fr.j.values,index=fr.j).to_dict()
        nx.set_node_attributes(g, node_id_j, 'node_id')
        
        
        missing_is = [i for i in range(nodes) if i not in fr.i.unique()]
        missing_js = [j for j in range(nodes) if j not in fr.j.unique()]

        # # # this ensures that every t has all of the nodes that are present in the agg graph; no edges (as intended)
        g.add_nodes_from(missing_is)
        g.add_nodes_from(missing_js)
        ##########
        # g = recalculate_degrees_strengths_for_nodes(g)
        g.remove_edges_from(nx.selfloop_edges(g))
        edges_to_drop = []
        # Iterate over all edges and their attributes
        for u, v, attrs in g.edges(data=True):
            # Check if 'weight' or 't' attribute is missing
            if 'weight' not in attrs or 't' not in attrs:
                edges_to_drop.append((u, v))

        # Remove the identified edges from the graph
        g.remove_edges_from(edges_to_drop)
        # edges_missing_attrs = [(u, v) for u, v, attrs in g.edges(data=True) if 'weight' not in attrs or 't' not in attrs]
        # print(f"Edges still missing attributes: {edges_missing_attrs}")
                    # update_aggregate_weights(AGG, go) only counts weights >0 for temproal edges, so ok to just delete missing edges
        graphs_list.append(deepcopy(g))
    end = time.time()
    tt = (end-start)/60
    print(f'time taken (min) to run loop part of build_graphs() = {tt}')
    
    start = time.time()
    AGG = nx.Graph()
    graphs_list_dc = deepcopy(graphs_list)
    for gt in graphs_list_dc:
        AGG = update_aggregate_weights(AGG, gt)
    
    end = time.time()
    tt = (end-start)/60

    print(f'time taken (min) to create aggregate network in build_graphs() = {tt}') 
    return graphs_list_dc, AGG


def update_aggregate_weights(AGG, go):
    for u, v, data in go.edges(data=True):
        # Add or update edges
        # I doubled checked - since the graph is undirected, only one ordering for each edge appears:
            # - eg, (0, 70) is an edge, but not (70, 0)
        weight = data.get('weight', 1) # leave one here, since it's jsut for agg checks
        # if weight == 0:
            # continue
        if AGG.has_edge(u, v):
            AGG[u][v]['weight'] += weight
            AGG[u][v]['temp_edge_count'] += 1
        else:
            AGG.add_edge(u, v, weight=weight)
            AGG[u][v]['temp_edge_count'] = 1

        # Ensure node attributes (node_id, geo_id) are preserved or transferred
        for node in [u, v]:
            if node in go.nodes:
                node_data = go.nodes[node]
                if 'node_id' in node_data:
                    AGG.nodes[node]['node_id'] = node_data['node_id']
                if 'geoid' in node_data:
                    AGG.nodes[node]['geoid'] = node_data['geoid']

    return AGG


# def topological_rich_club(Gt,AggG,k,delta,N,T,nodes): #original
def topological_rich_club(Gt, AggG, k, delta, N, T, nodes, args):
    # x=np.array([d[1] for d in AggG.degree()])
    # if args.randomize == "pwt" or args.weighted_degree == "False" or args.district == "DEG"::
    if args.district == "DEG":
        degrees = [(n, AggG.nodes[n]['node_id'], AggG.nodes[n]['geoid'], d) for n, d in AggG.degree()]
    else:
        degrees = [(n, AggG.nodes[n]['node_id'], AggG.nodes[n]['geoid'], d) for n, d in AggG.degree(weight='temp_edge_count')]
    degrees= sorted(degrees, key=lambda x: x[1]) # sort from least to greatest - matching node ids
    x = np.array([i[-1] for i in degrees])
    # Identify rich nodes and geoids
    rich_nodes = [node_id for node, node_id, geoid, degree in degrees if degree > k]
    rich_geoids = [geoid for node, node_id, geoid, degree in degrees if degree > k]
    set_k=set(nodes[np.where(x>k)[0]])#[0] accesses the first (and in this case, the only) array of indices from the tuple.
    vec_k=nodes[np.where(x>k)[0]]
    if len(rich_nodes) != len(vec_k):
        print('rich_nodes != vec_k', flush=True)
        print(f'rich_nodes = {rich_nodes}', flush=True)
        print(f'vec_k = {vec_k}', flush=True)
    size_Sk=len(vec_k)
    M_s=np.zeros(T-delta)
    if size_Sk>3:
        for t in range(T-delta):
            g=Gt[t]
            neighs=[[] for h in range(size_Sk)]
            for node in range(size_Sk):
                deh=set(nx.neighbors(Gt[t],vec_k[node]))
                neighs[node]=np.array(list(set_k & deh))
            for D in range(delta):
                for n in range(size_Sk):
                    doh=set(nx.neighbors(Gt[t+D],vec_k[n]))
                    neighs[n]=np.array(list(set(neighs[n]) & doh))
            
            eps=[len(x) for x in neighs]
            if args.norm == 'og':
                M_s[t]=np.sum(eps)/float(size_Sk*(size_Sk-1))
            else:
                M_s[t]=np.sum(eps)#/float(size_Sk*(size_Sk-1))

        return np.max(M_s), M_s, np.argmax(M_s), np.array(rich_geoids)
    else:
        print('len size_Sk = 0')
        return 0, M_s, 0, np.array(rich_geoids)


def weighted_rich_club(Gt, AggG, k, delta, N, T, nodes, args):
    if args.network_type == 'trc':
        np_max_M_s,M_s,time_max_t, rich_geoids = topological_rich_club(Gt, AggG, k, delta, N, T, nodes, args)
        return np_max_M_s,M_s,time_max_t, rich_geoids
    elif args.district == "DEG":
        # print('Using degree sequence for richness sequence')
        degrees = [(n, AggG.nodes[n]['node_id'], AggG.nodes[n]['geoid'], d) for n, d in AggG.degree()]
    else:
        # print('Using temporal edge counts for richness sequence')
        degrees = [(n, AggG.nodes[n]['node_id'], AggG.nodes[n]['geoid'], d) for n, d in AggG.degree(weight='temp_edge_count')]
        # Identify rich nodes and geoids
    rich_nodes = [node_id for node, node_id, geoid, degree in degrees if degree > k]
    rich_geoids = [geoid for node, node_id, geoid, degree in degrees if degree > k]
    size_Sk = len(rich_nodes)
    T = int(T)
    delta = int(delta)
    M_s = np.zeros(T - delta+1)

    # Only proceed if we have enough rich nodes
    if size_Sk > 3: # it was >= as of 3/25/24, but I changed it that day to be the same as the trc definition
        # Calculate subgraph sizes at each timestep once
        subgraph_sizes_at_t = [Gt[t].subgraph(rich_nodes).size(weight='weight') for t in range(T)]
        # Now iterate over the range (T - delta) for the sliding window
        for t in range(T - delta):
            # Calculate the subgraph weights for the sliding window of size 'delta'
            window_subgraph_sizes = subgraph_sizes_at_t[t:t+delta]
            # Compute the mean size of the subgraph over the delta period
            M_s[t] = np.mean(window_subgraph_sizes)

    # Return the maximum mean size, the array of mean sizes, the index of the maximum, and the array of rich geoids
    return np.max(M_s), M_s, np.argmax(M_s), np.array(rich_geoids)


def run_pool(var_lists): 
    # print(f'args.weighted_degree == {args.weighted_degree}')
    pool = Pool(args.pool)
    results = pool.map_async(temporal_rich_club_mod, var_lists)
    pool.close()
    pool.join()
    return results

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def build_graph_series(df, max_ind, args):
    shuffle=args.shuffle
    # tranform data in series of nx graphs
    iis=np.unique(df['i'])
    jjs=np.unique(df['j'])
    # nodes=np.union1d(iis,jjs)
    combined_unique = np.unique(np.concatenate((iis, jjs)))
    # Count the number of unique values
    nodes = len(combined_unique)
    x = max(df['t'])
    # print(f'x (np.unique(df['t'])= {x}')
    
    print(f'starting to produce graphs')
    start_i = time.time()

    graphs_list, AGG = build_graphs(df, x, nodes, iis, jjs)
    print('edge and node count of OG graph, prior to randomization:')
    en = []
    G = AGG
    edge_count = G.number_of_edges()
    node_count = G.number_of_nodes()
    en.append((edge_count,node_count))
    print(en,'\n')
    # original_graph = deepcopy([graphs_list, agg, AGG])
    # every element in this array is identical - randomize independently 
    
    
    # make sure randomized are simple and connected, for viger-latapy
    # Apply the make_graph_simple function to each graph in the list...doing here saves having to do it 20 times later
    # graphs_list_simple = [make_simple_connected(G) for G in graphs_list]

    # graphs_notRandomized = [deepcopy((graphs_list_simple,AGG,args)) for t in range(shuffle)]
    graphs_list_simple =  deepcopy(graphs_list)
    graphs_notRandomized = [deepcopy((graphs_list_simple,AGG,args)) for t in range(shuffle)]
    if args.randomize == 'False':
        print('Returning list of non-randomized graphs')
        graphs_notRandomized.insert(0, deepcopy((graphs_list, AGG)))
        return graphs_notRandomized
    
    print('edge and node counts prior to randomization:')
    print(en,'\n')
    print('starting randomization...\n')
    # if args.network_type == "topological":
    graphs_randomized = run_randomization_pool(graphs_notRandomized, args)
        
    # insert the original, unmodified graph_list back into the simplified, connected, and randomized gprahs_lists's 
    graphs_randomized.insert(0, deepcopy((graphs_list, AGG))) # add back to front of list
    end_i = time.time()
    time_taken = (end_i-start_i)/60
    print(f'total time (mins) to produce all graphs = {time_taken}')
    return graphs_randomized

def shuffle_topology_pool(var_lists, args): 
    pool = Pool(args.pool)
    results = pool.map_async(randomize_topology_and_time, var_lists)
    pool.close()
    pool.join()
    return results

def run_randomization_pool(var_lists, args):
    results = shuffle_topology_pool(var_lists, args)
    results = results.get()
    graphs_randomized = []
    for tup in results:
        graphs_list, AGG = tup
        graphs_randomized.append([graphs_list, AGG])
    return graphs_randomized


def temporal_rich_club_mod(var_list):
    deltas, ks, delta, k, graphs_list, AGG, nodes,args = var_list
    # val = temporal_rich_club(graphs_list,AGG,ks[k],deltas[delta],np.shape(graphs_list)[1],np.shape(graphs_list)[0],nodes,args)[0]
    # return (delta,k,val)
    # temporal_rich_club(Gt, AggG, k, delta, N, T, nodes, args)
    np_max_M_s,M_s,time_max_t, rich_geoids = temporal_rich_club(graphs_list,AGG,ks[k],deltas[delta],np.shape(graphs_list)[1],np.shape(graphs_list)[0],nodes,args)
    tup = (delta,k,np_max_M_s,M_s,time_max_t, rich_geoids)
    return tup





# for aiport as of 3/31/25
def compute_k_and_delta_ranges(df, graphs_array, args, weighted_degree='False'):
    print('Calculating steps based on original graph, NOT randomized graphs.')

    # Initialize variables to store aggregate degrees
    all_agg_k = []

    # # Extracting degree information from the first graph in graphs_array
    _, AGG = graphs_array[0]
    # if args.district == "DEG":
    #     print('Using degree for richness sequence')
    #     agg_k = [fr[1] for fr in AGG.degree()]
    # else:
    print('Using temporal edge count for richness sequence')
    agg_k = [fr[1] for fr in AGG.degree(weight='temp_edge_count')]
    all_agg_k.extend(agg_k)
    sorted_aggs = sorted(all_agg_k) # sorts min to max
    # kmin = sorted_ag gs[10]
    kmax = sorted_aggs[-4]-1
    kmin = 100 # 
    k_min = kmin
    k_min = 100 # just use 100 since it works for airport TEC
    k_max = kmax
    # Calculate min and max values for deltas ensuring d_min is at least 5
    d_min = max(max(min(df['t']), 1), 5)  # Ensure d_min is at least 1 and at least 5
    d_max = max(df['t'])-1  # Assuming 't' is a column in df
    # Adjust d_step to ensure a minimum delta value of 5
    d_step = max((d_max - d_min) // (args.d_step - 1), 5)  # Ensure d_step is at least 5

    
    kstep = max(int(k_max / args.k_divisor), 1)  # Ensure k_step is at least 1
    
    ks=np.arange(kmin,kmax,kstep)
    all_ks = [int(k) for k in ks]
    # Calculate all_deltas within the specified range ensuring each step is at least 5
    all_deltas = np.arange(d_min, d_max + d_step, d_step)[:args.d_step]
    all_deltas = [int(d) for d in all_deltas]

    print('overriding ds and ks for airport:')
    print(f'd_min = {d_min},\nd_max = {d_max},\nd_step = {d_step}\n')
    print(f'k_min = {k_min}, k_max = {k_max}')
    print(f'ks = {list(all_ks)}')
    print(f'all_deltas = {all_deltas}')

    # Return all_deltas, all_ks, d_step
    return all_deltas, all_ks, d_step






def run_rich_club_segments(RC_mat, var_lists_segments, RC_maxTs, RC_rich_geoids):
    for i in range(len(var_lists_segments)):
        var_lists = var_lists_segments[i]
        results = run_pool(var_lists)
        results = results.get()
        for tup in results:
            delta,k,np_max_M_s,M_s,time_max_t, rich_geoids = tup
            RC_mat[delta,k] = np_max_M_s
            RC_maxTs[delta,k] = time_max_t
            RC_rich_geoids[delta,k] = rich_geoids
            
    return RC_mat, RC_maxTs, RC_rich_geoids


def temporal_rich_club(Gt, AggG, k, delta, N, T, nodes, args):
    # degrees = [(n, AggG.nodes[n]['node_id'], AggG.nodes[n]['geoid'], d) for n, d in AggG.degree(weight='temp_edge_count')]
    np_max_M_s,M_s,time_max_t, rich_geoids = weighted_rich_club(Gt,AggG,k,delta,N,T,nodes,args) # run OUR TTRC
    return np_max_M_s,M_s,time_max_t, rich_geoids





def load_flows(args):
    print(f'Network type = {args.network_type}')
    # Now use args.data_dir and args.data_file in your script
    path = args.data_dir
    flows_df = pd.read_csv(path + args.data_file)
    
    # flows_file_combiner.py writes this column as 'weight'; older files called
    # it 'flows', so accept either.
    if 'flows' in flows_df.columns:
        flows_df.rename(columns={'flows': 'weight'}, inplace=True)
        # print(flows_df.shape)
    flows_df['weight'] = flows_df['weight'].astype(int)
    
    # print(flows_df.shape)
    # print('Dropping weight==0 rows')
    flows_df = flows_df[flows_df['weight']!=0]
    # print(flows_df.shape)
        # Step 1: Filter the DataFrame
    flows_df = flows_df[(flows_df['t'] >= args.ti) & (flows_df['t'] <= args.t)]
    # Step 2: Reset the 't' values
    flows_df['t'] = flows_df['t'] - args.ti
    # print(flows_df.shape)
    # print(f'number of unique flows_df.i = {flows_df.i.nunique()}')
    # print(f'number of unique flows_df.j = {flows_df.j.nunique()}')


    if args.testing == "True":
        print('Using testing dataset')
        flows_df = flows_df[flows_df['t'] < 20]
        flows_df = flows_df[flows_df['i'] < 20]
        flows_df = flows_df[flows_df['j'] < 20]
        # print(flows_df.shape)
    else:
        # print(f'Using {args.t-args.ti} many timesteps')
        flows_df = flows_df[flows_df['t'] < int(args.t)]
        t_count = len(set(flows_df.t))
        print(f"Actual number of unique timesteps: {t_count}") 
        # print(flows_df.shape)
    
    # Modify the 'weight' column if network type is 'topological'
    if args.network_type == 'topological' or args.network_type == 'trc':
        flows_df['weight'] = 1
        print('Since network_type == topological, weights have been set to 1')
        print(flows_df.shape)
    # Drop flows within each census unit; makes it easier to compare the input df to the temporal edge count
    flows_df = flows_df[flows_df['i'] != flows_df['j']]
    print(flows_df.shape)
    print(f'Len of flows_df = {len(flows_df)}.... = number of edges across all timesteps')
    print(f'number of unique flows_df.i = {flows_df.i.nunique()}')
    print(f'number of unique flows_df.j = {flows_df.j.nunique()}')
    # Assuming max_node_index(flows_df) is a function that you have defined elsewhere
    max_ind = max_node_index(flows_df)

    return flows_df, max_ind


def calculate_rich_club_matrices(flows_df, graphs_array, args):
    def chunker(seq, size):
        """Yield successive size chunks from seq."""
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    
    start_time = datetime.datetime.now()
    RC_matrices_list = []
    RC_maxTs_list = []
    RC_geoids_list = []
    M_s_mat_list = []

    start = time.time()
    
    if args.district == "DEG":
        print('Using degree sequence for richness sequence')
        deltas, ks, d_step = compute_k_and_delta_ranges(flows_df, graphs_array, args, weighted_degree=args)
    else:
        deltas, ks, d_step = compute_k_and_delta_ranges(flows_df, graphs_array, args, weighted_degree=args.weighted_degree)
    
    # T = np.shape(graphs_array)[0] # get number of graphs
    T = max(flows_df.t)
    longest_length = int(T) + 1  # When delta is 0 -> this is the longest case
    
    loop_count = 0
    for graphs_list, AGG in graphs_array:
        # if loop_count == 0 and int(args.district)>=10: #graphs_randomized.insert(0, deepcopy((graphs_list, AGG))) # add original network to front of list
        #     loop_count+=1
        #     print("skipping the first loop so that the WTRC for the original network isn't calculated again") 
        #     continue
        nodes = np.sort(np.array([v for v in AGG.nodes]), axis=0)
        RC_mat = np.zeros((len(deltas), len(ks)))
        geoid_mat = np.empty((len(deltas), len(ks)), dtype=object)
        RC_maxTs = np.zeros((len(deltas), len(ks)))
        # M_s_mat = np.zeros((len(deltas), len(ks), x, y))
        M_s_mat = np.full((len(deltas), len(ks), longest_length), np.nan)
        
        # Prepare data for multiprocessing, breaking it into chunks
        var_lists = [(deltas, ks, delta, k, graphs_list, AGG, nodes, args) for delta in range(len(deltas)) for k in range(len(ks))]
        chunk_size = 5 # Adjust based on memory capacity and requirements
        var_chunks = list(chunker(var_lists, chunk_size))

        # Process each chunk using multiprocessing Pool
        for chunk_index, var_chunk in enumerate(var_chunks):
            with Pool(args.pool) as pool:  # Adjust pool size as needed
                results = pool.map(temporal_rich_club_mod, var_chunk)

            # Unpack results and update matrices
            for i, result in enumerate(results):
                delta, k, np_max_M_s, M_s, time_max_t, rich_geoids = result
                RC_mat[delta, k] = np_max_M_s
                RC_maxTs[delta, k] = time_max_t
                geoid_mat[delta, k] = rich_geoids
                if M_s.size < longest_length:
                    M_s_padded = np.pad(M_s, (0, longest_length - M_s.size), 'constant', constant_values=np.nan)
                    M_s_mat[delta, k] = M_s_padded
                else:
                    M_s_mat[delta, k] = M_s

            print(f'Processed chunk {chunk_index + 1}/{len(var_chunks)} for loop_count = {loop_count}')

        RC_matrices_list.append(RC_mat)
        RC_maxTs_list.append(RC_maxTs)
        RC_geoids_list.append(geoid_mat)
        M_s_mat_list.append(M_s_mat)
        loop_count += 1
        print(f'Completed processing for graphs_list {loop_count} at {datetime.datetime.now()}')

    print(f'Total processing time: {datetime.datetime.now() - start_time}')
    end = time.time()
    tt = (end - start) / 60
    tt_div = tt / len(graphs_array)
    print(f'Time taken (min) to calculate all RC matrices: {tt}; average mins per loop = {tt_div}')

    RC_matrices = deepcopy(RC_matrices_list)
    RC_maxTs_matrices = deepcopy(RC_maxTs_list)
    RC_geoids_matrices = deepcopy(RC_geoids_list)

    # You can save them in a .npz file with np.savez
    path = output_path(args, "m_s", ext="npz")
    np.savez(path, **{'M_s_mat{}'.format(i): M_s_mat for i, M_s_mat in enumerate(M_s_mat_list)})

                          
                          
    path = output_path(args, "scan")
    RCs_array_save = np.stack(RC_matrices_list)
    print(f'saving to path = {path}')
    np.save(path, RCs_array_save)
    
    path = output_path(args, "max_t")
    RCs_array_save_maxTs = np.stack(RC_maxTs_list)
    print(f'saving to path = {path}')
    np.save(path, RCs_array_save_maxTs)

    
    path = output_path(args, "geoids")
    RC_geoids_matrices = np.stack(RC_geoids_matrices)
    print(f'saving to path = {path}')
    np.save(path, RC_geoids_matrices)
    
    # Printout end time and length
    end_time = datetime.datetime.now()
    print(f'Script ended at {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    total_time = end_time - start_time
    total_hours = total_time.seconds / 3600
    print(f'Total time taken to TRC prtion of script: {total_hours:.2f} hours')

    # return RC_matrices,RC_maxTs_matrices,RC_geoids_matrices, path, total_hours
    print('Saved RC matrices to ', path) 
    return RC_matrices_list, RC_maxTs_list, RC_geoids_list


def load_results(args):
    # Construct the file path
    # path = args.path_prefix + args.date + "_" + args.npy_file + "_" + args.network_type + '.npy'
    path = output_path(args, "scan")
    path_maxTs = output_path(args, "max_t")
    path_geoids = output_path(args, "geoids")
    # print(path)
    # Load the data
    RC_geoids_matrices = np.load(path_geoids, allow_pickle=True)
    data_new = np.load(path)
    data_new_maxTs = np.load(path_maxTs) # this will be the max Ts for the original scan and for each of the randomized scans
    # Determine the number of matrices based on the first dimension of the shape
    num_matrices_new = data_new.shape[0]

    # Extract the first matrix and the rest of the matrices
    og_matrix = data_new[0]
    RCs_array = data_new[1:]

    # Calculate the mean of the remaining matrices
    mean_matrix = np.mean(RCs_array, axis=0)

    # Divide the original matrix by the mean of the remaining matrices
    RC_norm = np.divide(og_matrix, mean_matrix)

    # return RC_norm, num_matrices_new, data_new_maxTs
    return og_matrix, RCs_array, data_new_maxTs, mean_matrix, RC_norm, RC_geoids_matrices



def plot_results(args):

    # path+='ut/240322_0_105_M_s_airports_trc_disDEG_M_s_matrices'
    path = output_path(args, "m_s", ext="npz")
    # path+= '.npz'
    # path = '/media/raid/jkruse/Temporal-Rich-Club/Human_Mobility_Flows/airport_wtrc/output/240322_0_105_M_s_airports_trc_disDEG_M_s_matrices.npz'
    data = np.load(path)
    list_of_M_s_mats = []
    # List all arrays in the .npz file
    keys = data.files
    print("Keys in the npz file:", keys)

    # Now, to look at the array stored within each of the matrices:
    for key in keys:
        M_s_mat = data[key]
        list_of_M_s_mats.append(M_s_mat)
    # Don't forget to close the file when you're done
    data.close()
    og_matrix = list_of_M_s_mats[0]

    # Assuming list_of_M_s_mats is your list of matrices with the shape [delta, k, time_step]

    # Step 1: Calculate the mean across matrices (excluding the first one), ignoring nan or 0 as padding
    # Replace 0 with nan temporarily for calculation
    averages = np.stack([np.where(mat==0, np.nan, mat) for mat in list_of_M_s_mats[1:]])
    # Calculate mean ignoring nan values
    averages = np.nanmean(averages, axis=0)

    # Step 2: Normalize the values in the first matrix by the averages and find the max value across time steps
    # Initialize a matrix for normalized values with the same shape as delta and k dimensions
    normalized_max_values = np.zeros((list_of_M_s_mats[0].shape[0], list_of_M_s_mats[0].shape[1]))

    for delta in range(list_of_M_s_mats[0].shape[0]):
        for k in range(list_of_M_s_mats[0].shape[1]):
            # Normalize values at time t by the average at time t, ignoring division by zero or nan in averages
            normalized_values = np.divide(list_of_M_s_mats[0][delta, k, :], averages[delta, k, :], 
                                          out=np.zeros_like(list_of_M_s_mats[0][delta, k, :]), 
                                          where=(averages[delta, k, :] != 0) & ~np.isnan(averages[delta, k, :]))
            # Replace 0 and nan in normalized values with -99 for masking later
            normalized_values = np.nan_to_num(normalized_values, nan=np.nan)
            # Ignore padding by considering only values not equal to -99
            valid_normalized_values = normalized_values[normalized_values != np.nan]
            # Find the maximum normalized value across all time steps for each delta and k
            if valid_normalized_values.size > 0:
                normalized_max_values[delta, k] = np.max(valid_normalized_values)
                # print(delta,k)
                # print(np.max(valid_normalized_values))
            else:
                normalized_max_values[delta, k] = np.nan

    # Mask the -99 values to not display them
    normalized_max_values_masked = np.ma.masked_invalid(normalized_max_values)

    max_values_adjusted = normalized_max_values_masked
    max_values_adjusted_masked = max_values_adjusted
    RC_norm = max_values_adjusted_masked


    return (RC_norm, list_of_M_s_mats, og_matrix, max_values_adjusted, max_values_adjusted_masked)


# to replicate original TRC workbook:


# Note: The function build_graphs needs to be defined with its logic matching your specific requirements.



