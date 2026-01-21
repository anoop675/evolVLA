import argparse
from collections import Counter
from urllib.parse import urlparse
import requests
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

def load_from_source(source):
    """
    Load data from file path or URL.
    
    Args:
        source: File path (str) or URL (str)
    
    Returns:
        bytes: Raw content (text or binary)
    
    Raises:
        requests.HTTPError: If URL request fails
        FileNotFoundError: If file does not exist
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source)
        response.raise_for_status()
        return response.content
    else:
        with open(source, "rb") as f:
            return f.read()


def build_unweighted_graph(nodes, adjacency_counts):
    """
    Build an unweighted NetworkX graph from adjacency counts.

    This function creates a graph where nodes are connected by edges if they
    appear adjacent in the text. The edge weights (how many times tokens appear
    together) are ignored - we only care about whether two tokens ever appear
    next to each other, not how often.

    Think of it like a social network: if two people have ever talked, there's
    a connection between them, regardless of how many conversations they've had.

    Important: This creates an UNDIRECTED graph, meaning if A is connected to B,
    then B is automatically connected to A. We also avoid duplicate edges - each
    unique pair gets exactly one edge.

    Parameters:
    -----------
    nodes : list
        List of node identifiers (e.g., tokens/words to include in the graph)
    adjacency_counts : Counter
        Mapping of (node1, node2) tuples to their frequency counts
        Example: {('cat', 'dog'): 3, ('dog', 'bird'): 1}

    Returns:
    --------
    networkx.Graph
        An undirected, unweighted graph with edges between adjacent nodes

    Examples:
    ---------
    >>> nodes = ['cat', 'dog', 'bird']
    >>> adjacency_counts = Counter({('cat', 'dog'): 3, ('dog', 'bird'): 1})
    >>> G = build_unweighted_graph(nodes, adjacency_counts)
    >>> list(G.edges())
    [('cat', 'dog'), ('dog', 'bird')]
    >>> G.number_of_nodes()
    3

    Algorithm Steps:
    ----------------
    1. Create empty NetworkX Graph
    2. Add all nodes to the graph
    3. Loop through adjacency_counts to find pairs
    4. For each pair, check if both nodes are valid and edge doesn't exist
    5. Add edge between valid node pairs
    6. Return the completed graph
    """

    # STEP 1: Create an empty NetworkX Graph (undirected)
    # HINT: Use nx.Graph() to create an undirected graph
    G = nx.Graph()

    # print(f"Created empty graph") # Debug helper (optional)

    # STEP 2: Add all nodes to the graph
    # HINT: Use G.add_nodes_from(nodes) to add all nodes at once
    G.add_nodes_from(nodes)

    # print(f"Added {G.number_of_nodes()} nodes to graph") # Debug helper (optional)
    
    valid_nodes = set(nodes)

    # STEP 3: Loop through adjacency_counts to process each pair
    # HINT: Use .items() to get both the pair tuple and the count
    # HINT: for (a, b), count in adjacency_counts.items():
    for (a, b), count in adjacency_counts.items():

        # print(f"  Processing pair ({a}, {b}) with count {count}") # Debug helper (optional)

        # STEP 4: Check if both nodes are valid and edge doesn't already exist
        # HINT: Check three conditions:
        #   1. a in nodes (node a is in our node list)
        #   2. b in nodes (node b is in our node list)
        #   3. not G.has_edge(a, b) (edge doesn't already exist)
        # HINT: Combine with 'and': if a in nodes and b in nodes and not G.has_edge(a, b):
        if a in valid_nodes and b in valid_nodes and not G.has_edge(a, b):

            # STEP 5: Add an edge between nodes a and b
            # HINT: Use G.add_edge(a, b)
            G.add_edge(a, b)

            # print(f"    Added edge ({a}, {b})") # Debug helper (optional)

    # print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # STEP 6: Return the completed graph
    return G

def compute_distance_matrix(nodes, adjacency_counts, distance_mode="inverted"):
    """
    Compute a distance matrix from adjacency counts between nodes.

    This function takes counts of how often nodes (tokens) appear next to each other
    and converts them into a distance matrix suitable for network analysis algorithms.

    The key insight: tokens that appear together frequently should be "close" in
    distance, while tokens that never appear together should be "far apart."

    Important Transformations:
    --------------------------
    1. SYMMETRIZATION: Adjacency counts may be directional (A->B might differ from B->A),
        but we create an undirected representation by averaging: (count(A,B) + count(B,A)) / 2

    2. SELF-CONNECTIONS: The diagonal (distance from a node to itself) is always zero

    3. DISTANCE MODES:
        - 'direct': Higher count = Higher distance (rarely used)
        - 'inverted': Higher count = Lower distance (default, more intuitive)
          Formula: distance = (max_count + 1) - count

    Think of it like a city map: if two neighborhoods share many connections (high count),
    they're close together (low distance). If they never connect, they're far apart.

    Parameters:
    -----------
    nodes : list
        Ordered list of node identifiers (defines matrix row/column order)
        Example: ['cat', 'dog', 'bird'] -> cat is index 0, dog is 1, bird is 2
    
    adjacency_counts : Counter
        Mapping of (node1, node2) -> frequency (may be directional)
        Example: {('cat', 'dog'): 5, ('dog', 'cat'): 3, ('dog', 'bird'): 2}
    
    distance_mode : str, default='inverted'
        How to convert counts to distances:
        - 'direct': distance = count (higher count = farther apart)
        - 'inverted': distance = (max_count + 1) - count (higher count = closer)

    Returns:
    --------
    tuple of (distance_matrix, count_matrix)
        distance_matrix : numpy array (n x n)
            Symmetric distance matrix with zero diagonal
        count_matrix : numpy array (n x n)
            Raw symmetrized adjacency counts (before distance conversion)

    Raises:
    -------
    ValueError
        If distance_mode is not 'direct' or 'inverted'

    Examples:
    ---------
    >>> nodes = ['A', 'B', 'C']
    >>> adjacency_counts = Counter({('A', 'B'): 4, ('B', 'A'): 2, ('B', 'C'): 1})
    >>> dist_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, 'inverted')
    >>> print(count_matrix)
    [[0.  3.  0. ]  # A-B averaged: (4+2)/2 = 3
     [3.  0.  1. ]  # B-C: 1 (no reverse)
     [0.  1.  0. ]] # Symmetric
    >>> print(dist_matrix)  # max_count=3, so (3+1)-count = 4-count
    [[0.  1.  4. ]  # A-B: 4-3=1 (close), A-C: 4-0=4 (far)
     [1.  0.  3. ]
     [4.  3.  0. ]]

    Algorithm Steps:
    ----------------
    1. Get number of nodes and create node-to-index mapping
    2. Create empty count matrix (n x n)
    3. Fill matrix with directional counts from adjacency_counts
    4. Symmetrize the matrix by averaging with its transpose
    5. Remove self-connections (set diagonal to zero)
    6. Convert counts to distances based on distance_mode
    7. Ensure diagonal is zero in distance matrix
    8. Return both distance_matrix and count_matrix
    """

    # STEP 1: Get number of nodes and create node-to-index mapping
    # HINT: n = len(nodes)
    # HINT: Create dictionary: {node: index} using enumerate
    # HINT: node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    # print(f"Processing {n} nodes: {nodes}") # Debug helper (optional)
    # print(f"Node index mapping: {node_index}")

    # STEP 2: Create empty count matrix (n x n) filled with zeros
    # HINT: Use np.zeros((n, n), dtype=float)
    count_matrix = np.zeros((n, n), dtype=float)

    # print(f"Initialized {n}x{n} count matrix")

    # STEP 3: Fill count matrix with directional counts from adjacency_counts
    # HINT: Loop through adjacency_counts.items() to get (a, b) and count
    # HINT: Check if both a and b are in node_index before adding
    # HINT: Set count_matrix[node_index[a], node_index[b]] = count
    for (a, b), count in adjacency_counts.items():
        if a in node_index and b in node_index:
            # Add the count to the appropriate matrix position
            count_matrix[node_index[a], node_index[b]] = count

    # print(f"Filled count matrix with directional counts")
    # print(f"Count matrix (before symmetrization):\n{count_matrix}")

    # STEP 4: Symmetrize the matrix for undirected representation
    # HINT: Average the matrix with its transpose: (count_matrix + count_matrix.T) / 2
    # HINT: count_matrix.T is the transpose (rows become columns)
    count_matrix = (count_matrix + count_matrix.T) / 2

    # print(f"Symmetrized count matrix:\n{count_matrix}")

    # STEP 5: Remove self-connections (set diagonal to zero)
    # HINT: Use np.fill_diagonal(count_matrix, 0)
    np.fill_diagonal(count_matrix, 0)

    # print(f"Removed self-connections (diagonal set to 0)")

    # STEP 6: Convert counts to distances based on distance_mode
    # HINT: Use if-elif-else to handle three cases: 'direct', 'inverted', and invalid

    if distance_mode == "direct":
        # Direct mode: distance equals count (higher count = higher distance)
        # HINT: distance_matrix = count_matrix.copy()
        distance_matrix = count_matrix.copy()
        # print(f"Using direct distance mode")

    elif distance_mode == "inverted":
        # Inverted mode: higher count = lower distance
        # HINT: First find max_count using np.max(count_matrix)
        max_count = np.max(count_matrix)
        # print(f"Max count in matrix: {max_count}")

        if max_count > 0:
            # HINT: distance = (max_count + 1) - count_matrix
            distance_matrix = (max_count + 1) - count_matrix
            # print(f"Using inverted distance: (max_count + 1) - count")
        else:
            # No edges exist: set all distances to 1 (except diagonal which is 0)
            # HINT: Use np.ones_like(count_matrix) to create matrix of 1s
            distance_matrix = np.ones_like(count_matrix)
            # print(f"No edges found, setting all distances to 1")

    else:
        # Invalid distance_mode
        # HINT: raise ValueError("distance_mode must be 'direct' or 'inverted'")
        raise ValueError("distance_mode must be 'direct' or 'inverted'")

    # print(f"Distance matrix (before diagonal fix):\n{distance_matrix}")

    # STEP 7: Ensure diagonal is zero in distance matrix
    # HINT: Use np.fill_diagonal(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)

    # print(f"Final distance matrix:\n{distance_matrix}")

    # STEP 8: Return both matrices as a tuple
    return distance_matrix, count_matrix

def replace_rare_tokens_with_mapping(tokens, rare_threshold=0.01, rare_token="<RARE>"):
    """
    Replace infrequent word tokens with a special rare token and return mapping info.

    Returns:
        new_tokens: list of tokens after replacements
        rare_set: set of original words considered rare
        final_counts: Counter of tokens in new_tokens
        rare_map: dict mapping index_in_original_tokens -> original_rare_word
    """
    punctuation = {',', '.', '!'}
    
    # Words considered for threshold calculation (exclude punctuation)
    word_tokens = [t for t in tokens if t not in punctuation]
    total_words = len(word_tokens)
    
    if total_words == 0:
        # Nothing to mark as rare; return copies and an empty mapping
        return tokens[:], set(), Counter(tokens[:]), {}
    
    # Count word frequencies (punctuation excluded)
    word_counts = Counter(word_tokens)
    
    # Identify rare words (based on fraction of total words)
    rare_set = {word for word, cnt in word_counts.items() if (cnt / total_words) < rare_threshold}
    
    # Build new token list and record positions of replaced tokens
    new_tokens = []
    rare_map = {}
    for idx, token in enumerate(tokens):
        if token in rare_set:
            new_tokens.append(rare_token)
            rare_map[idx] = token
        else:
            new_tokens.append(token)
    
    final_counts = Counter(new_tokens)
    return new_tokens, rare_set, final_counts, rare_map

def visualize_network(G, distance_matrix, nodes, node_colors=None, node_labels=None,
                      figsize=(14, 14), title="Network Graph", jitter=0.2, random_state=42):
    """
    Visualize network in matrix-style grid layout with edge distance labels.
    
    Nodes are arranged in a grid pattern with small random jitter to prevent
    edge overlap. Edge labels show distance values from the distance matrix.
    
    Args:
        G: NetworkX graph
        distance_matrix: numpy array of distances (n x n)
        nodes: ordered list of nodes (must match distance_matrix order)
        node_colors: list of colors for nodes (default: all 'lightblue')
        node_labels: dict of node -> label string (default: str(node)[:10])
        figsize: matplotlib figure size tuple
        title: plot title string
        jitter: max random offset for node positions (default: 0.2)
        random_state: random seed for reproducible jitter (default: 42)
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    side = int(np.ceil(np.sqrt(n)))
    
    # Grid positions with jitter to avoid edge overlap
    pos = {}
    rng = np.random.RandomState(random_state)
    for i, node in enumerate(nodes):
        row = i // side
        col = i % side
        jitter_x = rng.uniform(-jitter, jitter)
        jitter_y = rng.uniform(-jitter, jitter)
        pos[node] = (col + jitter_x, -row + jitter_y)
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title)
    
    # Default node properties
    if node_colors is None:
        node_colors = ["lightblue"] * n
    if node_labels is None:
        node_labels = {node: str(node)[:10] for node in nodes}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors,
                           edgecolors="black", linewidths=1, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)
    
    # Edge labels (distances)
    edge_labels = {}
    for u, v in G.edges():
        if u in node_index and v in node_index:
            dist_val = distance_matrix[node_index[u], node_index[v]]
            edge_labels[(u, v)] = f"{dist_val:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax, label_pos=0.5)
    
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# TEXT PROCESSING
def tokenize_text(text, char_like=["'"]): 
    """
    Tokenize text into individual words and punctuation marks.

    This function breaks down a text string into meaningful pieces (tokens).
    It separates words from punctuation and handles various text characters.
    Think of it like breaking a sentence into individual puzzle pieces.

    Rules:
    ------
    - Words: sequences of alphabetic characters (a-z, A-Z, Unicode letters)
    - Punctuation: comma (,), period (.), and exclamation (!) are kept as separate tokens
    - Characters in `char_like` (e.g., apostrophes) are treated as part of words
    - Other characters: spaces, numbers, symbols act as word separators (ignored)
    - Case: everything is converted to lowercase

    Parameters:
    -----------
    text : str
        Input text to tokenize (e.g., "Hello, world! How are you?")
    char_like : list
        Optional characters that should be treated as if they were alphabet letters

    Returns:
    --------
    list of str
        List of tokens (words and punctuation marks)

    Examples:
    ---------
    >>> tokenize_text("Hello, world!")
    ['hello', ',', 'world']

    >>> tokenize_text("I like cats, dogs, and birds.")
    ['i', 'like', 'cats', ',', 'dogs', ',', 'and', 'birds', '.']

    >>> tokenize_text("Numbers123 and symbols@#$ are ignored")
    ['numbers', 'and', 'symbols', 'are', 'ignored']

    Algorithm Overview:
    ------------------
    1. Convert text to lowercase
    2. Scan character by character
    3. Build words by collecting alphabetic characters
    4. When we hit punctuation or separators, finish the current word
    5. Keep commas, periods, and exclamation marks as tokens, ignore other separators
    """

    # STEP 1: Prepare the text and initialize variables
    text = text.lower()
    tokens = []
    current_word = []
    punctuation = {',', '.', '!'}

    # print(f"Processing text: '{text[:50]}...'") # Debug helper (optional)

    # STEP 2: Process each character in the text
    for char in text:

        # STEP 3: Handle alphabetic characters (letters) or 'char_like' characters
        # HINT: Use char.isalpha() to check if character is a letter OR char in char_like
        # HINT: Add the character to current_word list using append()
        # HINT: Use ''.join(current_word) to combine characters into a string
        if char.isalpha() or char in char_like:
            current_word.append(char)

        # STEP 4: Handle punctuation (comma, period, exclamation mark)
        # HINT: if char in punctuation:
        # HINT: Flush current_word, then add punctuation as token
        elif char in punctuation:
            if current_word:
                tokens.append(''.join(current_word)) # Corrected: used .join()
                current_word = []
            tokens.append(char) # Punctuation mark itself is a token
                
        # STEP 5: Handle separators (spaces, numbers, symbols)
        # HINT: Flush current_word if needed, ignore the separator
        else:
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []

    # STEP 6: Handle final word if exists
    # HINT: After the loop, if current_word has content, join and add it
    if current_word:
        tokens.append(''.join(current_word))

    # STEP 7: Return the list of tokens
    return tokens

def replace_rare_tokens(tokens, rare_threshold=0.01, rare_token="<RARE>"):
    """
    Replace infrequent words with a special rare token to reduce vocabulary size.

    This function helps clean up text data by replacing words that appear very rarely
    with a single special token. This is useful in text analysis because rare words
    often don't provide much meaningful information but can make datasets harder to work with.

    Important: Only words are considered for replacement, punctuation (like , and .)
    is always kept as-is regardless of frequency.

    Parameters:
    -----------
    tokens : list of str
        List of tokens from tokenized text (e.g., ['hello', ',', 'world', '.'])
    rare_threshold : float, default=0.01
        Fractional threshold between 0 and 1.
        Words appearing less than this fraction of total words will be replaced.
    rare_token : str, default="<RARE>"
        The special token to use as replacement for rare words.

    Returns:
    --------
    tuple of (new_tokens, rare_set, final_counts)
        new_tokens : list of str
            Original token list with rare words replaced
        rare_set : set of str
            Set of original words that were considered rare and replaced
        final_counts : Counter
            Frequency count of tokens in the final new_tokens list

    Example:
    --------
    >>> tokens = ['cat', 'dog', 'cat', 'elephant', 'dog', 'cat', '.']
    >>> new_tokens, rare_set, final_counts = replace_rare_tokens(tokens, rare_threshold=0.25)
    >>> print(f"Rare words: {rare_set}")
    {'elephant'}

    Algorithm Steps:
    ----------------
    1. Filter out punctuation to count only actual words
    2. Calculate what proportion each word represents
    3. Identify words below the threshold as "rare"
    4. Replace rare words in original token list
    5. Count final token frequencies
    """
    
    # We include '!' here, matching tokenize_text's punctuation set
    punctuation = {',', '.', '!'} 

    # STEP 1: Extract only words (non-punctuation) for threshold calculation
    # HINT: Use list comprehension to keep only tokens not in punctuation
    word_tokens = [token for token in tokens if token not in punctuation]  

    # print(f"Word tokens only: {word_tokens}")

    # STEP 2: Calculate total word count and handle edge case
    total_words = len(word_tokens)
    # print(f"Total words: {total_words}")

    if total_words == 0:
        # print("No words found - returning original tokens unchanged")
        return tokens[:], set(), Counter(tokens)

    # STEP 3: Count frequency of each word
    # HINT: Use Counter(word_tokens)
    word_counts = Counter(word_tokens)  
    # print(f"Word counts: {dict(word_counts)}")

    # STEP 4: Identify rare words based on threshold
    # HINT: Use set comprehension:
    # {word for word, count in word_counts.items() if count / total_words < rare_threshold}
    # Note: Punctuation is never considered rare, as it's not in word_counts
    rare_set = {word for word, count in word_counts.items() if count / total_words < rare_threshold}
    
    new_tokens = []
    rare_map = {}
    for i, token in enumerate(tokens):
        if token in rare_set:
            new_tokens.append(rare_token)
            rare_map[i] = token  # store original rare token at this position
        else:
            new_tokens.append(token)        
    # print(f"Rare words identified: {rare_set}")

    # STEP 5: Create new token list with replacements
    # HINT: Use list comprehension:
    # [rare_token if token in rare_set else token for token in tokens]
    #new_tokens = [rare_token if token in rare_set else token for token in tokens]
    # print(f"Tokens after replacement: {new_tokens}")

    # STEP 6: Count frequencies in the final token list
    final_counts = Counter(new_tokens)
    # print(f"Final token counts: {dict(final_counts)}")

    # STEP 7: Return all three results as a tuple
    return new_tokens, rare_set, final_counts

def get_text_adjacencies(tokens):
    """
    Count how often each pair of consecutive tokens appears together in the text.

    This function creates a "bigram" analysis - it looks at every pair of adjacent
    tokens and counts how many times each pair occurs. This helps us understand
    which words tend to follow other words in the text.

    Think of it like analyzing conversation patterns: if someone says "good",
    how often is the next word "morning" vs "afternoon" vs "luck"?

    Parameters:
    -----------
    tokens : list of str
        List of tokens in order (e.g., ['the', 'cat', 'sat', 'on', 'the', 'mat'])

    Returns:
    --------
    Counter
        Dictionary-like object mapping (token1, token2) tuples to their frequency count
        Key: (current_token, next_token)
        Value: number of times this pair appears consecutively

    Examples:
    ---------
    >>> tokens = ['the', 'cat', 'sat', 'on', 'the', 'cat']
    >>> adjacencies = get_text_adjacencies(tokens)
    >>> print(dict(adjacencies))
    {('the', 'cat'): 2, ('cat', 'sat'): 1, ('sat', 'on'): 1, ('on', 'the'): 1}

    Explanation:
    - 'the' is followed by 'cat' twice: positions (0,1) and (4,5)
    - 'cat' is followed by 'sat' once: position (1,2)
    - 'sat' is followed by 'on' once: position (2,3)
    - 'on' is followed by 'the' once: position (3,4)

    Visual Example:
    ---------------
    tokens = ['I', 'love', 'cats', '.', 'I', 'love', 'dogs']
    pairs:      ^       ^      ^     ^     ^       ^
                (0,1)  (1,2)  (2,3) (3,4) (4,5) (5,6)

    Results: ('I', 'love'): 2, ('love', 'cats'): 1, ('cats', '.'): 1,
             ('.', 'I'): 1, ('love', 'dogs'): 1

    Algorithm Steps:
    ----------------
    1. Initialize Counter to store adjacency counts
    2. Loop through token positions 0 to len(tokens)-2
    3. For each position i, get tokens[i] and tokens[i+1]
    4. Only count pairs where the tokens are different (no self-loops)
    5. Create tuple pair (current_token, next_token) and increment count
    6. Return the Counter with all pair frequencies
    """

    # STEP 1: Initialize Counter to store adjacency counts
    adjacency_counts = Counter()
    num_tokens = len(tokens)

    # print(f"Analyzing adjacencies for {num_tokens} tokens") # Debug helper (optional)

    # STEP 2: Loop through all consecutive pairs of tokens
    # HINT: We need to examine pairs at positions (0,1), (1,2), ..., (n-2, n-1)
    # HINT: This means i goes from 0 to len(tokens)-2, so use range(len(tokens) - 1)
    for i in range(num_tokens - 1):

        # STEP 3: Get the current pair of tokens
        # HINT: Current token is at index i, next token is at index i+1
        current_token = tokens[i]
        next_token = tokens[i+1]

        # print(f"  Pair {i}: '{current_token}' -> '{next_token}'") # Debug helper (optional)

        # STEP 4: Only count pairs where tokens are different (exclude self-loops)
        # HINT: Check if current_token != next_token
        if current_token != next_token:

            # STEP 5: Create the pair tuple and increment its count
            # HINT: Use tuple (current_token, next_token) as the key
            # HINT: Increment count with adjacency_counts[(current_token, next_token)] += 1
            adjacency_counts[(current_token, next_token)] += 1

    # STEP 6: Return the Counter with all pair frequencies
    return adjacency_counts

def process_text_network(source, rare_threshold=0.01, rare_token="<RARE>",
                         distance_mode="inverted", verbose=True, nsample_tokens=20):
    """
    Complete text network processing pipeline.
    
    Pipeline:
    1. Load text from source (file or URL)
    2. Tokenize into words and punctuation
    3. Replace rare tokens
    4. Build directional adjacency counts from consecutive tokens
    5. Create unweighted graph and distance matrix
    
    Args:
        source: file path or URL
        rare_threshold: token rarity threshold (0-1)
        rare_token: replacement string for rare tokens
        distance_mode: 'direct' or 'inverted' (see compute_distance_matrix)
        verbose: if True, print processing details
        nsample_tokens: number of sample tokens to print (if verbose)
    
    Returns:
        dict with keys:
            - graph: NetworkX Graph
            - nodes: list of node identifiers (sorted by frequency)
            - adjacency_counts: Counter of (node1, node2) -> count
            - distance_matrix: numpy array (n x n)
            - count_matrix: symmetrized adjacency counts
            - token_counts: Counter of final token frequencies
            - rare_tokens: set of replaced tokens
            - original_tokens: list of tokens before rare replacement
    """
    # Load and tokenize
    content = load_from_source(source)
    text = content.decode('utf-8', errors='ignore')
    
    if verbose:
        print(f"Loaded text: {len(text)} characters")
    
    tokens = tokenize_text(text)
    
    if verbose:
        print(f"Tokenized: {len(tokens)} tokens")
        print(f"Sample tokens: {list(set(tokens))[:nsample_tokens]}")
    
    # Handle rare tokens
    #processed_tokens, rare_set, token_counts, rare_map = replace_rare_tokens_with_mapping(
    #    tokens, rare_threshold, rare_token
    #)

    processed_tokens, rare_set, token_counts = replace_rare_tokens(tokens, rare_threshold)
    
    if verbose:
        print(f"Replaced {len(rare_set)} rare tokens (threshold={rare_threshold})")
        print(f"Final vocabulary: {len(token_counts)} unique tokens")
        print(f"Sample tokens: {list(set(processed_tokens))[:nsample_tokens]}")
    
    # Build adjacencies and graph
    adjacency_counts = get_text_adjacencies(processed_tokens)
    nodes = sorted(token_counts.keys(), key=lambda x: (-token_counts[x], x))
    graph = build_unweighted_graph(nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Top tokens by frequency:")
        for i, node in enumerate(nodes[:10]):
            print(f"  {i+1:2d}. '{node}' (freq={token_counts[node]})")
    
    return {
        'graph': graph,
        'nodes': nodes,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'token_counts': token_counts,
        'rare_tokens': rare_set,
        #'rare_map': rare_map,
        'processed_tokens': processed_tokens,
        'original_tokens': tokens
    }


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def preprocess_image(image_data, target_size=(128, 128), quantize_levels=16):
    """
    Load and preprocess image: resize and quantize colors.
    
    Quantization maps each pixel channel value to the nearest of N evenly-spaced
    levels. For uniform channels (min == max), all pixels map to level 0.
    
    Performance note: Uses vectorized numpy operations for efficient quantization.
    
    Args:
        image_data: raw image bytes
        target_size: resize dimensions (width, height)
        quantize_levels: number of quantization levels per channel
    
    Returns:
        tuple of:
            - quantized_image: numpy array of quantized indices (height x width x channels)
            - quantization_info: dict mapping channel_idx -> {'min', 'max', 'levels'}
                where 'levels' is the array of quantization boundary values
    """
    img = Image.open(BytesIO(image_data))
    
    # Handle different image modes
    if img.mode == 'RGBA':
        # Composite with white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Add channel dimension for grayscale
    if len(img_array.shape) == 2:
        img_array = img_array[:, :, np.newaxis]
    
    height, width, channels = img_array.shape
    quantized = np.zeros_like(img_array)
    quantization_info = {}

    for c in range(channels):
        channel_data = img_array[:, :, c].astype(np.float64)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        if max_val == min_val:
            # Uniform channel: all pixels map to level 0
            levels = np.array([min_val])
            quantized[:, :, c] = 0
        else:
            # Define bins/levels for quantization
            # Levels are the midpoints, we need N+1 boundaries
            boundaries = np.linspace(min_val, max_val, quantize_levels + 1)
            
            # Use np.digitize to find which bin each value falls into
            # The result is the index of the boundary *after* the value
            # Subtract 1 to get the index of the bin (0 to N-1)
            quantized_indices = np.digitize(channel_data, boundaries[1:], right=True)
            quantized[:, :, c] = quantized_indices
            
            # Levels are the bin indices 0 to N-1 for this channel
            levels = np.arange(quantize_levels)
            
        quantization_info[c] = {
            'min': min_val, 
            'max': max_val, 
            'levels': levels
        }
    
    # Return as integer array for indices
    return quantized.astype(int), quantization_info

def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUnifiedNetworks)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Only run tests when script is executed directly
    success = run_tests()
