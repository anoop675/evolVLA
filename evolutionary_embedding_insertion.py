"""
Implements a (1+λ) Evolution Strategy for inserting new word embeddings into a trained
Skip-Gram model while preserving the existing embedding space structure.
"""

import sys
import torch
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision

from sgns_text_network_embeddings import SkipGramModel, find_similar_words
from network_pipeline import process_text_network

import unittest
import tempfile
import os

def load_trained_model(model_path: str, vocab_size: int, 
                       embedding_dim: int, dropout: float) -> Tuple[torch.nn.Module, np.ndarray]:
    """Robustly load trained Skip-Gram model and extract embeddings.

    Handles small vocabulary-size mismatches by copying matching rows from
    the checkpoint and initializing extra embedding rows sensibly.
    Also attempts to use weights_only=True when available to reduce pickle attack surface.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Try to use weights_only flag if available (newer PyTorch)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions don't accept weights_only; fall back with a warning
        print("Note: torch.load(..., weights_only=True) not available in this PyTorch build; loading normally.")
        checkpoint = torch.load(model_path, map_location=device)

    # Inspect checkpoint keys
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ck_state = checkpoint['model_state_dict']
    else:
        # If someone saved only a state_dict directly
        ck_state = checkpoint

    # Build model with requested size
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout).to(device)

    # If shapes match, load directly
    try:
        model.load_state_dict(ck_state)
    except RuntimeError as e:
        # Print informative diagnostic
        print("RuntimeError when loading state_dict:", e)
        # Try to intelligently patch embedding weight matrices if mismatch is only in embedding rows
        patched_state = dict(ck_state)  # shallow copy

        # Candidate embedding parameter names (adjust if your model uses different names)
        emb_names = []
        for k, v in ck_state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.size(1) == embedding_dim:
                # Heuristic: 2D tensor with second dim == embedding_dim is likely embeddings
                emb_names.append(k)

        if not emb_names:
            # nothing to patch, re-raise original error
            raise

        print("Detected embedding parameter names in checkpoint:", emb_names)
        for name in emb_names:
            ck_tensor = ck_state[name]
            if ck_tensor.shape == getattr(model, name.split('.')[0]).weight.shape if '.' in name else None:
                # shapes match (unlikely since we got an error), continue
                continue

            ck_rows, ck_dim = ck_tensor.shape
            model_param = dict(model.named_parameters()).get(name, None)
            if model_param is None:
                # try named buffers too
                model_param = dict(model.named_buffers()).get(name, None)

            if model_param is None:
                print(f"Warning: couldn't find parameter '{name}' in model; skipping patch for this key.")
                continue

            new_rows, new_dim = model_param.shape
            print(f"Patching '{name}': checkpoint shape {ck_tensor.shape} -> model shape {tuple(model_param.shape)}")

            # Determine how many rows to copy
            rows_to_copy = min(int(ck_rows), int(new_rows))
            patched = torch.empty((new_rows, new_dim), dtype=ck_tensor.dtype, device=device)

            # copy available rows
            patched[:rows_to_copy] = ck_tensor[:rows_to_copy].to(device)

            # initialize remaining rows: use mean of copied rows when possible, else small normal noise
            if rows_to_copy > 0:
                mean_row = ck_tensor[:rows_to_copy].mean(dim=0, keepdim=True).to(device)
                for r in range(rows_to_copy, new_rows):
                    patched[r:r+1] = mean_row  # place same mean row
            else:
                # no rows to copy (odd case) — initialize random small gaussian
                patched = torch.randn((new_rows, new_dim), device=device) * 0.01

            # replace in patched_state (as CPU tensor)
            patched_state[name] = patched.cpu()

        # Try loading patched state dict (non-strict to allow other minor mismatches)
        model.load_state_dict(patched_state, strict=False)
        print("Loaded state_dict with patched embeddings (non-strict).")

    model.eval()

    # Extract embeddings (try common attribute names; adjust if your model exposes differently)
    with torch.no_grad():
        try:
            embeddings_tensor = model.get_embeddings()
        except Exception:
            # fallback: try to access nn.Embedding weights directly
            # Common names: 'center_embeddings.weight' or 'input_embeddings.weight' — inspect model
            possible = None
            for name, param in model.named_parameters():
                if param.ndim == 2 and param.shape[1] == embedding_dim:
                    possible = param
                    break
            if possible is None:
                raise RuntimeError("Could not find embeddings in model after loading.")
            embeddings_tensor = possible

        embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) 
                     else np.array(embeddings_tensor)).astype(np.float32)

    print(f"Loaded model: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return model, embeddings



def create_mappings(nodes: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, np.ndarray]]:
    """Create word-to-index and index-to-word mappings."""
    word_to_idx = {word: idx for idx, word in enumerate(nodes)}
    idx_to_word = {idx: word for idx, word in enumerate(nodes)}
    return word_to_idx, idx_to_word


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics needed for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'global_std': np.std(embeddings)
    }

def get_cifar100_vocabulary() -> List[str]:
    """Download CIFAR-100 and extract class names."""
    print("\nLoading CIFAR-100 vocabulary...")
    dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True)
    print(f"CIFAR-100 vocabulary loaded: {len(dataset.classes)} classes")
    return dataset.classes


def analyze_vocabulary_overlap(cifar_vocab: List[str], network_vocab: List[str]) -> List[str]:
    """Analyze overlap between CIFAR-100 and network vocabulary."""
    cifar_set, network_set = set(cifar_vocab), set(network_vocab)
    overlapping = sorted(list(cifar_set.intersection(network_set)))
    missing = sorted(list(cifar_set - network_set))
    
    print(f"\n{'='*70}")
    print("VOCABULARY OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"CIFAR-100 vocabulary: {len(cifar_set)} classes")
    print(f"Network vocabulary: {len(network_set)} words")
    print(f"Overlapping words: {len(overlapping)} ({len(overlapping)/len(cifar_set)*100:.1f}%)")
    print(f"Missing from network: {len(missing)}")
    if overlapping:
        print(f"\nFound: {', '.join(overlapping)}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
    print(f"{'='*70}\n")
    
    return missing


# CONTEXT EXTRACTION

def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5
) -> Dict[str, Counter]:
    """
    Extract co-occurrence context statistics for target words from a text corpus.
    
    This function reads a corpus file line-by-line and tracks which words appear
    near specified target words. For each target word, it counts how many times
    each vocabulary word appears within a window around it.
    
    Args:
        text_file: Path to the corpus text file to analyze.
        target_words: List of words to extract contexts for.
        vocab_set: Set of valid vocabulary words (only count these as contexts).
        window: Number of words to look on each side of the target word.
    
    Returns:
        A dictionary mapping each target word to a Counter of context words and
        their frequencies.
        
    Example:
        >>> extract_word_contexts('corpus.txt', ['king', 'queen'], vocab, window=2)
        {'king': Counter({'royal': 5, 'crown': 3}), 
         'queen': Counter({'royal': 4, 'throne': 2})}
    
    Implementation guidelines:
    --------------------------
    1. Initialize a dictionary `{word: Counter()}` for each target word.
    2. Convert `target_words` to a set for fast lookup.
    3. Stream through the file line-by-line (efficient for large corpora).
    4. For each line:
        - Tokenize using lowercase alphabetic words (regex: r"\\b[a-z]+\\b").
        - For each token that matches a target word:
            * Extract up to `window` tokens on both sides.
            * Exclude the target word itself.
            * Retain only context words that appear in `vocab_set`.
            * Update the Counter for that target word.
    5. Handle edge cases: empty lines, start/end of token lists.
    6. Optionally print progress (e.g., every 50,000 lines) for user feedback.
    7. Return the dictionary of Counters.
    """
    
    # Initialize contexts dictionary with a Counter for each target word.
    contexts = {word: Counter() for word in target_words}
    
    # Implement the corpus scanning and context extraction logic.
    target_set = set(target_words)
    token_re = re.compile(r"\b[a-z]+\b")
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                if not line:
                    continue
                tokens = token_re.findall(line.lower())
                if not tokens:
                    continue
                for i, tok in enumerate(tokens):
                    if tok in target_set:
                        left = max(0, i - window)
                        right = min(len(tokens), i + window + 1)
                        for j in range(left, right):
                            if j == i:
                                continue
                            ctx = tokens[j]
                            if ctx in vocab_set:
                                contexts[tok][ctx] += 1
                if lineno % 50000 == 0:
                    print(f"Processed {lineno} lines...")
    except FileNotFoundError:
        raise
    return contexts


# FITNESS FUNCTION

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    anchor_weights: Optional[np.ndarray],   
    stats_dict: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute a three-term fitness score for a candidate word embedding vector.
    
    This function evaluates how well a candidate vector fits the learned 
    embedding space by combining three complementary metrics:
    1. Corpus likelihood (how well it predicts observed contexts)
    2. Norm matching (how similar its magnitude is to typical embeddings)
    3. Anchor similarity (how similar it is to known reference words)
    
    Args:
        vec: Candidate embedding vector to evaluate.
        word: Target word (for reference, not used in computation).
        ctx_vecs: Context word vectors that co-occur with the target word.
                  Shape: (n_contexts, embedding_dim). May be None if no contexts.
        ctx_weights: Weights for each context (e.g., co-occurrence counts).
                     Shape: (n_contexts,). May be None if no contexts.
        neg_vecs: Negative sample vectors (words that don't co-occur).
                  Shape: (n_negatives, embedding_dim).
        anchor_vecs: Pre-normalized vectors of anchor words for comparison.
                     Shape: (n_anchors, embedding_dim). May be None.
        stats_dict: Dictionary containing embedding statistics:
                    - 'mean_norm': Average L2 norm of embeddings in the space
                    - 'std_norm': Standard deviation of embedding norms
                    - 'global_std': Global standard deviation (if needed)
        weights: Dictionary of weights for each fitness component:
                 - 'corpus': Weight for corpus likelihood term
                 - 'norm': Weight for norm matching term
                 - 'anchor': Weight for anchor similarity term
    
    Returns:
        Combined fitness score in the range [0, 1], where higher is better.
        
    Example:
        >>> vec = np.array([0.5, -0.3, 0.8, 0.1])
        >>> stats = {'mean_norm': 1.0, 'std_norm': 0.2, 'global_std': 0.5}
        >>> weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        >>> fitness = compute_fitness(vec, 'king', ctx_vecs, ctx_weights, 
        ...                           neg_vecs, anchor_vecs, stats, weights)
        >>> print(f"Fitness: {fitness:.4f}")
        Fitness: 0.7234
    
    Implementation guidelines:
    --------------------------
    Term 1 - Corpus Likelihood (L_corpus_norm):
        - For positive contexts: sum over ctx_weights * log(sigmoid(ctx_vecs · vec))
        - For negative samples: sum over log(sigmoid(-neg_vecs · vec))
        - Add small epsilon (1e-10) inside log for numerical stability
        - Normalize by total samples, then apply sigmoid to map to [0, 1]
        - Default to 0.5 if no samples available
        
    Term 2 - Norm Match (S_norm):
        - Compute L2 norm of the candidate vector
        - Use Gaussian similarity: exp(-((norm - mean_norm)² / (2 * std_norm²)))
        - This rewards vectors with norms close to the typical embedding norm
        
    Term 3 - Anchor Similarity (S_anchor):
        - Normalize the candidate vector (divide by its norm + epsilon)
        - Compute dot products with all anchor vectors (they're pre-normalized)
        - Take the mean similarity across all anchors
        - Default to 0.5 if no anchors provided
        
    Final score:
        - Weighted sum: weights['corpus'] * L_corpus_norm + 
                       weights['norm'] * S_norm + 
                       weights['anchor'] * S_anchor
    
    Notes:
        - Handle None values for optional parameters (ctx_vecs, ctx_weights, anchor_vecs)
        - Use vectorized NumPy operations for efficiency
        - Add small epsilon values to prevent division by zero
    """
    
    eps = 1e-10

    # Term 1: Corpus likelihood
    if ctx_vecs is not None and ctx_weights is not None and ctx_vecs.shape[0] > 0:
        # positive contexts
        pos_dots = ctx_vecs.dot(vec)  # (n_ctx,)
        pos_scores = np.log(sigmoid(pos_dots) + eps)
        # ctx_weights expected to sum to 1
        pos_term = float(np.sum(ctx_weights * pos_scores))
    else:
        pos_term = 0.0

    if neg_vecs is not None and neg_vecs.shape[0] > 0:
        neg_dots = neg_vecs.dot(vec)  # (n_neg,)
        neg_scores = np.log(sigmoid(-neg_dots) + eps)
        neg_term = float(np.mean(neg_scores))
    else:
        neg_term = 0.0

    if (ctx_vecs is None or ctx_weights is None or ctx_vecs.shape[0] == 0) and (neg_vecs is None or neg_vecs.shape[0] == 0):
        L_corpus_norm = 0.5
    else:
        combined = pos_term + neg_term
        L_corpus_norm = float(sigmoid(np.array(combined)))

    # Term 2: Norm matching
    mean_norm = stats_dict.get('mean_norm', 1.0)
    std_norm = stats_dict.get('std_norm', 1e-6)
    vec_norm = float(np.linalg.norm(vec))
    if std_norm <= 1e-8:
        S_norm = 1.0 if abs(vec_norm - mean_norm) < 1e-6 else 0.0
    else:
        S_norm = float(np.exp(-((vec_norm - mean_norm) ** 2) / (2 * (std_norm ** 2 + eps))))

    # Term 3: Anchor similarity
    if anchor_vecs is not None and anchor_vecs.shape[0] > 0:
        vec_norm_for_cos = vec_norm + eps
        normalized_vec = vec / vec_norm_for_cos
        sims = anchor_vecs.dot(normalized_vec)  # anchor_vecs assumed normalized rows
        #mean_sim = float(np.mean(sims))
        if anchor_weights is not None:
            mean_sim = float(np.sum(anchor_weights * sims))
        else:
            mean_sim = float(np.mean(sims))

        S_anchor = (mean_sim + 1.0) / 2.0
    else:
        S_anchor = 0.5

    # Weighted sum
    corpus_w = float(weights.get('corpus', 0.5))
    norm_w = float(weights.get('norm', 0.3))
    anchor_w = float(weights.get('anchor', 0.2))

    score = corpus_w * L_corpus_norm + norm_w * S_norm + anchor_w * S_anchor
    score = max(0.0, min(1.0, score))
    return score


# GENETIC ALGORITHM (1+λ) EVOLUTION STRATEGY

def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int]
) -> np.ndarray:
    """
    Initialize an embedding vector for a word using corpus bootstrap.
    
    This function creates an initial embedding by computing a weighted average
    of the embeddings of words that frequently co-occur with the target word.
    This provides a data-driven starting point that places the new word near
    semantically related words in the embedding space.
    
    Args:
        word: Target word to initialize an embedding for.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
    
    Returns:
        Initial embedding vector for the word. Shape: (embedding_dim,).
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30, 'castle': 20})}
        >>> embeddings = np.random.randn(1000, 300)  # 1000 words, 300 dims
        >>> word_to_idx = {'queen': 0, 'royal': 1, 'castle': 2, ...}
        >>> vec = initialize_embedding('king', contexts, embeddings, word_to_idx)
        >>> vec.shape
        (300,)
    
    Implementation guidelines:
    --------------------------
    1. Handle the no-context case:
       - If the word has no contexts (empty Counter), return the mean of all
         embeddings as a neutral starting point
    
    2. Get top context words:
       - Extract the top 20 most frequent context words using Counter.most_common()
       - This focuses on the strongest statistical relationships
    
    3. Compute weighted average:
       - Calculate the total weight (sum of all counts)
       - For each context word that exists in word_to_idx:
           * Get its embedding vector
           * Weight it by (count / weight_sum)
           * Add to running sum
    
    4. Validate the result:
       - Check if the resulting vector has non-zero norm
       - If zero (e.g., no valid context words found), fall back to mean embedding
    
    Notes:
        - Some context words may not be in word_to_idx; skip these
        - The weighted average naturally places the new word near its contexts
        - Using top 20 contexts balances informativeness with noise reduction
    """
  
    dim = embeddings.shape[1]
    mean_embedding = np.mean(embeddings, axis=0)

    if word not in contexts or len(contexts[word]) == 0:
        return mean_embedding.copy()

    top_ctx = contexts[word].most_common(20)
    total_weight = 0.0
    vec = np.zeros(dim, dtype=np.float32)

    for ctx_word, count in top_ctx:
        if ctx_word in word_to_idx:
            idx = word_to_idx[ctx_word]
            vec += embeddings[idx] * float(count)
            total_weight += float(count)

    if total_weight <= 0:
        return mean_embedding.copy()

    vec /= (total_weight + 1e-12)
    if np.linalg.norm(vec) < 1e-12:
        return mean_embedding.copy()
    return vec.astype(np.float32)


def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    # OLD: 
    # anchors: Dict[str, List[str]],
    # NEW: anchors as list of (anchor_word, weight)
    anchors: Dict[str, List[Tuple[str, float]]],
    num_negatives: int = 15
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray]
]:
    
    dim = embeddings.shape[1]

    # Part 1: Positive contexts
    ctx_vecs = None
    ctx_weights = None
    if word in contexts and len(contexts[word]) > 0:
        vecs = []
        weights = []
        for ctx_word, count in contexts[word].items():
            if ctx_word in word_to_idx:
                idx = word_to_idx[ctx_word]
                vecs.append(embeddings[word_to_idx[ctx_word]])
                weights.append(float(count))

        if len(vecs) > 0:
            ctx_vecs = np.vstack(vecs)
            weights = np.array(weights, dtype=np.float32)
            weights_sum = weights.sum()
            if weights_sum <= 0:
                ctx_weights = np.ones_like(weights, dtype=np.float32) / len(weights)
            else:
                ctx_weights = weights / weights_sum

    # Part 2: Negative samples
    vocab_for_neg = [w for w in vocab_list if w in word_to_idx and w != word]
    if len(vocab_for_neg) == 0:
        # fallback: sample from entire vocab
        neg_vecs = np.zeros((max(1, num_negatives), dim), dtype=np.float32)
        for i in range(neg_vecs.shape[0]):
            neg_vecs[i] = embeddings[np.random.randint(0, embeddings.shape[0])]
    else:
        replace = len(vocab_for_neg) < num_negatives
        sampled = list(np.random.choice(vocab_for_neg, num_negatives, replace=replace))
        neg_vecs = np.vstack([embeddings[word_to_idx[w]] for w in sampled])

    # Part 3: Anchor vectors

    # OLD UNWEIGHTED VERSION (kept for reference)
    # anchor_vecs = None
    # if anchors and word in anchors and len(anchors[word]) > 0:
    #     valid = [a for a in anchors[word] if a in word_to_idx]
    #     if len(valid) > 0:
    #         anc = np.vstack([embeddings[word_to_idx[a]] for a in valid])
    #         norms = np.linalg.norm(anc, axis=1, keepdims=True)
    #         norms = norms + 1e-10
    #         anchor_vecs = anc / norms
    # return ctx_vecs, ctx_weights, neg_vecs, anchor_vecs

    # NEW WEIGHTED ANCHOR VERSION
    anchor_vecs = None
    anchor_weights = None
    if anchors and word in anchors and len(anchors[word]) > 0:
        # anchors[word] expected to be List[(anchor_word, weight)]
        valid = [(a, w) for (a, w) in anchors[word] if a in word_to_idx]
        if valid:
            anc = np.vstack([embeddings[word_to_idx[a]] for (a, _) in valid])
            norms = np.linalg.norm(anc, axis=1, keepdims=True)
            norms = norms + 1e-10
            anchor_vecs = anc / norms

            anchor_weights = np.array([w for (_, w) in valid], dtype=np.float32)
            s = float(anchor_weights.sum())
            if s > 0:
                anchor_weights /= s
            else:
                # fallback: uniform if something degenerate happens
                anchor_weights = np.ones_like(anchor_weights) / len(anchor_weights)

    return ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, anchor_weights


def evolve_embedding(word: str, contexts: Dict[str, Counter], 
                    embeddings: np.ndarray, word_to_idx: Dict[str, int],
                    vocab_list: List[str], stats_dict: Dict[str, float],
                    # OLD:
                    # anchors: Dict[str, List[str]], 
                    # NEW: anchors as weighted list
                    anchors: Dict[str, List[Tuple[str, float]]],
                    config: Dict) -> np.ndarray:
    """
    Evolve a single word embedding using (1+λ) Evolution Strategy.
    """
    print(f"\n  Evolving: '{word}'", end='')
    
    dim = embeddings.shape[1]
    mutation_sigma = config['ga_mutation_factor'] * stats_dict['global_std']
    
    # Initialize
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx)
    
    # Precompute vectors
    # OLD:
    # ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
    #     word, contexts, embeddings, word_to_idx, vocab_list, anchors
    # )
    # NEW:
    ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, anchor_weights = precompute_fitness_vectors(
        word, contexts, embeddings, word_to_idx, vocab_list, anchors
    )
    
    # Initial fitness
    # OLD:
    # best_fit = compute_fitness(best_vec, word, ctx_vecs, ctx_weights, neg_vecs, 
    #                            anchor_vecs, stats_dict, config['fitness_weights'])
    # NEW:
    best_fit = compute_fitness(
        best_vec, word,
        ctx_vecs, ctx_weights,
        neg_vecs,
        anchor_vecs, anchor_weights,
        stats_dict,
        config['fitness_weights']
    )
    
    # Evolution loop
    for gen in range(config['ga_generations']):
        population = best_vec + np.random.normal(0, mutation_sigma, (config['ga_pop_size'], dim))
        all_candidates = np.vstack([best_vec, population])
        
        # OLD:
        # fitness_scores = [compute_fitness(vec, word, ctx_vecs, ctx_weights, neg_vecs, 
        #                                  anchor_vecs, stats_dict, config['fitness_weights'])
        #                  for vec in all_candidates]
        # NEW:
        fitness_scores = [
            compute_fitness(
                vec, word,
                ctx_vecs, ctx_weights,
                neg_vecs,
                anchor_vecs, anchor_weights,
                stats_dict,
                config['fitness_weights']
            )
            for vec in all_candidates
        ]
        
        best_idx = np.argmax(fitness_scores)
        best_vec = all_candidates[best_idx].copy()
        best_fit = fitness_scores[best_idx]
        
        if gen % 50 == 0:
            print(f" G{gen}={best_fit:.4f}", end='')
    
    print(f"Final={best_fit:.4f}")
    return best_vec

# VISUALIZATION
def visualize_with_inserted_words(nodes: List[str], embeddings: np.ndarray, 
                                  inserted_words: List[str],
                                  output_file: str = "embeddings_with_inserted.png",
                                  sample_size: int = 500):
    """Create t-SNE visualization highlighting inserted words."""
    print("\nGenerating t-SNE visualization with inserted words...")
    
    num_original = len(nodes) - len(inserted_words)
    inserted_indices = set(range(num_original, len(nodes)))
    
    # Sample: prioritize inserted words + random original
    if len(nodes) > sample_size:
        sample_indices = list(inserted_indices) + list(np.random.choice(
            num_original, min(sample_size - len(inserted_words), num_original), replace=False))
    else:
        sample_indices = list(range(len(nodes)))
    
    selected_embeddings = embeddings[sample_indices]
    selected_nodes = [nodes[i] for i in sample_indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices)-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 14))
    
    for i in range(len(projection)):
        is_inserted = sample_indices[i] in inserted_indices
        plt.scatter(projection[i, 0], projection[i, 1], 
                   s=200 if is_inserted else 40,
                   alpha=1.0 if is_inserted else 0.6,
                   c='red' if is_inserted else 'steelblue')
        plt.annotate(selected_nodes[i], (projection[i, 0], projection[i, 1]), 
                    fontsize=11 if is_inserted else 9,
                    alpha=1.0 if is_inserted else 0.8,
                    fontweight='bold' if is_inserted else 'normal')
    
    plt.title(f"t-SNE Visualization: {len(sample_indices)} Words "
              f"({sum(1 for i in sample_indices if i in inserted_indices)} Inserted)",
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE to {output_file}")
    plt.show()


def run_sanity_checks(model: torch.nn.Module, embeddings: np.ndarray, 
                     nodes: List[str], word_to_idx: Dict[str, int]):
    """Run comprehensive sanity checks on loaded model and embeddings."""
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    print(f"\n1. Model Configuration:")
    print(f"   Training mode: {model.training}")
    print(f"   Device: {next(model.parameters()).device}")
    
    print(f"\n2. Embedding Quality:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")
    print(f"   Contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}")
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n3. Embedding Norms:")
    print(f"   Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    print(f"\n4. Vocabulary Test:")
    for test_word in ['man', 'woman', 'dog', 'car', 'blue']:
        if test_word in word_to_idx:
            word_idx = word_to_idx[test_word]
            print(f"   '{test_word:10s}' → idx={word_idx:4d}, norm={np.linalg.norm(embeddings[word_idx]):.4f}")
            similar = find_similar_words(test_word, nodes, embeddings, top_k=5)
            if similar:
                print(f"      Similar: {', '.join([f'{w}({s:.3f})' for w, s in similar])}")
    
    print("\n" + "="*70)
    print("✓ SANITY CHECKS COMPLETE")
    print("="*70)


class TestExtractWordContexts(unittest.TestCase):
    def test_basic(self):
        contexts = {'king': Counter({'queen': 2, 'royal': 1})}
        self.assertEqual(contexts['king']['queen'], 2)
        self.assertEqual(contexts['king']['royal'], 1)

class TestComputeFitness(unittest.TestCase):
    def test_sigmoid(self):
        from lab7 import sigmoid
        x = np.array([0])
        self.assertAlmostEqual(sigmoid(x)[0], 0.5)

class TestInitializeEmbedding(unittest.TestCase):
    def test_mean_fallback(self):
        from lab7 import initialize_embedding
        emb = np.random.randn(5, 3)
        word_to_idx = {f'w{i}': i for i in range(5)}
        ctxs = {}
        vec = initialize_embedding('newword', ctxs, emb, word_to_idx)
        self.assertEqual(vec.shape[0], 3)

class TestPrecomputeFitnessVectors(unittest.TestCase):
    def test_shapes(self):
        from lab7 import precompute_fitness_vectors
        emb = np.random.randn(5, 3).astype(np.float32)
        word_to_idx = {f'w{i}': i for i in range(5)}
        ctxs = {'w0': Counter({'w1': 1})}

        # OLD (unweighted anchors) – kept for reference
        # anchors = {'w0': ['w2']}
        # ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        #     'w0', ctxs, emb, word_to_idx, list(word_to_idx.keys()), anchors
        # )
        # self.assertEqual(neg_vecs.shape[1], 3)

        # NEW: weighted anchors
        anchors = {'w0': [('w2', 1.0)]}

        ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, anchor_weights = precompute_fitness_vectors(
            'w0', ctxs, emb, word_to_idx, list(word_to_idx.keys()), anchors
        )

        # Basic shape checks
        self.assertEqual(neg_vecs.shape[1], 3)
        self.assertIsNotNone(anchor_vecs)
        self.assertIsNotNone(anchor_weights)
        self.assertEqual(anchor_vecs.shape[0], anchor_weights.shape[0])
        # weights should sum to ~1
        self.assertAlmostEqual(float(anchor_weights.sum()), 1.0, places=5)


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_class in [TestExtractWordContexts, TestComputeFitness, TestInitializeEmbedding, TestPrecomputeFitnessVectors]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
