import math
from collections import defaultdict, Counter

def apply_prf(query_tokens, search_index, n=5, t=20, lmbda=0.8):
    """
    Expands a search query using Pseudo-Relevance Feedback (PRF).
    """
    # 1. Baseline retrieval
    base_hits = search_index.search(query_tokens, k=n)
    if not base_hits:
        return {term: 1.0 / len(query_tokens) for term in query_tokens}

    total_score = sum(hit.score for hit in base_hits)
    candidate_terms = defaultdict(float)
    
    # 2. Term extraction & document weighting
    for hit in base_hits:
        doc_weight = hit.score / total_score if total_score > 0 else (1.0 / n)
        tf = Counter(hit.tokens)
        doc_len = len(hit.tokens)
        
        for term, count in tf.items():
            candidate_terms[term] += (count / doc_len) * doc_weight

    # 3. Apply IDF and rank candidates
    term_scores = []
    for term, weight in candidate_terms.items():
        idf = search_index.get_idf(term)
        term_scores.append((term, weight * idf))

    term_scores.sort(key=lambda x: x[1], reverse=True)
    top_expansion_terms = term_scores[:t]
    
    # 4. Construct the weighted blended query
    expanded_query = defaultdict(float)
    sum_expansion_scores = sum(score for _, score in top_expansion_terms)

    # Original terms get (1 - lambda)
    for term in query_tokens:
        expanded_query[term] += (1.0 - lmbda) * (1.0 / len(query_tokens))

    # Expansion terms get (lambda)
    if sum_expansion_scores > 0:
        for term, score in top_expansion_terms:
            expanded_query[term] += lmbda * (score / sum_expansion_scores)

    return dict(expanded_query)
