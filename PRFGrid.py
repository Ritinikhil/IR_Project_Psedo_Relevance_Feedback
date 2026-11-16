import sys
import os
import math
from collections import defaultdict
import lucene
from java.nio.file import Paths
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, BooleanQuery, TermQuery, BoostQuery, BooleanClause
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute

lucene.initVM(vmargs=['-Djava.awt.headless=true']) #java VM to use PyLucene

class PRFGridPython:
    def __init__(self, index_dir, analyzer, similarity):
        self.analyzer = analyzer
        directory = FSDirectory.open(Paths.get(index_dir))
        self.reader = DirectoryReader.open(directory)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(similarity)
        
        self.total_docs = self.reader.numDocs()
        self.term_freq_cache = {}
        self.doc_id_cache = {}

    def tokenize(self, text):
        """Breaks down raw text into individual tokens (words)"""
        tokens = []
        stream = self.analyzer.tokenStream("contents", text)
        attr = stream.addAttribute(CharTermAttribute.class_)
        stream.reset()
        while stream.incrementToken():
            tokens.append(attr.toString())
        stream.end()
        stream.close()
        return tokens

    def get_doc_term_frequencies(self, doc_id):
        """Calculates the term frequency for a specific document and caches it"""
        if doc_id not in self.term_freq_cache:
            doc = self.searcher.storedFields().document(doc_id)
            # Combine title and text
            text_content = doc.get("title") + " " + doc.get("text")
            tokens = self.tokenize(text_content)
            
            freqs = defaultdict(float)
            if tokens:
                weight = 1.0 / len(tokens)
                for word in tokens:
                    freqs[word] += weight
                    
            self.term_freq_cache[doc_id] = freqs
            self.doc_id_cache[doc_id] = doc.get("id")
            
        return self.term_freq_cache[doc_id]

    def build_weighted_query(self, weights):
        """Turns our map of weighted words into an actual Lucene query"""
        builder = BooleanQuery.Builder()
        for word, weight in weights.items():
            term = Term("contents", word)
            term_query = TermQuery(term)
            boost_query = BoostQuery(term_query, float(weight))
            builder.add(boost_query, BooleanClause.Occur.SHOULD)
        return builder.build()

    def write_results(self, file_handle, query_id, hits, run_name):
        """Helper to save the search results to a TREC format file"""
        for i, hit in enumerate(hits):
            self.get_doc_term_frequencies(hit.doc)  # ensure it's in the cache
            doc_string_id = self.doc_id_cache[hit.doc]
            file_handle.write(f"{query_id} Q0 {doc_string_id} {i + 1} {hit.score:.6f} {run_name}\n")

    def run_grid_search(self, queries, output_dir, tag, top_n_docs, top_t_terms, lambda_weights):
        os.makedirs(output_dir, exist_ok=True)
        run_files = {}
        
        # Setup file writers for every combination
        base_name = f"{tag}_base"
        run_files[base_name] = open(os.path.join(output_dir, f"{base_name}.run"), "w")
        
        for n in top_n_docs:
            for t in top_t_terms:
                for l in lambda_weights:
                    name = f"{tag}_N{n}_T{t}_L{l:.1f}"
                    run_files[name] = open(os.path.join(output_dir, f"{name}.run"), "w")

        # Run the pipeline for every single query
        for query_id, query_text in queries:
            query_tokens = self.tokenize(query_text)
            original_weights = defaultdict(float)
            
            if query_tokens:
                base_weight = 1.0 / len(query_tokens)
                for word in query_tokens:
                    original_weights[word] += base_weight

            # Step 1: Baseline search
            base_query = self.build_weighted_query(original_weights)
            baseline_hits = self.searcher.search(base_query, 1000).scoreDocs
            self.write_results(run_files[base_name], query_id, baseline_hits, base_name)

            # Step 2: Test different N values (how many top docs to look at)
            for n in top_n_docs:
                valid_n = min(n, len(baseline_hits))
                total_score = sum(baseline_hits[i].score for i in range(valid_n))
                
                candidate_terms = defaultdict(float)
                for i in range(valid_n):
                    doc_weight = (baseline_hits[i].score / total_score) if total_score > 0 else (1.0 / valid_n)
                    doc_freqs = self.get_doc_term_frequencies(baseline_hits[i].doc)
                    for word, freq in doc_freqs.items():
                        candidate_terms[word] += doc_weight * freq

                # Step 3: Score candidate terms using IDF
                scored_terms = []
                for word, score in candidate_terms.items():
                    doc_freq = self.reader.docFreq(Term("contents", word))
                    idf = math.log(1.0 + self.total_docs / (1.0 + doc_freq))
                    scored_terms.append((word, score * idf))
                
                # Sort highest scoring terms to the top
                scored_terms.sort(key=lambda x: x[1], reverse=True)

                # Step 4: Test different T and Lambda values
                for t in top_t_terms:
                    valid_t = min(t, len(scored_terms))
                    term_score_sum = sum(scored_terms[i][1] for i in range(valid_t))

                    for l in lambda_weights:
                        final_query_weights = defaultdict(float)
                        
                        # Add original query terms (shrunk by 1 - lambda)
                        for word, weight in original_weights.items():
                            final_query_weights[word] += (1 - l) * weight
                            
                        # Add new discovered terms (boosted by lambda)
                        if term_score_sum > 0:
                            for i in range(valid_t):
                                word, score = scored_terms[i]
                                final_query_weights[word] += l * (score / term_score_sum)

                        # Execute final smarter search
                        run_name = f"{tag}_N{n}_T{t}_L{l:.1f}"
                        final_query = self.build_weighted_query(final_query_weights)
                        final_hits = self.searcher.search(final_query, 1000).scoreDocs
                        self.write_results(run_files[run_name], query_id, final_hits, run_name)

        # Cleanup
        for file_handle in run_files.values():
            file_handle.close()
        print(f"Wrote {len(run_files)} runs for tag {tag}")
