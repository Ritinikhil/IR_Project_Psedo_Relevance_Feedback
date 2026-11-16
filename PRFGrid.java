package ca1;

import java.io.PrintWriter;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;

public class PRFGrid {
    static Analyzer analyzer;
    static IndexSearcher searcher;
    
    // caches to save time so we don't recalculate the same document multiple times
    static Map<Integer, Map<String, Double>> termFreqCache = new HashMap<>();
    static Map<Integer, String> docIdCache = new HashMap<>();

    // breaks down raw text into individual tokens (words)
    static List<String> tokenize(String text) throws Exception {
        List<String> tokens = new ArrayList<>();
        try (TokenStream ts = analyzer.tokenStream("contents", text)) {
            CharTermAttribute attr = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) {
                tokens.add(attr.toString());
            }
            ts.end();
        }
        return tokens;
    }

    // calculates the term frequency for a specific document and caches it
    static Map<String, Double> getDocTermFrequencies(int docId) throws Exception {
        Map<String, Double> freqs = termFreqCache.get(docId);
        if (freqs == null) {
            org.apache.lucene.document.Document doc = searcher.storedFields().document(docId);
            List<String> tokens = tokenize(doc.get("title") + " " + doc.get("text"));
            
            freqs = new HashMap<>();
            for (String word : tokens) {
                // weight each word by how often it appears relative to document length
                freqs.merge(word, 1.0 / tokens.size(), Double::sum);
            }
            termFreqCache.put(docId, freqs);
            docIdCache.put(docId, doc.get("id"));
        }
        return freqs;
    }

    // turns our map of weighted words into an actual Lucene query
    static Query buildWeightedQuery(Map<String, Double> weights) {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (Map.Entry<String, Double> entry : weights.entrySet()) {
            builder.add(
                new BoostQuery(new TermQuery(new Term("contents", entry.getKey())), entry.getValue().floatValue()),
                BooleanClause.Occur.SHOULD
            );
        }
        return builder.build();
    }

    // helper to save the search results to a TREC format file
    static void writeResults(PrintWriter writer, int queryId, ScoreDoc[] hits, String runName) throws Exception {
        for (int i = 0; i < hits.length; i++) {
            getDocTermFrequencies(hits[i].doc); // ensure it's in the cache
            writer.printf(Locale.ROOT, "%d Q0 %s %d %.6f %s%n", 
                queryId, docIdCache.get(hits[i].doc), i + 1, hits[i].score, runName);
        }
    }

    public static void main(String[] args) throws Exception {
        analyzer = Indexer.makeAnalyzer(args[1]);
        List<Searcher.CranQuery> queries = Searcher.parseQueries(Paths.get(args[3]));
        Path outputDir = Paths.get(args[4]);
        Files.createDirectories(outputDir);
        String tag = args[5];
        
        // our grid search dials: N (top docs), T (new terms), L (lambda weight)
        int[] topNDocs = {3, 5, 10};
        int[] topTTerms = {5, 10, 20, 40};
        double[] lambdaWeights = {0.2, 0.4, 0.6, 0.8};
        
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(args[0])))) {
            searcher = new IndexSearcher(reader);
            searcher.setSimilarity(Searcher.makeSimilarity(args[2]));
            int totalDocsInIndex = reader.numDocs();
            
            // set up writers for all the different combination run files
            Map<String, PrintWriter> runFiles = new LinkedHashMap<>();
            runFiles.put(tag + "_base", new PrintWriter(Files.newBufferedWriter(outputDir.resolve(tag + "_base.run"))));
            
            for (int n : topNDocs) {
                for (int t : topTTerms) {
                    for (double l : lambdaWeights) {
                        String name = String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, l);
                        runFiles.put(name, new PrintWriter(Files.newBufferedWriter(outputDir.resolve(name + ".run"))));
                    }
                }
            }
            
            // run the pipeline for every single query
            for (Searcher.CranQuery query : queries) {
                List<String> queryTokens = tokenize(query.text);
                Map<String, Double> originalQueryWeights = new HashMap<>();
                for (String word : queryTokens) {
                    originalQueryWeights.merge(word, 1.0 / queryTokens.size(), Double::sum);
                }
                
                // step 1: do a normal search to get our baseline documents
                ScoreDoc[] baselineHits = searcher.search(buildWeightedQuery(originalQueryWeights), 1000).scoreDocs;
                writeResults(runFiles.get(tag + "_base"), query.id, baselineHits, tag + "_base");
                
                // step 2: test different N values (how many top docs to look at)
                for (int n : topNDocs) {
                    int validN = Math.min(n, baselineHits.length);
                    
                    // calculate total score of top N docs to weight them properly
                    double totalScore = 0;
                    for (int i = 0; i < validN; i++) totalScore += baselineHits[i].score;
                    
                    // extract candidate terms from these top N documents
                    Map<String, Double> candidateTerms = new HashMap<>();
                    for (int i = 0; i < validN; i++) {
                        double docWeight = (totalScore > 0) ? (baselineHits[i].score / totalScore) : (1.0 / validN);
                        for (Map.Entry<String, Double> term : getDocTermFrequencies(baselineHits[i].doc).entrySet()) {
                            candidateTerms.merge(term.getKey(), docWeight * term.getValue(), Double::sum);
                        }
                    }
                    
                    // step 3: score candidate terms using IDF (Inverse Document Frequency)
                    List> scoredTerms = new ArrayList<>();
                    for (Map.Entry<String, Double> term : candidateTerms.entrySet()) {
                        int docFreq = reader.docFreq(new Term("contents", term.getKey()));
                        double idf = Math.log(1.0 + totalDocsInIndex / (1.0 + docFreq));
                        scoredTerms.add(new AbstractMap.SimpleEntry<>(term.getKey(), term.getValue() * idf));
                    }
                    
                    // sort so the highest scoring terms are at the top of the list
                    scoredTerms.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));
                    
                    // step 4: test different T values (how many new terms to add)
                    for (int t : topTTerms) {
                        int validT = Math.min(t, scoredTerms.size());
                        double termScoreSum = 0;
                        for (int i = 0; i < validT; i++) termScoreSum += scoredTerms.get(i).getValue();
                        
                        // test different Lambda values (how much to trust the new terms vs the old query)
                        for (double l : lambdaWeights) {
                            Map<String, Double> finalQuery = new HashMap<>();
                            
                            // add original query terms, shrinking their weight by (1 - lambda)
                            for (Map.Entry<String, Double> orig : originalQueryWeights.entrySet()) {
                                finalQuery.merge(orig.getKey(), (1 - l) * orig.getValue(), Double::sum);
                            }
                            
                            // add the newly discovered terms, boosting them by lambda
                            for (int i = 0; i < validT; i++) {
                                finalQuery.merge(scoredTerms.get(i).getKey(), l * scoredTerms.get(i).getValue() / termScoreSum, Double::sum);
                            }
                            
                            // execute the final, smarter search and save it
                            String runName = String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, l);
                            writeResults(runFiles.get(runName), query.id, searcher.search(buildWeightedQuery(finalQuery), 1000).scoreDocs, runName);
                        }
                    }
                }
            }
            
            // clean up
            for (PrintWriter pw : runFiles.values()) pw.close();
            System.out.println("Wrote " + runFiles.size() + " runs for tag " + tag);
        }
    }
}
