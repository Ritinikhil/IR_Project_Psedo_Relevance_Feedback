cd ~/ir-ca1/search/src/main/java/ca1 && cp PRFGrid.java /tmp/PRFGrid.java.bak
cat > PRFGrid.java << 'EOF'
package ca1;

import java.io.PrintWriter;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;

/**
 * Pseudo-relevance feedback over a grid of settings.
 * Usage: PRFGrid <indexDir> <analyzer> <simSpec> <cran.qry> <outDir> <tag>
 */
public class PRFGrid {
    static Analyzer analyzer;
    static IndexSearcher searcher;

    // settings swept: N feedback documents, T expansion terms, lambda weight of the new terms
    static final int[] TOP_N = {3, 5, 10};
    static final int[] TOP_T = {5, 10, 20, 40};
    static final double[] LAMBDA = {0.2, 0.4, 0.6, 0.8};

    // caches so each document is tokenised / looked up only once
    static final Map<Integer, Map<String, Double>> tfCache = new HashMap<>();
    static final Map<Integer, String> idCache = new HashMap<>();

    /** Cleans text with the analyzer and returns the resulting words. */
    static List<String> tokenize(String text) throws Exception {
        List<String> tokens = new ArrayList<>();
        try (TokenStream ts = analyzer.tokenStream("contents", text)) {
            CharTermAttribute attr = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) tokens.add(attr.toString());
            ts.end();
        }
        return tokens;
    }

    /** Word weights of one document: how often the word appears divided by document length. */
    static Map<String, Double> getDocTermFrequencies(int docId) throws Exception {
        Map<String, Double> freqs = tfCache.get(docId);
        if (freqs == null) {
            Document doc = searcher.storedFields().document(docId);
            List<String> tokens = tokenize(doc.get("title") + " " + doc.get("text"));
            freqs = new HashMap<>();
            for (String word : tokens) freqs.merge(word, 1.0 / tokens.size(), Double::sum);
            tfCache.put(docId, freqs);
        }
        return freqs;
    }

    /** The Cranfield id (as used in the qrels) of a Lucene document. */
    static String externalId(int docId) throws Exception {
        String id = idCache.get(docId);
        if (id == null) {
            id = searcher.storedFields().document(docId).get("id");
            idCache.put(docId, id);
        }
        return id;
    }

    /** Turns a map of word weights into a Lucene query where each word is boosted by its weight. */
    static Query buildWeightedQuery(Map<String, Double> weights) {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (Map.Entry<String, Double> e : weights.entrySet()) {
            builder.add(new BoostQuery(new TermQuery(new Term("contents", e.getKey())),
                    e.getValue().floatValue()), BooleanClause.Occur.SHOULD);
        }
        return builder.build();
    }

    /**
     * Feedback steps 2 and 3: weight the top n documents by their search score, give every word in them
     * (frequency x document weight) x IDF points, and return the words best-first.
     */
    static List<Map.Entry<String, Double>> rankExpansionTerms(DirectoryReader reader, ScoreDoc[] hits, int n) throws Exception {
        int validN = Math.min(n, hits.length);
        double totalScore = 0;
        for (int i = 0; i < validN; i++) totalScore += hits[i].score;

        Map<String, Double> candidates = new HashMap<>();
        for (int i = 0; i < validN; i++) {
            double docWeight = (totalScore > 0) ? hits[i].score / totalScore : 1.0 / validN;
            for (Map.Entry<String, Double> term : getDocTermFrequencies(hits[i].doc).entrySet()) {
                candidates.merge(term.getKey(), docWeight * term.getValue(), Double::sum);
            }
        }

        List<Map.Entry<String, Double>> scored = new ArrayList<>();
        for (Map.Entry<String, Double> term : candidates.entrySet()) {
            int docFreq = reader.docFreq(new Term("contents", term.getKey()));
            double idf = Math.log(1.0 + reader.numDocs() / (1.0 + docFreq));
            scored.add(new AbstractMap.SimpleEntry<>(term.getKey(), term.getValue() * idf));
        }
        scored.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));
        return scored;
    }

    /** Writes one query's results in TREC format. */
    static void writeResults(PrintWriter writer, int queryId, ScoreDoc[] hits, String runName) throws Exception {
        for (int i = 0; i < hits.length; i++) {
            writer.printf(Locale.ROOT, "%d Q0 %s %d %.6f %s%n",
                    queryId, externalId(hits[i].doc), i + 1, hits[i].score, runName);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 6) {
            System.err.println("Usage: PRFGrid <indexDir> <analyzer> <simSpec> <cran.qry> <outDir> <tag>");
            return;
        }
        analyzer = Indexer.makeAnalyzer(args[1]);
        List<Searcher.CranQuery> queries = Searcher.parseQueries(Paths.get(args[3]));
        Path outDir = Paths.get(args[4]);
        Files.createDirectories(outDir);
        String tag = args[5];

        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(args[0])))) {
            searcher = new IndexSearcher(reader);
            searcher.setSimilarity(Searcher.makeSimilarity(args[2]));

            // one run file for the no-feedback baseline and one for every (N, T, lambda) combination
            Map<String, PrintWriter> runFiles = new LinkedHashMap<>();
            runFiles.put(tag + "_base", new PrintWriter(Files.newBufferedWriter(outDir.resolve(tag + "_base.run"))));
            for (int n : TOP_N) for (int t : TOP_T) for (double l : LAMBDA) {
                String name = String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, l);
                runFiles.put(name, new PrintWriter(Files.newBufferedWriter(outDir.resolve(name + ".run"))));
            }

            for (Searcher.CranQuery query : queries) {
                // original query words, each weighted 1 / (number of words)
                List<String> queryTokens = tokenize(query.text);
                Map<String, Double> original = new HashMap<>();
                for (String word : queryTokens) original.merge(word, 1.0 / queryTokens.size(), Double::sum);

                // step 1: normal search (baseline)
                ScoreDoc[] baseline = searcher.search(buildWeightedQuery(original), 1000).scoreDocs;
                writeResults(runFiles.get(tag + "_base"), query.id, baseline, tag + "_base");

                for (int n : TOP_N) {
                    // steps 2-3: expansion words from the top n documents, best first
                    List<Map.Entry<String, Double>> terms = rankExpansionTerms(reader, baseline, n);

                    for (int t : TOP_T) {
                        int validT = Math.min(t, terms.size());
                        double termScoreSum = 0;
                        for (int i = 0; i < validT; i++) termScoreSum += terms.get(i).getValue();

                        for (double l : LAMBDA) {
                            // step 4: new query = (1 - lambda) * original words + lambda * expansion words
                            Map<String, Double> expanded = new HashMap<>();
                            for (Map.Entry<String, Double> o : original.entrySet())
                                expanded.merge(o.getKey(), (1 - l) * o.getValue(), Double::sum);
                            for (int i = 0; i < validT; i++)
                                expanded.merge(terms.get(i).getKey(), l * terms.get(i).getValue() / termScoreSum, Double::sum);

                            String runName = String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, l);
                            writeResults(runFiles.get(runName), query.id,
                                    searcher.search(buildWeightedQuery(expanded), 1000).scoreDocs, runName);
                        }
                    }
                }
            }
            for (PrintWriter pw : runFiles.values()) pw.close();
            System.out.println("Wrote " + runFiles.size() + " runs for tag " + tag);
        }
    }
}
EOF
python3 -c "s=open('PRFGrid.java').read(); print('braces', s.count('{'), s.count('}'))"
