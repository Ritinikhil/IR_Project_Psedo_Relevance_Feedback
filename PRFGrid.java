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
 * Pseudo-relevance feedback (query expansion), tried with many settings.
 * Usage: PRFGrid <indexDir> <analyzer> <simSpec> <cran.qry> <outDir> <tag>
 *
 * The idea: search once, assume the top results are good, take useful words from them,
 * add those words to the question, and search again.
 */
public class PRFGrid {
    static Analyzer analyzer;          // cleans words (same one used for the index)
    static IndexSearcher searcher;     // runs the searches
    static String[] docIds;            // docIds[i] = Cranfield id of Lucene document i

    // the settings I tried
    static final int[] TOP_N = {3, 5, 10};              // N: how many top documents to learn from
    static final int[] TOP_T = {5, 10, 20, 40};         // T: how many new words to add
    static final double[] LAMBDA = {0.2, 0.4, 0.6, 0.8}; // lambda: how much the new words count

    // ---------- small helpers ----------

    /** Cleans text with the analyzer (lowercase, stop words, stemming) and returns the words. */
    static List<String> tokenize(String text) throws Exception {
        List<String> words = new ArrayList<>();
        try (TokenStream ts = analyzer.tokenStream("contents", text)) {
            CharTermAttribute attr = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) words.add(attr.toString());
            ts.end();
        }
        return words;
    }

    /** Gives every word of a document a weight: how often it appears, divided by the document length. */
    static Map<String, Double> getDocTermFrequencies(int docId) throws Exception {
        Document doc = searcher.storedFields().document(docId);
        List<String> words = tokenize(doc.get("title") + " " + doc.get("text"));
        Map<String, Double> weights = new HashMap<>();
        for (String w : words) weights.merge(w, 1.0 / words.size(), Double::sum);
        return weights;
    }

    /** Makes a Lucene query where each word counts as much as its weight. */
    static Query buildWeightedQuery(Map<String, Double> weights) {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (Map.Entry<String, Double> e : weights.entrySet()) {
            builder.add(new BoostQuery(new TermQuery(new Term("contents", e.getKey())),
                    e.getValue().floatValue()), BooleanClause.Occur.SHOULD);
        }
        return builder.build();
    }

    /** Searches with a weighted query and returns the top 1,000 results. */
    static ScoreDoc[] search(Map<String, Double> weights) throws Exception {
        return searcher.search(buildWeightedQuery(weights), 1000).scoreDocs;
    }

    // ---------- the feedback steps ----------

    /** Step 0: the question's words, each with weight 1 / (number of words). */
    static Map<String, Double> queryWeights(String text) throws Exception {
        List<String> words = tokenize(text);
        Map<String, Double> weights = new HashMap<>();
        for (String w : words) weights.merge(w, 1.0 / words.size(), Double::sum);
        return weights;
    }

    /**
     * Steps 2 and 3: look at the top n results and rank their words, best first.
     * A word gets points for appearing a lot in the good documents (better-ranked documents count more),
     * multiplied by IDF so that rare words score higher than common words.
     */
    static List<Map.Entry<String, Double>> rankExpansionTerms(ScoreDoc[] hits, int n) throws Exception {
        int validN = Math.min(n, hits.length);

        double totalScore = 0;                                   // used to turn scores into shares
        for (int i = 0; i < validN; i++) totalScore += hits[i].score;

        Map<String, Double> points = new HashMap<>();            // word -> points
        for (int i = 0; i < validN; i++) {
            double docShare = (totalScore > 0) ? hits[i].score / totalScore : 1.0 / validN;
            for (Map.Entry<String, Double> w : getDocTermFrequencies(hits[i].doc).entrySet()) {
                points.merge(w.getKey(), docShare * w.getValue(), Double::sum);
            }
        }

        List<Map.Entry<String, Double>> ranked = new ArrayList<>();
        for (Map.Entry<String, Double> w : points.entrySet()) {
            int docFreq = searcher.getIndexReader().docFreq(new Term("contents", w.getKey())); // documents containing the word
            double idf = Math.log(1.0 + searcher.getIndexReader().numDocs() / (1.0 + docFreq)); // rare word = bigger IDF
            ranked.add(new AbstractMap.SimpleEntry<>(w.getKey(), w.getValue() * idf));
        }
        ranked.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));   // best words first
        return ranked;
    }

    /** Step 4: new question = (1 - lambda) x original words + lambda x the best t new words. */
    static Map<String, Double> mixQuery(Map<String, Double> original,
                                        List<Map.Entry<String, Double>> terms, int t, double lambda) {
        int validT = Math.min(t, terms.size());
        double sum = 0;                                          // so the new words' weights add up to 1
        for (int i = 0; i < validT; i++) sum += terms.get(i).getValue();

        Map<String, Double> mixed = new HashMap<>();
        for (Map.Entry<String, Double> o : original.entrySet())
            mixed.merge(o.getKey(), (1 - lambda) * o.getValue(), Double::sum);
        for (int i = 0; i < validT; i++)
            mixed.merge(terms.get(i).getKey(), lambda * terms.get(i).getValue() / sum, Double::sum);
        return mixed;
    }

    // ---------- running everything ----------

    static String runName(String tag, int n, int t, double lambda) {
        return String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, lambda);
    }

    /** Runs one question: normal search first, then feedback with every (N, T, lambda). */
    static void processQuery(Searcher.CranQuery query, Map<String, PrintWriter> runFiles, String tag) throws Exception {
        Map<String, Double> original = queryWeights(query.text);

        ScoreDoc[] baseline = search(original);                                   // step 1: normal search
        writeResults(runFiles.get(tag + "_base"), query.id, baseline, tag + "_base");

        for (int n : TOP_N) {
            List<Map.Entry<String, Double>> terms = rankExpansionTerms(baseline, n); // steps 2-3
            for (int t : TOP_T) {
                for (double l : LAMBDA) {
                    String name = runName(tag, n, t, l);
                    ScoreDoc[] better = search(mixQuery(original, terms, t, l));      // step 4: search again
                    writeResults(runFiles.get(name), query.id, better, name);
                }
            }
        }
    }

    /** Writes one question's results in TREC format: question, Q0, document, rank, score, run name. */
    static void writeResults(PrintWriter writer, int queryId, ScoreDoc[] hits, String runName) {
        for (int i = 0; i < hits.length; i++) {
            writer.printf(Locale.ROOT, "%d Q0 %s %d %.6f %s%n",
                    queryId, docIds[hits[i].doc], i + 1, hits[i].score, runName);
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

            docIds = new String[reader.maxDoc()];                  // remember every document's id once
            for (int i = 0; i < docIds.length; i++) docIds[i] = searcher.storedFields().document(i).get("id");

            // one results file for no feedback, and one for every (N, T, lambda) combination
            Map<String, PrintWriter> runFiles = new LinkedHashMap<>();
            runFiles.put(tag + "_base", new PrintWriter(Files.newBufferedWriter(outDir.resolve(tag + "_base.run"))));
            for (int n : TOP_N) for (int t : TOP_T) for (double l : LAMBDA) {
                String name = runName(tag, n, t, l);
                runFiles.put(name, new PrintWriter(Files.newBufferedWriter(outDir.resolve(name + ".run"))));
            }

            for (Searcher.CranQuery query : queries) processQuery(query, runFiles, tag);

            for (PrintWriter pw : runFiles.values()) pw.close();
            System.out.println("Wrote " + runFiles.size() + " runs for tag " + tag);
        }
    }
}
