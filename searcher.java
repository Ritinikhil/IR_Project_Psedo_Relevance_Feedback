/*Parse the queries by physical position to fix the ID mismatch in the dataset*/
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.FSDirectory;

public class Searcher {

    static class CranQuery {
        int id;
        String text = "";
    }

    /** Reads cran.qry: ".I 001" starts a query, ".W" starts its text. */
    static List<CranQuery> parseQueries(Path file) throws Exception {
        List<CranQuery> qs = new ArrayList<>();
        CranQuery cur = null;
        boolean inText = false;
        for (String line : Files.readAllLines(file, StandardCharsets.ISO_8859_1)) {
            if (line.startsWith(".I")) {
                cur = new CranQuery();
                cur.id = Integer.parseInt(line.substring(2).trim()); // "001" becomes 1
                qs.add(cur);
                inText = false;
            } else if (line.startsWith(".W")) {
                inText = true;
            } else if (cur != null && inText) {
                cur.text += line + " ";
            }
        }
        return qs;
    }

    /** "bm25" = defaults (k1=1.2, b=0.75), "bm25:1.5:0.6", "vsm", "lmd" or "lmd:2000". */
    static Similarity makeSimilarity(String spec) {
        String[] p = spec.split(":");
        switch (p[0]) {
            case "bm25":
                return p.length == 3
                        ? new BM25Similarity(Float.parseFloat(p[1]), Float.parseFloat(p[2]))
                        : new BM25Similarity();
            case "vsm":
                return new ClassicSimilarity(); // Lucene's TF-IDF vector space model
            case "lmd":
                return p.length == 2
                        ? new LMDirichletSimilarity(Float.parseFloat(p[1]))
                        : new LMDirichletSimilarity();
            default:
                throw new IllegalArgumentException("Unknown similarity: " + spec);
        }
    }

    /** Runs every query and writes a TREC run file. GridSearch will reuse this. */
    static void run(IndexSearcher searcher, Analyzer analyzer, List<CranQuery> queries,
                    Path out, String runName) throws Exception {
        QueryParser parser = new QueryParser("contents", analyzer);
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out))) {
            for (CranQuery q : queries) {
                // escape() stops characters like ? ( ) : - from breaking the parser
                Query query = parser.parse(QueryParser.escape(q.text.trim()));
                ScoreDoc[] hits = searcher.search(query, 1000).scoreDocs;
                for (int i = 0; i < hits.length; i++) {
                    String docId = searcher.storedFields().document(hits[i].doc).get("id");
                    w.printf(Locale.ROOT, "%d Q0 %s %d %.6f %s%n",
                            q.id, docId, i + 1, hits[i].score, runName);
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println("Usage: Searcher <indexDir> <standard|english> <simSpec> <cran.qry> <out.run>");
            return;
        }
        Analyzer analyzer = Indexer.makeAnalyzer(args[1]);
        List<CranQuery> queries = parseQueries(Paths.get(args[3]));
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(args[0])))) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(makeSimilarity(args[2]));
            run(searcher, analyzer, queries, Paths.get(args[4]), args[1] + "_" + args[2]);
        }
        System.out.println("Ran " + queries.size() + " queries -> " + args[4]);
    }
}
mvn -q compile
