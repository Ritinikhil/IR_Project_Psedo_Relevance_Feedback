import java.nio.file.*;
import java.util.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;

public class GridSearch {
    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: GridSearch <indexDir> <analyzer> <cran.qry> <outDir>");
            return;
        }
        Analyzer analyzer = Indexer.makeAnalyzer(args[1]);
        List<Searcher.CranQuery> queries = Searcher.parseQueries(Paths.get(args[2]));
        Path outDir = Paths.get(args[3]);
        Files.createDirectories(outDir);

        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(args[0])))) {
            IndexSearcher searcher = new IndexSearcher(reader);
            int count = 0;
            for (int ki = 2; ki <= 30; ki++) {          // k1 = 0.2 ... 3.0
                for (int bi = 0; bi <= 10; bi++) {      // b  = 0.0 ... 1.0
                    float k1 = ki / 10f, b = bi / 10f;
                    searcher.setSimilarity(new BM25Similarity(k1, b));
                    String name = String.format(Locale.ROOT, "%s_k%.1f_b%.1f", args[1], k1, b);
                    Searcher.run(searcher, analyzer, queries, outDir.resolve(name + ".run"), name);
                    count++;
                }
            }
            System.out.println("Wrote " + count + " run files to " + outDir);
        }
    }
}
mvn -q compile
