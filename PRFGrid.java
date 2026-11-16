cat > ~/ir-ca1/search/src/main/java/ca1/PRFGrid.java << 'EOF'
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

/** Usage: PRFGrid indexDir analyzer simSpec cran.qry outDir tag */
public class PRFGrid {
    static Analyzer analyzer;
    static IndexSearcher s;
    static Map<Integer, Map<String, Double>> tfCache = new HashMap<>();
    static Map<Integer, String> idCache = new HashMap<>();

    static List<String> tokens(String text) throws Exception {
        List<String> out = new ArrayList<>();
        try (TokenStream ts = analyzer.tokenStream("contents", text)) {
            CharTermAttribute a = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) out.add(a.toString());
            ts.end();
        }
        return out;
    }

    static Map<String, Double> docTf(int id) throws Exception {
        Map<String, Double> m = tfCache.get(id);
        if (m == null) {
            org.apache.lucene.document.Document d = s.storedFields().document(id);
            List<String> t = tokens(d.get("title") + " " + d.get("text"));
            m = new HashMap<>();
            for (String x : t) m.merge(x, 1.0 / t.size(), Double::sum);
            tfCache.put(id, m);
            idCache.put(id, d.get("id"));
        }
        return m;
    }

    static Query build(Map<String, Double> w) {
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (Map.Entry<String, Double> e : w.entrySet())
            b.add(new BoostQuery(new TermQuery(new Term("contents", e.getKey())), e.getValue().floatValue()),
                  BooleanClause.Occur.SHOULD);
        return b.build();
    }

    static void write(PrintWriter pw, int qid, ScoreDoc[] hits, String name) throws Exception {
        for (int i = 0; i < hits.length; i++) {
            docTf(hits[i].doc);
            pw.printf(Locale.ROOT, "%d Q0 %s %d %.6f %s%n", qid, idCache.get(hits[i].doc), i + 1, hits[i].score, name);
        }
    }

    public static void main(String[] a) throws Exception {
        analyzer = Indexer.makeAnalyzer(a[1]);
        List<Searcher.CranQuery> qs = Searcher.parseQueries(Paths.get(a[3]));
        Path out = Paths.get(a[4]);
        Files.createDirectories(out);
        String tag = a[5];
        int[] Ns = {3, 5, 10}, Ts = {5, 10, 20, 40};
        double[] Ls = {0.2, 0.4, 0.6, 0.8};
        try (DirectoryReader r = DirectoryReader.open(FSDirectory.open(Paths.get(a[0])))) {
            s = new IndexSearcher(r);
            s.setSimilarity(Searcher.makeSimilarity(a[2]));
            int numDocs = r.numDocs();
            Map<String, PrintWriter> W = new LinkedHashMap<>();
            W.put(tag + "_base", new PrintWriter(Files.newBufferedWriter(out.resolve(tag + "_base.run"))));
            for (int n : Ns) for (int t : Ts) for (double l : Ls) {
                String name = String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, l);
                W.put(name, new PrintWriter(Files.newBufferedWriter(out.resolve(name + ".run"))));
            }
            for (Searcher.CranQuery q : qs) {
                List<String> qt = tokens(q.text);
                Map<String, Double> qw = new HashMap<>();
                for (String t : qt) qw.merge(t, 1.0 / qt.size(), Double::sum);
                ScoreDoc[] base = s.search(build(qw), 1000).scoreDocs;
                write(W.get(tag + "_base"), q.id, base, tag + "_base");
                for (int n : Ns) {
                    int nn = Math.min(n, base.length);
                    double sum = 0;
                    for (int i = 0; i < nn; i++) sum += base[i].score;
                    Map<String, Double> e = new HashMap<>();
                    for (int i = 0; i < nn; i++) {
                        double pd = sum > 0 ? base[i].score / sum : 1.0 / nn;
                        for (Map.Entry<String, Double> x : docTf(base[i].doc).entrySet())
                            e.merge(x.getKey(), pd * x.getValue(), Double::sum);
                    }
                    List<Map.Entry<String, Double>> cand = new ArrayList<>();
                    for (Map.Entry<String, Double> x : e.entrySet()) {
                        int df = r.docFreq(new Term("contents", x.getKey()));
                        cand.add(new AbstractMap.SimpleEntry<>(x.getKey(),
                                x.getValue() * Math.log(1.0 + numDocs / (1.0 + df))));
                    }
                    cand.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));
                    for (int t : Ts) {
                        int tt = Math.min(t, cand.size());
                        double z = 0;
                        for (int i = 0; i < tt; i++) z += cand.get(i).getValue();
                        for (double l : Ls) {
                            Map<String, Double> c = new HashMap<>();
                            for (Map.Entry<String, Double> x : qw.entrySet()) c.merge(x.getKey(), (1 - l) * x.getValue(), Double::sum);
                            for (int i = 0; i < tt; i++) c.merge(cand.get(i).getKey(), l * cand.get(i).getValue() / z, Double::sum);
                            String name = String.format(Locale.ROOT, "%s_N%d_T%d_L%.1f", tag, n, t, l);
                            write(W.get(name), q.id, s.search(build(c), 1000).scoreDocs, name);
                        }
                    }
                }
            }
            for (PrintWriter pw : W.values()) pw.close();
            System.out.println("Wrote " + W.size() + " runs for tag " + tag);
        }
    }
}
EOF
python3 -c "s=open('$HOME/ir-ca1/search/src/main/java/ca1/PRFGrid.java').read(); print(s.count('{'), s.count('}'))"
pgrep -fa "sweep|GridSearch|field_exp|run_|after" || echo NOTHING_RUNNING
cd ~/ir-ca1/search && export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED" && mvn -q compile 2>&1 | grep -v WARNING | head -30; echo COMPILE_DONE
