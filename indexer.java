/*  Read cran.all.1400 
Splitting the text using markers: T for title, .W for text
Bascially I m cleaning the dataset here to make it structured for the lucene Engine
Used Lucene 10.3.0
*/
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;

public class Indexer {

    /** One Cranfield document after parsing. */
    static class CranDoc {
        String id = "";
        StringBuilder title = new StringBuilder();
        StringBuilder author = new StringBuilder();
        StringBuilder text = new StringBuilder();
    }

    /** Analyzer factory: Searcher will reuse this, so both sides always match. */
    public static Analyzer makeAnalyzer(String name) {
        switch (name) {
            case "standard":
                // Note: StandardAnalyzer() with no arguments has NO stop words in Lucene 9.
                return new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
            case "english":
                return new EnglishAnalyzer(); // stop words + possessive removal + Porter stemming
            default:
                throw new IllegalArgumentException("Unknown analyzer: " + name);
        }
    }

    /** Splits cran.all.1400 into documents using the .I / .T / .A / .B / .W markers. */
    static List<CranDoc> parse(Path file) throws IOException {
        List<CranDoc> docs = new ArrayList<>();
        CranDoc cur = null;
        String section = "";
        for (String line : Files.readAllLines(file, StandardCharsets.ISO_8859_1)) {
            if (line.startsWith(".I ")) {
                cur = new CranDoc();
                cur.id = line.substring(3).trim();
                docs.add(cur);
                section = "";
            } else if (line.startsWith(".T") || line.startsWith(".A")
                    || line.startsWith(".B") || line.startsWith(".W")) {
                section = line.substring(0, 2);
            } else if (cur != null) {
                switch (section) {
                    case ".T": cur.title.append(line).append(' '); break;
                    case ".A": cur.author.append(line).append(' '); break;
                    case ".W": cur.text.append(line).append(' '); break;
                    default: break; // .B (bibliography) is ignored on purpose
                }
            }
        }
        return docs;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.err.println("Usage: Indexer <cran.all.1400> <indexDir> <standard|english>");
            return;
        }
        List<CranDoc> docs = parse(Paths.get(args[0]));
        Directory dir = FSDirectory.open(Paths.get(args[1]));
        IndexWriterConfig cfg = new IndexWriterConfig(makeAnalyzer(args[2]));
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE); // overwrite any old index

        try (IndexWriter writer = new IndexWriter(dir, cfg)) {
            for (CranDoc d : docs) {
                Document doc = new Document();
                doc.add(new StringField("id", d.id, Field.Store.YES));          // exact value, not analysed
                doc.add(new TextField("title", d.title.toString().trim(), Field.Store.YES));
                doc.add(new TextField("text", d.text.toString().trim(), Field.Store.YES));
                doc.add(new TextField("contents",                                // title + text together
                        d.title + " " + d.text, Field.Store.NO));
                writer.addDocument(doc);
            }
        }
        System.out.println("Indexed " + docs.size() + " documents into " + args[1]
                + " using analyzer '" + args[2] + "'");
    }
}
