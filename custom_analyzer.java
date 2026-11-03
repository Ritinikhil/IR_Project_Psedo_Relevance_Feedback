
import java.util.*;
import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;

/** Standard tokenizer + lowercase + stop words (English + query-noise words) + light KStem stemming. */
public class CustomAnalyzer extends Analyzer {
    static final CharArraySet STOP = new CharArraySet(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, true);
    static {
        STOP.addAll(Arrays.asList("paper", "papers", "discuss", "discussed", "describe",
                "information", "problem", "problems", "results", "investigation", "work", "relevant"));
    }

    @Override
    protected TokenStreamComponents createComponents(String field) {
        Tokenizer src = new StandardTokenizer();
        TokenStream ts = new LowerCaseFilter(src);
        ts = new StopFilter(ts, STOP);
        ts = new KStemFilter(ts);
        return new TokenStreamComponents(src, ts);
    }
}
