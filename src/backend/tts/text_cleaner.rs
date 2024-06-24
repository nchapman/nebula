use std::collections::HashMap;

pub struct TextCleaner {
    word_index_dictionary: HashMap<char, usize>,
}

impl TextCleaner {
    pub fn new() -> Self {
        let _pad = '$';
        let _punctuation = ";:,.!?¡¿—…\"«»“” ";
        let _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";
        
        // Export all symbols
        let mut symbols: Vec<char> = vec![_pad];
        symbols.extend(_punctuation.chars());
        symbols.extend(_letters.chars());
        symbols.extend(_letters_ipa.chars());

        let mut word_index_dictionary = HashMap::new();
        for (i, symbol) in symbols.into_iter().enumerate() {
            word_index_dictionary.insert(symbol, i);
        }

        TextCleaner { word_index_dictionary }
    }

    pub fn clean_text(&self, text: &str) -> Vec<i64> {
        text.chars()
            .filter_map(|char| self.word_index_dictionary.get(&char).cloned())
            .map(|elem| elem as i64)
            .collect()
    }
}

fn main() {
    let cleaner = TextCleaner::new();
    println!("{}", cleaner.word_index_dictionary.len());

    let text = "Some text";
    let cleaned_text = cleaner.clean_text(text);
    println!("{:?}", cleaned_text);
}
