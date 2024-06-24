use fancy_regex::Regex;

struct MacIntyreContractions {
    contractions2: Vec<Regex>,
    contractions3: Vec<Regex>,
    contractions4: Vec<Regex>,
}


impl MacIntyreContractions {
    fn new() -> Self {
        let contractions2 = vec![
            Regex::new(r"(?i)\b(can)(?#X)(not)\b").unwrap(),
            Regex::new(r"(?i)\b(d)(?#X)('ye)\b").unwrap(),
            Regex::new(r"(?i)\b(gim)(?#X)(me)\b").unwrap(),
            Regex::new(r"(?i)\b(gon)(?#X)(na)\b").unwrap(),
            Regex::new(r"(?i)\b(got)(?#X)(ta)\b").unwrap(),
            Regex::new(r"(?i)\b(lem)(?#X)(me)\b").unwrap(),
            Regex::new(r"(?i)\b(more)(?#X)('n)\b").unwrap(),
            Regex::new(r"(?i)\b(wan)(?#X)(na)(?=\s)").unwrap(),
        ];
        let contractions3 = vec![
            Regex::new(r"(?i) ('t)(?#X)(is)\b").unwrap(),
            Regex::new(r"(?i) ('t)(?#X)(was)\b").unwrap(),
        ];
        let contractions4 = vec![
            Regex::new(r"(?i)\b(whad)(dd)(ya)\b").unwrap(),
            Regex::new(r"(?i)\b(wha)(t)(cha)\b").unwrap(),
        ];

        MacIntyreContractions {
            contractions2,
            contractions3,
            contractions4,
        }
    }
}


pub struct TreebankWordTokenizer {
    starting_quotes: Vec<(Regex, &'static str)>,
    ending_quotes: Vec<(Regex, &'static str)>,
    punctuation: Vec<(Regex, &'static str)>,
    parens_brackets: (Regex, &'static str),
    convert_parentheses: Vec<(Regex, &'static str)>,
    double_dashes: (Regex, &'static str),
    contractions: MacIntyreContractions,
}


impl TreebankWordTokenizer {
    pub fn new() -> Self {
        let starting_quotes = vec![
            (Regex::new(r"([«“‘„]|[`]+)").unwrap(), " $1 "),
            (Regex::new(r#"^""#).unwrap(), "``"),
            (Regex::new(r"(``)").unwrap(), " $1 "),
            (
                Regex::new(r#"([ \(\[{<])("|\'{2})"#).unwrap(),
                r"$1 `` ",
            ),
            (
                Regex::new(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b").unwrap(),
                r"$1 $2",
            ),
        ];

        let ending_quotes = vec![
            (Regex::new(r"([»”’])").unwrap(), " $1 "),
            (Regex::new(r#"''"#).unwrap(), " '' "),
            (Regex::new(r#"""#).unwrap(), " '' "),
            (
                Regex::new(r"([^' ])('[sS]|'[mM]|'[dD]|') ").unwrap(),
                r"$1 $2 ",
            ),
            (
                Regex::new(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ").unwrap(),
                r"$1 $2 ",
            ),
        ];

        let punctuation = vec![
            (
                Regex::new(r#"([^\.])(\.)([\]\)}>"' "»”’ ]*)\s*$"#).unwrap(),
                r"$1 $2 $3 ",
            ),
            (Regex::new(r"([:,])([^\d])").unwrap(), r" $1 $2"),
            (Regex::new(r"([:,])$").unwrap(), r" $1 "),
            (
                Regex::new(r"\.{2,}").unwrap(),
                r" $0 ",
            ),
            (Regex::new(r"[;@#$%&]").unwrap(), r" $0 "),
            (
                Regex::new(r#"([^\.])(\.)([\]\)}>"'])\s*$"#).unwrap(),
                r"$1 $2$3 ",
            ),
            (Regex::new(r"[?!]").unwrap(), r" $0 "),
            (Regex::new(r#"([^'])' "#).unwrap(), r"$1 ' "),
            (
                Regex::new(r"[*]").unwrap(),
                r" $0 ",
            ),
        ];

        let parens_brackets = (Regex::new(r"[\]\[\(\)\{\}\<\>]").unwrap(), r" $0 ");

        let convert_parentheses = vec![
            (Regex::new(r"\(").unwrap(), "-LRB-"),
            (Regex::new(r"\)").unwrap(), "-RRB-"),
            (Regex::new(r"\[").unwrap(), "-LSB-"),
            (Regex::new(r"\]").unwrap(), "-RSB-"),
            (Regex::new(r"\{").unwrap(), "-LCB-"),
            (Regex::new(r"\}").unwrap(), "-RCB-"),
        ];

        let double_dashes = (Regex::new(r"--").unwrap(), r" -- ");

        let contractions = MacIntyreContractions::new();

        TreebankWordTokenizer {
            starting_quotes,
            ending_quotes,
            punctuation,
            parens_brackets,
            convert_parentheses,
            double_dashes,
            contractions,
        }
    }

    pub fn tokenize(&self, mut text: String, convert_parentheses: bool) -> Vec<String> {
        for (regexp, substitution) in &self.starting_quotes {
            text = regexp.replace_all(&text, (*substitution).to_string()).into_owned();
        }

        for (regexp, substitution) in &self.punctuation {
            text = regexp.replace_all(&text, (*substitution).to_string()).into_owned();
        }

        let (regexp, substitution) = &self.parens_brackets;
        text = regexp.replace_all(&text, (*substitution).to_string()).into_owned();

        if convert_parentheses {
            for (regexp, substitution) in &self.convert_parentheses {
                text = regexp.replace_all(&text, (*substitution).to_string()).into_owned();
            }
        }

        let (regexp, substitution) = &self.double_dashes;
        text = regexp.replace_all(&text, (*substitution).to_string()).into_owned();

        for (regexp, substitution) in &self.ending_quotes {
            text = regexp.replace_all(&text, (*substitution).to_string()).into_owned();
        }

        for regexp in &self.contractions.contractions2 {
            text = regexp.replace_all(&text, " $1 $2 ").into_owned();
        }

        for regexp in &self.contractions.contractions3 {
            text = regexp.replace_all(&text, " $1 $2 ").into_owned();
        }

        text.split_whitespace().map(String::from).collect()
    }
}
