
from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

__all__ = [
    "clean_text", "preprocess_dataframe", "segment_vietnamese", "normalize_label",
    "make_training_dataframe", "EMOJI_MAP", "TEENCODE",
]


# Emoji are mapped to Vietnamese descriptions rather than assigned final sentiment labels.
_EMOJI_GROUPS = {
    # ---- positive ----
    "cười": "😂🤣😆😹",
    "vui": "😀😃😄😁😊🙂☺🙃😺😸🤗",
    "yêu": "😍🥰😘😗😙😚😻😽❤💕💖💗💓💞💘💝💙💚💛🧡💜🖤🤍🤎♥🫶",
    "tuyệt vời": "🤩👏🙌💯🔥⭐🌟🏆",
    "tốt": "👍👌🤝✌💪🆗✅",
    "ngầu": "😎",
    "chúc mừng": "🎉🥳🎊🎂",
    "cảm ơn": "🙏",
    "ngon": "😋",
    "đùa": "😉😜😝😛🤪",
    "xúc động": "🥹",
    # ---- negative ----
    "tức giận": "😡😠🤬👿😤💢",
    "khóc": "😭😢😿",
    "buồn": "😞😔☹🙁😟😕😩😫😣😖😪",
    "tội nghiệp": "🥺",
    "đau lòng": "💔",
    "tệ": "👎",
    "chửi": "🖕",
    "ghê tởm": "🤮🤢",
    "rác": "💩",
    "chán": "😒🙄😑😴🥱🤦",
    "sợ": "😱😨😰😧😦",
    # ---- neutral / ambiguous ----
    "ngạc nhiên": "😮😲😯🤯😳",
    "suy nghĩ": "🤔🧐",
    "hề": "🤡",
    "mỉa mai": "😏",
    "bình thường": "😐😶",
    "không biết": "🤷",
    "ngượng": "😅😬🙈",
    "lo lắng": "😥😓",
    "chết cười": "💀☠",  # Online slang can use skull icons to express laughter.
    "thư giãn": "😌",
    "hoa hồng": "🌹",
    "ôm": "🫂",
    "đồng ý": "🫡",
}

# The lookup table assigns one description to each recognized emoji.
EMOJI_MAP = {
    char: meaning
    for meaning, symbols in _EMOJI_GROUPS.items()
    for char in symbols
    if char not in "\ufe0f\ufe0e\u200d"
}

# Common online abbreviations are expanded only when they form a complete token.
_TEENCODE_GROUPS = {
    "không": "ko k kh khong khg hok hong hông hem k0 kô khum",
    "không phải": "kp kphai kphải",
    "không biết": "kbiet kbiết",
    "không có": "kco",
    "chưa": "chx chua chưaaa",
    "sản phẩm": "sp",
    "được": "dc đc dk đk dzc duoc",
    "với": "vs",
    "gì": "j",
    "vậy": "z zậy zay",
    "vui": "zui",
    "vô": "zô",
    "rồi": "r rùi ròi roài",
    "quá": "wa wá qá",
    "cũng": "cx cg cug cũg",
    "nhưng": "nhg nhưg",
    "nhưng mà": "nma nhma",
    "mình": "mk mik mjnh mih",
    "mọi người": "mn mng",
    "người": "ng ngừoi",
    "người ta": "ngta",
    "người yêu": "ny ngiu",
    "bình thường": "bt bthg bth",
    "biết": "bik bít bit",
    "thích": "thik thix thick",
    "yêu": "iu",
    "cảm ơn": "tks thks thanks thank thankiu camon",
    "xin lỗi": "sr sry sorry",
    "bình luận": "cmt cmts comment",
    "video": "vid vdeo vdo",
    "anh em": "ae",
    "bao nhiêu": "bn bnh",
    "trả lời": "tl trl",
    "điện thoại": "đt dt dthoai",
    "hôm nay": "hnay hnai",
    "hôm qua": "hqua",
    "trước": "trc",
    "bây giờ": "bh bgio",
    "nữa": "nx",
    "mới": "ms",
    "làm": "lm",
    "luôn": "lun",
    "liên quan": "lq",
    "ví dụ": "vd",
    "như thế nào": "ntn",
    "nói chuyện": "nch",
    "nhắn tin": "nt",
    "sao": "s",
    "tại sao": "tsao",
    "đang": "dg đg đag",
    "ok": "oke okê okie okay okey ôk ôkê",
    "làm ơn": "pls plz",
    "kết bạn": "kb",
    "dễ thương": "dthw dthuong",
    "xinh": "xink",
    "nhiều": "nhìu nhiu",
    "bé": "pé",
    "bạn": "pạn",
    "trai": "zai",
    # Profanity is kept (it carries sentiment, e.g. "hay vãi" / "dở đm"),
    # but spelling variants are merged into one token.
    "đm": "dm dmm đmm đcm dcm",
    "vãi": "vl vcl vkl",
}
TEENCODE = {
    abbr: full
    for full, abbrs in _TEENCODE_GROUPS.items()
    for abbr in abbrs.split()
}



# Compiled expressions recognize URLs, timestamps, emails, punctuation, and emoticons.
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\ufe0e\ufe0f]")
# Real tags start with a letter or "/", so "<3" (a heart) is NOT treated as a tag.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|vn|be|ly|tv|io|me|info)\b(?:/\S*)?",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\S+")
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # "3:45", "1:02:30"
_PUNCT_RE = re.compile(r"[^\w\s]|_")
_SENTENCE_PUNCT_RE = re.compile(r"[^\w\s.!?]|_")
_REPEATED_PUNCT_RE = re.compile(r"([.!?])\1+")
_CPP_RE = re.compile(r"(?<!\w)c\+\+(?!\w)")
_CSHARP_RE = re.compile(r"(?<!\w)c#(?!\w)")
_SPACES_RE = re.compile(r"\s+")

# Order matters: longer / more specific patterns first. Text is already lowercase.
_B = r"(?<!\w)"  # "not preceded by a word character"
_E = r"(?!\w)"   # "not followed by a word character"
_EMOTICONS = [
    (r"</3", "đau lòng"),
    (r"<3+", "yêu"),
    (r":'\(+", "khóc"),
    (r">:\(+|:@", "tức giận"),
    (r"-_-|:\||:-\|", "chán"),
    (rf"{_B}t_t{_E}|{_B}t\.t{_E}|;_;|:<", "buồn"),
    (r":-?\(+|=\(+", "buồn"),
    (r":-?\)+|=\)+|\(:|\^_*\^", "vui"),
    (rf":-?d+{_E}|=d+{_E}|{_B}xd+{_E}", "cười"),
    (rf":v+{_E}|:3+{_E}|:-?p+{_E}|;-?\)+", "đùa"),
    (rf":-?o{_E}|{_B}o_o{_E}", "ngạc nhiên"),
]
_EMOTICON_RES = [(re.compile(p), rf" {w} ") for p, w in _EMOTICONS]

# Laughter ("haha", "hihi", "kkkk", "hjhj") -> one token. "huhu" means crying.
_LAUGH_RE = re.compile(r"k{2,}|a?(?:ha|he|hi|hj){2,}h?|h{3,}")
_CRY_RE = re.compile(r"a?(?:hu){2,}h?")
# 3+ repeated LETTERS ("hayyyy", "quááááá") -> one letter. Digits are untouched.
_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")

_SYMBOL_CATEGORIES = {"So", "Sk"}  # emoji, pictographs, modifier symbols





# The optional segmenter is loaded only when word segmentation is requested.
_word_tokenize = None


def convert_emoticons(text: str) -> str:
    """Replace text emoticons such as :)) and <3 with Vietnamese descriptions."""
    for pattern, replacement in _EMOTICON_RES:
        text = pattern.sub(replacement, text)
    return text


def convert_emoji(text: str) -> str:
    """Convert recognized emoji into Vietnamese terms and remove unsupported symbols."""
    output = []
    for character in text:
        word = EMOJI_MAP.get(character)
        if word is not None:
            output.append(f" {word} ")
        elif unicodedata.category(character) in _SYMBOL_CATEGORIES:
            output.append(" ")
        else:
            output.append(character)
    return "".join(output)


def normalize_tokens(text: str) -> str:
    """Normalize slang, laughter and excessively repeated letters token by token."""
    normalized = []
    for token in text.split():
        if _LAUGH_RE.fullmatch(token):
            normalized.append("haha")
        elif _CRY_RE.fullmatch(token):
            normalized.append("khóc")
        else:
            token = _REPEAT_RE.sub(r"\1", token)
            normalized.append(TEENCODE.get(token, token))
    return " ".join(normalized)


def segment_vietnamese(text: str) -> str:
    """Join Vietnamese compound words with underscores using underthesea."""
    global _word_tokenize
    if _word_tokenize is None:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError("Word segmentation requires: pip install underthesea") from exc
        _word_tokenize = word_tokenize
    return _word_tokenize(text, format="text")


def clean_text(text, segment: bool = False, keep_sentence_punctuation: bool = False) -> str:
    """Clean one comment for the SVM pipeline or backend inference."""
    # Missing comments produce an empty string, including floating-point NaN.
    if text is None or (isinstance(text, float) and text != text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities and normalize Vietnamese Unicode characters.
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = _ZERO_WIDTH_RE.sub("", text)

    # Remove email addresses before URLs so email usernames are not left behind.
    text = _HTML_TAG_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    text = text.lower()

    # Emoji and emoticons must be translated before removing punctuation.
    text = convert_emoticons(text)
    text = convert_emoji(text)

    # Preserve programming-language tokens such as C++ and C#.
    text = _CPP_RE.sub(" cplusplus ", text)
    text = _CSHARP_RE.sub(" csharp ", text)

    # Punctuation is optional because some models use sentence boundaries.
    if keep_sentence_punctuation:
        text = _SENTENCE_PUNCT_RE.sub(" ", text)
        text = _REPEATED_PUNCT_RE.sub(r"\1", text)
        text = re.sub(r"\s*([.!?])\s*", r" \1 ", text)
    else:
        text = _PUNCT_RE.sub(" ", text)

    # Teencode is expanded before optional word segmentation.
    text = normalize_tokens(text)
    text = _SPACES_RE.sub(" ", text).strip()
    if segment and text:
        text = segment_vietnamese(text)
    return _SPACES_RE.sub(" ", text).strip()


# All labels are standardized to the three English classes used by the models.
_LABEL_MAP = {
    "negative": "negative", "neg": "negative", "0": "negative",
    "tiêu cực": "negative", "tieu cuc": "negative",
    "neutral": "neutral", "neu": "neutral", "1": "neutral",
    "trung lập": "neutral", "trung lap": "neutral",
    "positive": "positive", "pos": "positive", "2": "positive",
    "tích cực": "positive", "tich cuc": "positive",
}


def normalize_label(label) -> str:
    """Normalize a known sentiment label; preserve missing labels as empty strings."""
    import pandas as pd

    if pd.isna(label) or str(label).strip().lower() in {"", "nan", "none", "null"}:
        return ""
    value = str(label).strip().lower()
    try:
        numeric = float(value)
        if numeric.is_integer():
            value = str(int(numeric))
    except ValueError:
        pass
    if value not in _LABEL_MAP:
        raise ValueError(f"Unknown sentiment label: {label!r}; expected positive, negative or neutral")
    return _LABEL_MAP[value]


def preprocess_dataframe(
    df,
    text_col: str = "text",
    out_col: str = "clean_text",
    segment: bool = False,
    drop_empty: bool = True,
    drop_duplicates: bool = True,
    min_words: int = 3,
    drop_short: bool = False,
    add_segmented: bool = False,
    keep_sentence_punctuation: bool = False,
):
    """Prepare a labeled or unlabeled DataFrame while retaining original source data."""
    import pandas as pd

    if text_col not in df.columns:
        raise ValueError(f"Missing text column {text_col!r}. Available: {list(df.columns)}")
    if min_words < 1:
        raise ValueError("min_words must be at least 1")

    # Work on a copy so the original raw DataFrame remains unchanged.
    result = df.copy()

    # Confidence estimates are not verified labels and are excluded from the dataset.
    result = result.drop(columns=["label_confidence"], errors="ignore")

    # Each comment is cleaned with the same function used during inference.
    result[out_col] = result[text_col].map(
        lambda value: clean_text(
            value, segment=segment, keep_sentence_punctuation=keep_sentence_punctuation
        )
    )
    result["word_count"] = result[out_col].map(lambda value: len(value.split()))

    # A review flag helps identify short or nonlinguistic text without discarding
    # genuine short opinions such as "hay quá" or "dở quá" by default.
    def quality_flag(value: str) -> str:
        if not value:
            return "empty"
        if not any(character.isalpha() for character in value):
            return "non_linguistic"
        if len(value.split()) < min_words:
            return "short_text"
        return "ok"

    result["quality_flag"] = result[out_col].map(quality_flag)

    # Unlabeled comments stay unlabeled; no automatic label is called verified.
    if "label" in result.columns:
        result["label"] = result["label"].map(normalize_label)
    else:
        result["label"] = ""

    # A review status is set only when no status already exists in the dataset.
    if "review_status" not in result.columns:
        result["review_status"] = "needs_human_review"
    else:
        result["review_status"] = (
            result["review_status"].fillna("needs_human_review")
            .astype(str).str.strip().str.lower()
        )
        result.loc[result["review_status"] == "", "review_status"] = "needs_human_review"

    # Word segmentation is stored separately so basic TF-IDF can use clean_text.
    if add_segmented:
        result["segmented_text"] = result[out_col].map(
            lambda value: segment_vietnamese(value) if value else ""
        )

    # Empty and nonlinguistic comments are not useful for text classification.
    if drop_empty:
        result = result[result["quality_flag"].isin(["ok", "short_text"])].copy()
    if drop_short:
        result = result[result["quality_flag"] == "ok"].copy()

    if drop_duplicates:
        # Detect contradictory existing labels before collapsing identical texts.
        labeled = result.loc[result["label"] != "", [out_col, "label"]]
        conflicts = labeled.groupby(out_col)["label"].nunique()
        if (conflicts > 1).any():
            examples = conflicts.index[conflicts > 1].tolist()[:3]
            raise ValueError(f"Conflicting labels for duplicate cleaned texts: {examples}")

        # Prefer already verified examples when several rows clean to the same text.
        result["_row_order"] = range(len(result))
        result["_label_rank"] = (
            (result["label"] != "").astype(int) +
            (result["review_status"] == "verified").astype(int) * 2
        )
        result = (
            result.sort_values("_label_rank", ascending=False, kind="stable")
            .drop_duplicates(subset=[out_col], keep="first")
            .sort_values("_row_order", kind="stable")
            .drop(columns=["_row_order", "_label_rank"])
        )

    return result.reset_index(drop=True)


def make_training_dataframe(df, cleaned_col: str = "clean_text"):
    """Export exactly text,label for rows explicitly verified by a human."""
    required = {"label", "review_status", cleaned_col, "quality_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training export requires columns: {sorted(missing)}")
    mask = (
        df["review_status"].astype(str).str.lower().eq("verified")
        & df["label"].isin(["positive", "negative", "neutral"])
        & df["quality_flag"].isin(["ok", "short_text"])
    )
    training = df.loc[mask, [cleaned_col, "label"]].rename(columns={cleaned_col: "text"})
    training = training.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if training.empty:
        raise ValueError("No human-verified labeled comments are available for training")
    return training


def add_video_categories(df, metadata_file: str):
    """Add per-video categories from a separate video_id,category CSV."""
    import pandas as pd

    if "video_id" not in df.columns:
        raise ValueError("The raw CSV needs video_id for category enrichment")
    metadata = pd.read_csv(metadata_file, encoding="utf-8-sig", dtype={"video_id": str})
    if not {"video_id", "category"}.issubset(metadata.columns):
        raise ValueError("Metadata CSV must contain video_id and category columns")
    if metadata["video_id"].duplicated().any():
        raise ValueError("Each video_id must appear once in the metadata CSV")
    if "category" in df.columns:
        df = df.drop(columns=["category"])
    return df.merge(metadata[["video_id", "category"]], on="video_id", how="left", validate="many_to_one")


def _demo():
    """Print examples of original and cleaned Vietnamese comments."""
    examples = [
        "ko hay lắm 😡 https://youtu.be/abc",
        "Mình khum thích sp này",
        "chx xem 😭",
        "Học C++ và C# rất thú vị!!!",
        "Bài này hay quá 😂😂😂",
    ]
    for example in examples:
        print(f"RAW:   {example}\nCLEAN: {clean_text(example)}\n")


def _main():
    """Read a CSV, preprocess comments, and optionally export verified training data."""
    import pandas as pd

    parser = argparse.ArgumentParser(description="Prepare Vietnamese YouTube comment data")
    parser.add_argument("--input", help="Raw or human-reviewed CSV file")
    parser.add_argument("--output", help="Processed CSV file")
    parser.add_argument("--text-col", default="text", help="Column containing original text")
    parser.add_argument("--out-col", default="clean_text", help="Column for cleaned comments")
    parser.add_argument("--segment", action="store_true", help="Segment the clean_text column")
    parser.add_argument("--add-segmented", action="store_true", help="Add a separate segmented_text column")
    parser.add_argument("--keep-sentence-punctuation", action="store_true", help="Retain . ! ?")
    parser.add_argument("--min-words", type=int, default=3, help="Word count below which a comment is flagged")
    parser.add_argument("--drop-short", action="store_true", help="Remove comments below --min-words")
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty and numeric-only comments for review")
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate cleaned text rows")
    parser.add_argument("--metadata", help="CSV with video_id,category columns")
    parser.add_argument("--training-output", help="Export verified rows using only text,label")
    args = parser.parse_args()
    if not args.input:
        _demo()
        return

    # Read UTF-8 with BOM compatibility and retain video IDs as strings.
    raw = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"video_id": str, "comment_id": str})
    if args.metadata:
        raw = add_video_categories(raw, args.metadata)

    cleaned = preprocess_dataframe(
        raw, text_col=args.text_col, out_col=args.out_col, segment=args.segment,
        drop_empty=not args.keep_empty, drop_duplicates=not args.keep_duplicates,
        min_words=args.min_words, drop_short=args.drop_short,
        add_segmented=args.add_segmented,
        keep_sentence_punctuation=args.keep_sentence_punctuation,
    )
    output = Path(args.output or str(Path(args.input).with_name(Path(args.input).stem + "_clean.csv")))
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Original rows: {len(raw)}; processed rows: {len(cleaned)}; output: {output}")
    print("Quality flags:", cleaned["quality_flag"].value_counts().to_dict())
    print("Labels:", cleaned["label"].replace("", "unlabeled").value_counts().to_dict())

    if args.training_output:
        training = make_training_dataframe(cleaned, cleaned_col=args.out_col)
        target = Path(args.training_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        training.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"Human-verified training rows: {len(training)}; output: {target}")


if __name__ == "__main__":
    _main()
