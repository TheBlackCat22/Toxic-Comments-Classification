import contractions
import re
import string

# lower Case


def make_lower(text):
    return text.lower()

# expand contractions


def expand_contractions(text):
    return contractions.fix(text)

# remove URLs


def remove_url(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

# remove HTML tags


def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

# remove Non-ASCI


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

# remove special charecters


def remove_special_characters(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# remove punctuations


def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# remove numbers


def remove_num(text):
    return text.translate(str.maketrans('', '', string.digits))


def text_cleaner(text):

    text = make_lower(text)
    text = expand_contractions(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = remove_punct(text)
    text = remove_num(text)

    return text
