import nltk


def tokenize(sentence, word2vec):
    punctuations = ['.', '?', ',', '', '(', ')']
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    words = [word for word in words if word not in punctuations]
    return [word for word in words if word in word2vec]
