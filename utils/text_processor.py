import nltk
from torch import nn


def tokenize(sentence, use_prototype=False, word2vec=None):
    if type(sentence) is not str:
        return []
    punctuations = ['.', '?', ',', '', '(', ')', '!', ":"]
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    if use_prototype:
        words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words if word not in punctuations]
    else:
        words = [word for word in words if word not in punctuations]
    if word2vec is None:
        return words
    return [word for word in words if word in word2vec]


def is_noun(word):
    word_tuple = nltk.pos_tag(word)
    if word_tuple[1] in {'NN', 'NNS', 'NNP', 'NNPS'}:
        return True
    return False


def is_predicate(word):
    word_tuple = nltk.pos_tag(word)
    if word_tuple[1] in {'VB', 'VBD'}:
        return True
    return False


def get_stem(word):
    return nltk.PorterStemmer().stem_word(word)


class Vocabulary(object):
    def __init__(self, word2ind, ind2word):
        super(Vocabulary, self).__init__()
        self.word2ind = word2ind
        self.ind2word = ind2word

    def stoi(self, word):
        if word in self.word2ind:
            return self.word2ind[word]
        else:
            return self.word2ind['<UNK>']

    def itos(self, index):
        if index in self.ind2word:
            return self.ind2word[index]
        else:
            return '<UNK>'


if __name__ == "__main__":
    result = tokenize("Whether to repeat the iterator for multiple epochs. Default: False.")
    print(result)
