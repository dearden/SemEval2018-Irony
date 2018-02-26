import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from textstat.textstat import textstat
from nltk.corpus import stopwords
import os
import re


# Gets the 1000 most popular tokens.
def count_all_tokens(corpus, num=1000, exclude=[]):
    counts = dict()
    for line in corpus:
        for token in line:
            if token in counts and token not in exclude:
                counts[token] += 1
            else:
                counts[token] = 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_counts[0:num]]


# creates a bag of words feature table
def bag_of_words(train, test):
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    tr_toks = [tk(tweet) for tweet in train]                            # creates token representation of corpus
    te_toks = [tk(tweet) for tweet in test]                             # creates token representation of corpus
    all_words = count_all_tokens(tr_toks)                               # 1000 most popular words
    matrix = create_token_count(te_toks, all_words)                     # Create the matrix
    return np.asarray(matrix)


# creates a bag of words feature table
def baseline(train, test):
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    tr_toks = [tk(tweet) for tweet in train]                            # creates token representation of corpus
    te_toks = [tk(tweet) for tweet in test]                             # creates token representation of corpus
    all_words = count_all_tokens(tr_toks, num=-1)                       # All words
    matrix = create_token_count(te_toks, all_words)                     # Create the matrix
    return np.asarray(matrix)


# creates bag of part-of-speech feature table
def bag_of_pos(train, test):
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    tr_toks = [tk(tweet) for tweet in train]
    te_toks = [tk(tweet) for tweet in test]

    # get POS tags for each tweet
    tr_pos = []
    for line in tr_toks:
        tr_pos.append([tag for (word, tag) in nltk.pos_tag(line)])

    all_tags = count_all_tokens(tr_pos)

    # get POS tags for each tweet
    te_pos = []
    for line in te_toks:
        te_pos.append([tag for (word, tag) in nltk.pos_tag(line)])

    matrix = create_token_count(te_pos, all_tags)
    return np.asarray(matrix)


# create bag-of-semantic-tag feature table
def bag_of_sem(traindir, testdir):
    tr_sem = count_tags_from_file(traindir)
    te_sem = count_tags_from_file(testdir)
    all_tags = count_all_tokens(tr_sem)             # count frequencies of tags
    matrix = create_token_count(te_sem, all_tags)   # create matrix
    return np.asarray(matrix)


# creates list of tags in each tweet
def count_tags_from_file(rootdir):
    pattern = re.compile('(\S+)\s+(\S+)')
    tags = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            entry = []
            # loop through frequency list files
            with open(os.path.join(subdir, file), 'r') as file:
                for line in file:
                    if line.startswith("TOTAL"):
                        continue
                    text = line.strip()
                    m = pattern.match(text)
                    tag = m.group(1)            # get tag
                    count = int(m.group(2))     # get count

                    for i in range(count):
                        entry.append(tag)       # add count*tags to list
            tags.append(entry)
    return tags


# count appearances of tokens
def feature_count(tr_toks, te_toks):
    all_tokens = count_all_tokens(tr_toks, num=50)
    matrix = create_token_count(te_toks, all_tokens)
    return np.asarray(matrix)


# count appearances of tokens (1 if present, 0 if not)
def binary_feature_count(tr_toks, te_toks):
    all_tokens = count_all_tokens(tr_toks, num=50)
    matrix = create_binary_token_count(te_toks, all_tokens)
    return np.asarray(matrix)


# Count appearances of words in wordlist
def word_list_features(corpus, wordlist):
    out = []
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    c = [tk(entry) for entry in corpus]
    # loop through all entries in corpus
    for toks in c:
        count = 0
        # loop through each word in word list
        for word in wordlist:
            # check each token against current word
            for tk in toks:
                if tk == word:
                    count += 1 / len(toks)
        out.append(count)
    return np.array([[x] for x in out])


# Count appearances of pos tags based on regular expression
def pos_list(reg, corpus):
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    toks = [tk(tweet) for tweet in corpus]

    # get POS tags for each tweet
    pos = []
    for line in toks:
        pos.append([tag for (word, tag) in nltk.pos_tag(line)])

    out = tag_list_features(reg, pos)
    return out


# count tags based on regular expression
def tag_list_features(reg, tags):
    out = []
    for line in tags:
        found = 0
        for tag in line:
            if re.match(reg, tag):          # check if tag correct
                found += 1                  # count tag
        if len(line) > 0:
            out.append(found / len(line))
        else:
            out.append(0)
    return np.array([[x] for x in out])


# create feature matrix of token frequency given list of tokens and corpus
def create_token_count(corpus, tokens):
    matrix = [[token for token in tokens]]
    for tweet in corpus:
        line = list()
        for token in tokens:
            if len(tweet) > 0:
                line.append(float(tweet.count(token)) / len(tweet))     # count instances of token
            else:
                line.append(0)
        matrix.append(line)

    return matrix


# create matrix of token appearances (1 if present, else 0) given list of tokens and corpus
def create_binary_token_count(corpus, tokens):
    matrix = [[token for token in tokens]]
    for tweet in corpus:
        line = list()
        for token in tokens:
            if len(tweet) > 0:
                x = float(tweet.count(token))
                if float(tweet.count(token) > 0):
                    line.append(1)
                else:
                    line.append(0)
            else:
                line.append(0)
        matrix.append(line)

    return matrix


# Get useful text information and put into a table
def text_stats(corpus):
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    toks = [tk(entry) for entry in corpus]

    # get function words
    with open('function_words.txt') as file:
        funcs = file.read().split(',')
        funcs = [f.strip() for f in funcs]

    amb = ambiguity(toks)       # calculate ambiguity

    matrix = [["Chars/Word", "Lexical Diversity", "Lexical Density", "Function Words", "Syllables", "ARI"]]
    for tokens, sentence in zip(toks, corpus):
        unique = set(tokens)
        avchar = 0
        lexdiv = 0
        lexden = 0
        nfunc = 0
        numsyl = 0
        ari = 0

        if len(sentence) > 1:
            lexdiv = len(unique) / len(tokens)                                  # Lexical Diversity
            lexden = len([x for x in tokens if x not in funcs]) / len(tokens)   # Lexical Density
            numsyl = textstat.syllable_count(sentence) / len(tokens) / 10       # Number of syllables in text
            # may be a bit dodgy without punctuation
            ari = abs(textstat.automated_readability_index(sentence)) / 14      # Automated Readability index
        for t in tokens:
            avchar += len(t) / len(tokens) / len(sentence)                      # Average num chars
            if t in funcs:
                nfunc += 1 / len(tokens)                                        # Number of function words

        matrix.append([avchar, lexdiv, lexden, nfunc, numsyl, ari])

    matrix = [m + [a] for m, a in zip(matrix, amb)]

    return np.array(matrix)


# need a better way of normalising Ambiguity.
def ambiguity(tokens):
    amb = ["Ambiguity"]
    for line in tokens:
        if len(line) == 0:
            amb.append(float(0))
        else:
            line_amb = np.zeros(len(line))
            for word, i in zip(line, range(len(line))):
                if len(wn.synsets(word)) > 1:               # length of array of meanings (num meanings)
                    line_amb[i] = 1
                else:
                    line_amb[i] = 0
            a = np.sum(line_amb) / len(line_amb)       # Normalise mean number of meanings
            if np.isnan(a):
                a = 0
            amb.append(float(a))

    return amb


# create feature table of character trigrams
def char_trigrams(train, test):
    trigrams = []
    tricount = dict()
    for line in train:
        # get the char trigrams for the current line
        n = list(ngrams(line, 3, pad_left=True, pad_right=True, left_pad_symbol='^', right_pad_symbol='$'))
        trigrams.append([''.join(x) for x in n])
    feats = count_all_tokens(trigrams)

    trigrams = []
    for line in test:
        # get the char trigrams for the current line
        n = list(ngrams(line, 3, pad_left=True, pad_right=True, left_pad_symbol='^', right_pad_symbol='$'))
        trigrams.append([''.join(x) for x in n])
    out = create_token_count(trigrams, feats)
    return np.array(out)


# create feature table of word bigrams
def word_bigrams(train, test):
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    tr_toks = [tk(tweet) for tweet in train]  # creates token representation of corpus
    te_toks = [tk(tweet) for tweet in test]  # creates token representation of corpus
    bigrams = []
    for line in tr_toks:
        # get the word bigrams for the current line
        b = list(ngrams(line, 2, pad_left=True, pad_right=True, left_pad_symbol='^', right_pad_symbol='$'))
        bigrams.append([' '.join(x) for x in b])
    feats = count_all_tokens(bigrams)  # 1000 most popular bigrams

    bigrams = []
    for line in te_toks:
        # get the word bigrams for the current line
        b = list(ngrams(line, 2, pad_left=True, pad_right=True, left_pad_symbol='^', right_pad_symbol='$'))
        bigrams.append([' '.join(x) for x in b])

    out = create_token_count(bigrams, feats)  # Create the matrix
    return np.array(out)



if __name__ == "__main__":
    print(bag_of_words(["Hello this is a", "Big Corpus full", "of nice big words", "that may contain nuts",
                  "Gosh I love a nice corpus"]))
    print(bag_of_pos(["Hello this is a", "Big Corpus full", "of nice big words", "that may contain nuts",
                  "Gosh I love a nice corpus"]))
    print(char_trigrams(["Hello this is a", "Big Corpus full", "of nice big words", "that may contain nuts",
                  "Gosh I love a nice corpus"]))
    print(word_bigrams(["Hello this is a", "Big Corpus full", "of nice big words", "that may contain nuts",
                         "Gosh I love a nice corpus"]))
    print(text_stats(["this is a sentence", "wow another sentence wow wow", "yes yes yes yes"]))

