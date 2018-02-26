from nltk.tokenize import TweetTokenizer
import numpy as np
import string
import re
import csv
from text_process import bag_of_words
from text_process import bag_of_pos
from text_process import bag_of_sem
from text_process import feature_count
from text_process import binary_feature_count
from text_process import word_list_features
from text_process import text_stats
from text_process import char_trigrams
from text_process import word_bigrams
from text_process import count_tags_from_file
from text_process import tag_list_features
from text_process import pos_list
from text_process import baseline
from visualise_sentiment import VaderBreakdown
from vaderSentiment.vaderSentiment import NEGATE
from text_process import MeanEmbeddingVectoriser
from nltk.corpus import stopwords


# Preprocess the corpus, extracting non-text features
def preprocess(corpus):
    emoji = [[] for i in range(0, len(corpus))]                         # Emojis in tweet
    emoticon = [[] for i in range(0, len(corpus))]                      # Emoticons in tweet
    hashtag = [[] for i in range(0, len(corpus))]                       # Hashtags in tweet
    link = ["Num Links"] + [0 for i in range(0, len(corpus))]           # Number of links
    punct = [[] for i in range(0, len(corpus))]                         # Punctuation in tweet
    users = ["Num Mentions"] + [0 for i in range(0, len(corpus))]       # Number of mentions
    nhash = ["Num Hashtags"] + [0 for i in range(0, len(corpus))]       # Number of Hashtags
    npunc = ["Punct Count"] + [0 for i in range(0, len(corpus))]        # Amount of Punctuation
    clean = ["" for i in range(0, len(corpus))]                         # Cleaned text
    repeats = count_dupes(corpus)                                       # Number of duplicated characters
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    reg_emoji = ':[a-z_-]*:'
    reg_punct = '[!"#\$%&\'\(\)\*\+,-\.\\/:;<=>\?@\[\]\^_`\{\|\}~]+'
    reg_emoticon = "([:;8=])[Oo\-'`]?([sSDpPoO|$*@\\/\[\]{}()])"
    caps = ["Capitals"]
    n_chars = ["Tweet Length"]

    i = 0
    for tweet in corpus:
        n_chars.append(len(tweet) / 140)
        caps.append(count_caps(tweet))          # amount of text capitalised
        emoji[i] = re.findall(reg_emoji, tweet)
        for e in emoji[i]:                      # Remove emojis
            tweet = tweet.replace(e, '')
        tkd = tokenizer(tweet)
        clean_tweet = ""
        j = i + 1
        for t in tkd:
            if t[0] == '#':                     # Separate and count hashtags
                hashtag[i].append(t)
                nhash[j] += 1
            elif t.startswith("http"):          # Remove and count links
                link[j] += 1
            elif t[0] == '@':                   # Remove and count mentions
                users[j] += 1
            elif re.fullmatch(reg_punct, t):        # Separate punctuation and emoticons
                if re.fullmatch(reg_emoticon, t):
                    emoticon[i].append(t)
                else:
                    punct[i].append(t)
                    npunc[j] += 1
            # elif t in stopwords.words('english'):
            #     continue
            else:
                clean_tweet += t + " "          # Add only text to corpus
        # Normalise between 1 and 0
        if len(tweet) > 0:
            link[j] = link[j] / len(tkd)
            users[j] = users[j] / len(tkd)
            nhash[j] = nhash[j] / len(tkd)
            npunc[j] = npunc[j] / len(tweet)

        clean[i] = clean_tweet
        i += 1

    return clean, emoji, hashtag, link, punct, users, nhash, repeats, caps, n_chars, emoticon, npunc


# Count amount of a tweet that is capitalised
def count_caps(sentence):
    count = 0
    for c in sentence:
        if c.isupper():
            count += 1 / len(sentence)
    return count


# Count number of duplicate characters in a tweet.
def count_dupes(corpus):
    out = ["Num Dupe Chars"]
    for line in corpus:
        repeats = 0
        r = 0
        last = ''
        for char in line:
            if char == last:
                repeats += 1
            else:
                repeats = 0
            if repeats > 1:
                r += (1 / len(line))
            last = char
        out.append(r)
    return out


# Calculate the sentiment values
def get_sentiments(corpus, emos):
    anal = VaderBreakdown()
    sent = [["Positive Sentiment Score", "Negative Sentiment Score", "Std Sent",
             "Mean Sent", "Sent Range", "Average Change", "Number of PN Shifts"]]

    for line in corpus:
        entry = list()
        entry.append(anal.polarity_scores(line)["pos"])     # Positive Sentiment Score
        entry.append(anal.polarity_scores(line)["neg"])     # Negative Sentiment Score

        words, breakdown = anal.sentiment_breakdown(line)   # Create sentiment breakdown of text
        breakdown = np.asarray(breakdown).astype(float)

        if len(breakdown) > 0:
            entry.append(breakdown.std() / 5)               # Sentiment standard deviation
            entry.append(breakdown.mean() / 5)              # Sentiment mean
            entry.append((max(breakdown) - min(breakdown)) / 10)    # Sentiment range

            num_shifts = 0
            av_change = 0
            for i in range(len(breakdown)):
                if i != 0:
                    change = breakdown[i] - breakdown[i-1]
                    av_change += change / len(breakdown)    # Average change
                    if breakdown[i] * breakdown[i-1] < 0:
                        num_shifts += 1 / len(breakdown)    # Number of shifts

            entry.append(av_change)
            entry.append(num_shifts)
        else:
            entry = entry + [0, 0, 0, 0, 0]
        sent.append(entry)

    e_sent = emoji_sentiment(emos)

    sent = np.array(sent)
    sent = np.concatenate([sent, e_sent], axis=1)           # Attach sentiment and emoji sentiment tables

    return np.array(sent)


def emoji_sentiment(emos):
    with open('Emoji_Sentiment_Data_v1.0.csv', 'r') as file:
        doc = list(csv.reader(file))[1:]
        sents = [[x[7], (int(x[6]) / int(x[2])) - (int(x[4]) / int(x[2])), int(x[6]), int(x[4])] for x in doc]

    esent = [["Avg Emoji Sentiment", "Max Emoji Sentiment", "Min Emoji Sentiment",
              "Pos Emoji", "Neg Emoji"]]
    # loop through emojis from our data
    for line in emos:
        # loop through each emoji
        avg = 0.0
        max = -2.0
        min = 2.0
        pos = 0
        neg = 0
        for e in line:
            name = e.strip(":").replace("_", " ").upper()   # convert format of name from :an_emoji: to AN EMOJI
            # loop through emoji sentiment list
            for x in sents:
                if name == x[0]:                # if names match
                    avg += x[1] / len(line)     # add to the average emoji sentiment
                    if x[1] > max:
                        max = x[1]              # set max emoji sentiment
                    if x[1] < min:
                        min = x[1]              # set min emoji sentiment
                    if x[2] > x[3]:
                        pos += 1 / len(line)    # number of positive emoji
                    elif x[3] > x[2]:
                        neg += 1 / len(line)    # number of negative emoji
                    break
        if min > 1 or max < -1:
            max = 0
            min = 0

        diff = max - min
        esent.append([avg, max, min, pos, neg])

    return np.array(esent)


# find contrasting emotional words based on Semantic Tags
def contrasting_emotion(tags):
    reg_pos = 'E[\d\.]+[\+]+'
    reg_neg = 'E[\d\.]+[-]+'
    out = [['Contrasting Emotion']]
    # check every line
    for line in tags:
        pos = False
        neg = False
        # check every tag for positive or negative emotional meaning
        for tag in line:
            if re.match(reg_pos, tag):
                pos = True
            elif re.match(reg_neg, tag):
                neg = True
        # If there has been both a positive and negative tag, there is a contrast.
        if pos is True and neg is True:
            out.append([1])
        else:
            out.append([0])
    return np.array(out)


# Given test and train corpus, creates feature table
class IronyFeatures(object):
    def __init__(self, train, *test):
        self.train = train
        self.feature_table = None
        self.all_feats = dict()
        self.feat_names = []

        # Sort out whether to use the training corpus as the test corpus
        if not test:
            self.test = [line for line in train]
            TEST_DIR = './train_sem'
        else:
            self.test = [line for line in test[0]]
            TEST_DIR = './test_sem'

        # preprocess test and training corpus
        tr_corpus, tr_emoji, tr_hashtag, tr_link, tr_punct, tr_mentions, tr_nhash, \
        tr_repeats, tr_caps, tr_n_chars, tr_emoticon, tr_npunc = preprocess(self.train)

        te_corpus, te_emoji, te_hashtag, te_link, te_punct, te_mentions, te_nhash, \
        te_repeats, te_caps, te_n_chars, te_emoticon, te_npunc = preprocess(self.test)

        # Comment out feature groups not to be included in feature set.

        # base_feats = baseline(tr_corpus, te_corpus)                 # Baseline (leave as only one uncommented)
        # self.all_feats["base"] = base_feats[1:].astype(float)
        # self.feat_names.extend(base_feats[0])

        # word_feats = bag_of_words(tr_corpus, te_corpus)                 # Bag of Words
        # self.all_feats["words"] = word_feats[1:].astype(float)
        # self.feat_names.extend(word_feats[0])

        # pos_feats = bag_of_pos(tr_corpus, te_corpus)                    # Bag of Part-of-Speech
        # self.all_feats["pos"] = pos_feats[1:].astype(float)
        # self.feat_names.extend(pos_feats[0])
        #
        # sem_feats = bag_of_sem('./train_sem', TEST_DIR)             # Bag of USAS Semantic Tags
        # self.all_feats["sem"] = sem_feats[1:].astype(float)
        # self.feat_names.extend(sem_feats[0])

        # contrast = contrasting_emotion(count_tags_from_file(TEST_DIR))  # Emotional Contrast
        # self.all_feats["contrast"] = contrast[1:].astype(float)
        # self.feat_names.extend(contrast[0])

        # tri_feats = char_trigrams(tr_corpus, te_corpus)                 # Character Trigrams
        # self.all_feats["trigrams"] = tri_feats[1:].astype(float)
        # self.feat_names.extend(tri_feats[0])
        #
        # bi_feats = word_bigrams(tr_corpus, te_corpus)                   # Word Bigrams
        # self.all_feats["bigrams"] = bi_feats[1:].astype(float)
        # self.feat_names.extend(bi_feats[0])

        emo_feats = binary_feature_count(tr_emoji, te_emoji)                   # Emoji Count
        self.all_feats["emoji"] = emo_feats[1:].astype(float)
        self.feat_names.extend(emo_feats[0])

        emoticon_feats = binary_feature_count(tr_emoticon, te_emoticon)        # Emoticon Count
        self.all_feats["emoticon"] = emoticon_feats[1:].astype(float)
        self.feat_names.extend(emoticon_feats[0])

        punct_feats = binary_feature_count(tr_punct, te_punct)                 # Punctuation Count
        self.all_feats["punctuation"] = punct_feats[1:].astype(float)
        self.feat_names.extend(punct_feats[0])

        # hashtag_feats = feature_count(tr_hashtag, te_hashtag)           # Hashtag Count
        # self.all_feats["hashtags"] = hashtag_feats[1:].astype(float)
        # self.feat_names.extend(hashtag_feats[0])

        senti_feats = get_sentiments(te_corpus, te_emoji)               # Sentiment Scores
        self.all_feats["sentiments"] = senti_feats[1:].astype(float)
        self.feat_names.extend(senti_feats[0])

        negation_feats = word_list_features(te_corpus, NEGATE)          # Negation word list
        self.all_feats["negations"] = negation_feats[:].astype(float)
        self.feat_names.append("negations")

        # emotional_words = tag_list_features('E[\d\.]+[-\+]*', count_tags_from_file(TEST_DIR))
        # self.all_feats["Emotional Words"] = emotional_words[:].astype(float)
        # self.feat_names.append("Emotional Words")

        # sup_com_pos = pos_list('(JJR|JJS|RBS|RBR)', te_corpus)
        # self.all_feats["Superlative and Comparative"] = sup_com_pos[:].astype(float)
        # self.feat_names.append("Superlative and Comparative")

        lex_feats = text_stats(te_corpus)                               # Lexical Features
        self.all_feats["lexical"] = lex_feats[1:].astype(float)
        self.feat_names.extend(lex_feats[0])

        other_feats = np.asarray([[te_link[i],
                                   te_mentions[i],
                                   te_nhash[i],
                                   te_repeats[i],
                                   te_caps[i],
                                   te_n_chars[i],
                                   te_npunc[i]] for i in range(0, len(te_link))])  # Other features

        self.all_feats["other"] = other_feats[1:].astype(float)
        self.feat_names.extend(other_feats[0])

        # Make the feature table
        self.feature_table = np.concatenate(list(self.all_feats.values()), axis=1)
