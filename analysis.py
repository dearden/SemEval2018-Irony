# Prints the 50 tweets with the highest values of certain features.
def get_top_50_examples(corpus, labels, pred, feature):
    strings = [str(x) + '\t' + str(y) + '\t' + z for x, y, z in zip(pred, labels, corpus)]
    sort_feat = [[x, y] for x, y in zip(feature, strings)]
    sort_feat.sort(key=lambda x:x[0], reverse=True)
    for line in sort_feat[:50]:
        print(line[1])


# Prints tweets containing certain key words along with their mean sentiment.
def look_for_keywords(corpus, labels, pred, sentiments, words):
    strings = [str(x) + '\t' + str(y) + '\t' + str(z) + '\t' + a for x, y, z, a in zip(pred, labels, sentiments, corpus)]
    out = []
    for string in strings:
        for word in words:
            if word in string.lower():
                out.append(string)
    for line in out:
        print(line)
