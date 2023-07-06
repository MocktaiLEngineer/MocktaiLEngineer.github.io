---
title: Skipgram Word2Vec
mathjax: true
date: 2021-03-27
---

In this tutorial, we delve into the skip-gram neural network architecture used in Word2Vec. The purpose of this tutorial is to bypass the typical introductory and abstract explanations about Word2Vec and instead focus on the intricacies of the skip-gram neural network model.

## Readings

Here are the resources I used to build this notebook. I suggest reading these either beforehand or while you're working on this material.

* A really good [conceptual overview](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) of Word2Vec from Chris McCormick
* [First Word2Vec paper](https://arxiv.org/pdf/1301.3781.pdf) from Mikolov et al.


## The Model
The skip-gram neural network model, in its fundamental form, is surprisingly straightforward. However, as we delve into the details, various adjustments and enhancements can complicate the explanation.

To begin, let's gain a high-level understanding of our direction. Word2Vec employs a technique commonly utilized in machine learning. We train a simple neural network with a hidden layer to accomplish a specific task. However, we won't actually utilize this neural network for the task it was trained on! Instead, our objective is to grasp the weights of the hidden layer itself, as these weights serve as the "word vectors" we aim to learn.

## The Fake Task

Now, let's delve into the "fake" task that we will design the neural network to accomplish. Later on, we will explore how this task indirectly provides us with the desired word vectors.

The objective of the neural network is as follows: given a specific word positioned in the middle of a sentence (referred to as the input word), we examine the surrounding words and randomly select one. The network's role is to provide us with the probability of each word in our vocabulary being the chosen "nearby word."

```When we mention "nearby," there exists a parameter known as the "window size" within the algorithm. Typically, a window size of 5 is used, encompassing 5 preceding words and 5 succeeding words (10 in total).```

The output probabilities will indicate the likelihood of finding each vocabulary word in the vicinity of our input word. For instance, if we feed the trained network the input word "coffee," the probabilities will be higher for words like "mug" and "brew" compared to unrelated words such as "elephant" and "umbrella."

To train the neural network for this task, we will provide it with word pairs extracted from our training documents. It's okay if you still think this is magic, stick with me till the end and you'll understand how are related words able to cluster together in a high dimensional space.



## Loading Data

The below command loads the data for you -

1. Downloads the [text8 dataset](http://mattmahoney.net/dc/text8.zip); a file of cleaned up *Wikipedia article text* from Matt Mahoney.
2. Unzips the data and places that data in the `data` folder in the home directory.

Execute the below command to load the text8 file into your data directory: `data/text8`.


```python
!wget http://mattmahoney.net/dc/text8.zip && mkdir data && unzip text8.zip -d data
```

    --2023-06-22 16:27:13--  http://mattmahoney.net/dc/text8.zip
    Resolving mattmahoney.net (mattmahoney.net)... 34.198.1.81
    Connecting to mattmahoney.net (mattmahoney.net)|34.198.1.81|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 31344016 (30M) [application/zip]
    Saving to: ‘text8.zip’
    
    text8.zip           100%[===================>]  29.89M  7.63MB/s    in 3.9s    
    
    2023-06-22 16:27:18 (7.63 MB/s) - ‘text8.zip’ saved [31344016/31344016]
    
    Archive:  text8.zip
      inflating: data/text8              


# Let's take a look at the data




```python
# Open the file, and read its content into 'text'
with open('data/text8') as f:
    text = f.read()

print(text[:50])
```

     anarchism originated as a term of abuse first use


## Inspecting Word Counts<a name="word_counts"></a>
-------------------------



```python
from collections import Counter
import numpy as np

# Convert text into a list of words
text_words = text.split()

# Use the Counter to count the number of occurences for each word
word_counts = Counter(text_words)

# Sorting the Counter Dict based on the count values (In descending order)
sorted_vocab = sorted(word_counts.items(), key=lambda pair: pair[1], reverse=True)

# Convert the dictionary into two numpy arrays so we can do math on it easily.
words = np.asarray(list(word_counts.keys()))
word_counts = np.asarray(list(word_counts.values()))

# Total words in the training set.
# Make sure to sum with int64, otherwise it will overflow!
total_words = np.sum(word_counts, dtype=np.int64)

print('Number of words in vocabulary: {:,}\n'.format(len(words)))
print('Total word occurrences in text8 dataset: {:,}\n'.format(total_words))
```

    Number of words in vocabulary: 253,854
    
    Total word occurrences in text8 dataset: 17,005,207
    


Just out of curiosity, here are the most frequent and least frequent words.


```python
print('The 10 most frequent words:\n')
print('  --Count--    --Word--')

# For the first ten word counts...
for item in sorted_vocab[:10]:
    # Print the count with commas, and pad it to 12 characters.
    print('{:>12,}     {:}'.format(item[1], item[0]))

```

    The 10 most frequent words:
    
      --Count--    --Word--
       1,061,396     the
         593,677     of
         416,629     and
         411,764     one
         372,201     in
         325,873     a
         316,376     to
         264,975     zero
         250,430     nine
         192,644     two



```python
print('The 10 least frequent words:\n')
print('  --Count--    --Word--')

# For the first ten word counts...
for item in sorted_vocab[:-10:-1]:
    # Print the count with commas, and pad it to 12 characters.
    print('{:>12,}     {:}'.format(item[1], item[0]))

```

    The 10 least frequent words:
    
      --Count--    --Word--
               1     exortation
               1     fretensis
               1     metzuda
               1     metzada
               1     erniest
               1     workmans
               1     englander
               1     mikhailgorbachev
               1     gorbacheva


## Pre-processing

Let us pre-process the text to make it easier for us, and for the model to behave as expected.

Here's what I have in mind -

1.   Remove less frequent words, to reduce noise in the dataset and to improveme the quality of the word representations.
2.   Convert any punctuations into tokens, so for example, a comma is replaced as a "<COMMA>" - this will essentially help in other NLP problems.




```python
from collections import Counter

def preprocess(text: str) -> list:

    # Convert your text to lowercase
    text = text.lower()

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace(".", " <PERIOD> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace('"', " <QUOTATION_MARK> ")
    text = text.replace(";", " <SEMICOLON> ")
    text = text.replace("!", " <EXCLAMATION_MARK> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace("(", " <LEFT_PAREN> ")
    text = text.replace(")", " <RIGHT_PAREN> ")
    text = text.replace("--", " <HYPHENS> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace(":", " <COLON> ")

    # Split the text into a list of words
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    processed_words = [word for word in words if word_counts[word] > 5]

    return processed_words

words = preprocess(text)

print(words[:50])
```

    ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against', 'early', 'working', 'class', 'radicals', 'including', 'the', 'diggers', 'of', 'the', 'english', 'revolution', 'and', 'the', 'sans', 'culottes', 'of', 'the', 'french', 'revolution', 'whilst', 'the', 'term', 'is', 'still', 'used', 'in', 'a', 'pejorative', 'way', 'to', 'describe', 'any', 'act', 'that', 'used', 'violent', 'means', 'to', 'destroy', 'the']


# Building lookup tables


```python
def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, word_to_int, int_to_word
    """
    word_counts = Counter(words)

    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    # create int_to_vocab dictionaries
    int_to_word = {i: word for i, word in enumerate(sorted_vocab)}
    word_to_int = {word: i for i, word in int_to_word.items()}

    return word_to_int, int_to_word

word_to_int, int_to_word = create_lookup_tables(words)

encoded_words = [word_to_int[word] for word in words]

print(encoded_words[:30])
```

    [5233, 3080, 11, 5, 194, 1, 3133, 45, 58, 155, 127, 741, 476, 10571, 133, 0, 27349, 1, 0, 102, 854, 2, 0, 15067, 58112, 1, 0, 150, 854, 3580]



```python
print('  --Word--    --Int--')

# Iterate over the items of the word_to_int dictionary
for word, id in list(word_to_int.items())[:10]:
    print(f'  {word:<10} {id}')
```

      --Word--    --Int--
      the        0
      of         1
      and        2
      one        3
      in         4
      a          5
      to         6
      zero       7
      nine       8
      two        9



```python
print('  --Int--    --Word--')

# Iterate over the items of the int_to_word dictionary
for id, word in list(int_to_word.items())[:10]:
    print(f'  {id:<10} {word}')
```

      --Int--    --Word--
      0          the
      1          of
      2          and
      3          one
      4          in
      5          a
      6          to
      7          zero
      8          nine
      9          two


## Subsampling

Words that show up often such as "the", "of", and "for" don't provide much context to the nearby words. If we discard some of them, we can remove some of the noise from our data and in return get faster training and better representations. This process is called subsampling by Mikolov. For each word $w_i$ in the training set, we'll discard it with probability given by

$$ P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} $$

where $t$ is a threshold parameter and $f(w_i)$ is the frequency of word $w_i$ in the total dataset.

$$ P(0) = 1 - \sqrt{\frac{1*10^{-5}}{1*10^6/16*10^6}} = 0.98735 $$


```python
from collections import Counter
import random
import numpy as np

# Set the threshold for subsampling
threshold = 1e-5

# Count the occurrences of each word in the encoded_words list
word_counts = Counter(encoded_words)

# Calculate the total count of words in the encoded_words list
total_count = len(encoded_words)

# Calculate the frequencies of each word
freqs = {word: count/total_count for word, count in word_counts.items()}

# Calculate the probability of dropping each word based on its frequency
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}

# Discard some frequent words based on the subsampling equation
# Create a new list of words for training, keeping only the words that were not dropped
train_words = [word for word in encoded_words if random.random() < (1 - p_drop[word])]

# Print the first 30 words in the train_words list
print(train_words[:30])
```

    [5233, 194, 45, 58, 10571, 27349, 15067, 58112, 194, 190, 58, 10712, 1324, 708, 7088, 1052, 320, 44611, 2877, 5233, 1134, 2621, 8983, 279, 4147, 59, 6437, 5233, 1137, 4860]


# Making batches

Now that our data is in good shape, we need to get it into the proper form to pass it into our network. With the skip-gram architecture, for each word in the text, we want to define a surrounding _context_ and grab all the words in a window around that word, with size $C$.

From [Mikolov et al.](https://arxiv.org/pdf/1301.3781.pdf):

"Since the more distant words are usually less related to the current word than those close to it, we give less weight to the distant words by sampling less from those words in our training examples... If we choose $C = 5$, for each training word we will select randomly a number $R$ in range $[ 1: C ]$, and then use $R$ words from history and $R$ words from the future of the current word as correct labels."

> **Exercise:** Implement a function `get_target` that receives a list of words, an index, and a window size, then returns a list of words in the window around the index. Make sure to use the algorithm described above, where you chose a random number of words to from the window.

Say, we have an input and we're interested in the idx=2 token, `741`:
```
[5233, 58, 741, 10571, 27349, 0, 15067, 58112, 3580, 58, 10712]
```

For `R=2`, `get_target` should return a list of four values:
```
[5233, 58, 10571, 27349]
```


```python
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]

    return list(target_words)
```


```python
# run this cell multiple times to check for random window selection
int_text = [i for i in range(10)]
print('Input: ', int_text)

idx=5 # word index of interest

target = get_target(int_text, idx=idx, window_size=5)
print('Target: ', target)  # you should get some indices around the idx
```

    Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Target:  [2, 3, 4, 6, 7, 8]


### Generating Batches

Here's a generator function that returns batches of input and target data for our model, using the `get_target` function from above. The idea is that it grabs `batch_size` words from a words list. Then for each of those batches, it gets the target words in a window.


```python
def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words)//batch_size

    # only full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

```

---
## Validation

Here, I'm creating a function that will help us observe our model as it learns. We're going to choose a few common words and few uncommon words. Then, we'll print out the closest words to them using the cosine similarity:

$$
\mathrm{similarity} = \cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}
$$


We can encode the validation words as vectors $\vec{a}$ using the embedding table, then calculate the similarity with each word vector $\vec{b}$ in the embedding table. With the similarities, we can print out the validation words and words in our embedding table semantically similar to those words. It's a nice way to check that our embedding table is grouping together words with similar semantic meanings.


```python
def cosine_similarity(embedding, valid_size=8, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """

    # Here we're calculating the cosine similarity between some random words and
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.

    # sim = (a . b) / |a||b|

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)

    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes

    return valid_examples, similarities
```

## SkipGram model

Define and train the SkipGram model.
> You'll need to define an [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) and a final, softmax output layer.

An Embedding layer takes in a number of inputs, importantly:
* **num_embeddings** – the size of the dictionary of embeddings, or how many rows you'll want in the embedding weight matrix
* **embedding_dim** – the size of each embedding vector; the embedding dimension


```python
import torch
from torch import nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)

        return log_ps
```

### Training

Below is our training loop, and I recommend that you train on GPU, if available.

**Note that, because we applied a softmax function to our model output, we are using NLLLoss** as opposed to cross entropy. This is because Softmax  in combination with NLLLoss = CrossEntropy loss .


```python
# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim=300 # you can change, if you want

model = SkipGram(len(word_to_int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 2500
steps = 0
epochs = 5

# train for some number of epochs
for e in range(epochs):

    # get input and target batches
    for inputs, targets in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
        inputs, targets = inputs.to(device), targets.to(device)

        log_ps = model(inputs)
        loss = criterion(log_ps, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            # getting examples and similarities
            valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
            _, closest_idxs = valid_similarities.topk(6) # topk highest similarities

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for i, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_word[idx.item()] for idx in closest_idxs[i]][1:]
                print(int_to_word[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...")
```

    up | nablus, amelia, enchantments, civilize, adding
    during | poinsot, jannah, shambhala, deva, bezprym
    such | portrayed, megawatts, westminsters, to, ayers
    about | innings, frigate, persist, fide, alternation
    consists | hires, wheel, rida, sesame, exotics
    ice | kaiserliche, melatonin, reed, domine, psych
    frac | homicidal, filmi, emrys, wilbert, byelorussian
    police | billiard, lavinia, zagros, possums, tab
    ...
    seven | nine, poorest, five, claudia, collectivisation
    often | musial, entails, orchha, confusing, zimbardo
    eight | leitch, aho, krishnan, redesigned, zero
    by | micros, critica, balk, healthier, pre
    recorded | paintings, mano, thrown, decks, iib
    freedom | vorbis, marginalize, exoteric, earn, blair
    mainly | verso, singapore, annex, buonarroti, hikari
    liberal | sturmgewehr, hugging, lunas, openly, jng
    ...
    b | antonio, mathematician, tat, nicky, shanghainese
    their | residence, lorne, liking, tgs, until
    its | stalled, nashwaak, socony, chabad, of
    than | triumphed, accurate, unstoppable, biopolymers, pressurized
    discovered | nighttime, voracious, ascendant, herakles, pi
    numerous | bribery, eddic, theobromine, mongo, cautiously
    pressure | danforth, extraction, masjid, kara, essequibo
    woman | wasn, roses, milah, lipstick, exports
    ...
    had | he, barely, railroaders, sale, addressed
    who | erudition, prevailed, aeschylus, pascual, elderly
    may | observant, duplication, individualist, acumen, recited
    they | stalk, unborn, protects, forceful, even
    discovered | pi, nighttime, endowed, voracious, authorship
    marriage | jondo, ultimate, attitudes, doves, caesarian
    applied | student, prism, stargate, papp, dai
    dr | gill, backpacker, socketed, mirroring, jun
    ...
    who | receive, their, erudition, humbert, adultery
    may | duplication, observant, not, myriad, provisions
    up | anything, to, before, voltages, your
    the | a, of, in, for, which
    brother | taisho, unison, isolates, odysseus, died
    institute | engineering, textbook, vayikra, cookbook, kangaroo
    joseph | insistence, martin, kohl, freethought, stacey
    centre | equal, erected, rancid, melchett, watt
    ...
    state | act, federal, states, guthrum, exercised
    that | the, be, this, to, himself
    no | illusions, feminine, xa, catechetical, replacement
    their | they, them, are, have, to
    rise | empire, pico, declan, anaxagoras, became
    experience | camphor, talkative, artemis, pantheists, akin
    arts | disciplines, art, education, intv, timelines
    something | anything, hopefully, hiker, furry, silly
    ...
    for | also, a, ml, more, and
    use | used, useful, vendors, standard, or
    see | history, article, list, of, links
    were | many, their, was, by, themselves
    orthodox | ecumenical, christian, churches, church, conservative
    pre | ingrained, expansion, handsets, mythologies, raid
    question | questions, answer, assumptions, what, prescribes
    road | amtrak, lanes, roads, highway, autopia
    ...
    it | is, be, to, come, but
    state | act, federal, legislature, government, missouri
    of | in, the, and, as, by
    also | other, with, as, are, for
    orthodox | church, ecumenical, christian, conservative, churches
    proposed | freie, subcommittee, kuiper, jlp, principles
    operating | hardware, user, unix, lotus, system
    resources | web, soils, arable, land, private
    ...
    as | is, or, used, the, of
    four | six, five, three, two, one
    i | we, t, me, you, my
    other | such, also, various, used, refer
    hit | hits, albums, pop, album, billboard
    woman | her, women, birth, male, husband
    question | questions, answer, whether, answers, answered
    engineering | technology, institute, disciplines, development, mathematics
    ...
    use | used, using, are, available, commonly
    called | is, which, referred, then, usually
    may | not, or, remaining, duplication, consent
    first | in, the, eight, of, two
    operating | unix, platforms, platform, interface, hardware
    proposed | suggested, agreed, agreement, freie, xlii
    additional | each, provision, fewer, esti, browser
    troops | forces, battle, war, army, allied
    ...

