---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="UgG1BxLH-RuW">

#Skip-gram Word2Vec

In this tutorial, we delve into the skip-gram neural network
architecture used in Word2Vec. The purpose of this tutorial is to bypass
the typical introductory and abstract explanations about Word2Vec and
instead focus on the intricacies of the skip-gram neural network model.

## Readings

Here are the resources I used to build this notebook. I suggest reading
these either beforehand or while you're working on this material.

-   A really good [conceptual
    overview](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
    of Word2Vec from Chris McCormick
-   [First Word2Vec paper](https://arxiv.org/pdf/1301.3781.pdf) from
    Mikolov et al.

## The Model

The skip-gram neural network model, in its fundamental form, is
surprisingly straightforward. However, as we delve into the details,
various adjustments and enhancements can complicate the explanation.

To begin, let's gain a high-level understanding of our direction.
Word2Vec employs a technique commonly utilized in machine learning. We
train a simple neural network with a hidden layer to accomplish a
specific task. However, we won't actually utilize this neural network
for the task it was trained on! Instead, our objective is to grasp the
weights of the hidden layer itself, as these weights serve as the "word
vectors" we aim to learn.

## The Fake Task

Now, let's delve into the "fake" task that we will design the neural
network to accomplish. Later on, we will explore how this task
indirectly provides us with the desired word vectors.

The objective of the neural network is as follows: given a specific word
positioned in the middle of a sentence (referred to as the input word),
we examine the surrounding words and randomly select one. The network's
role is to provide us with the probability of each word in our
vocabulary being the chosen "nearby word."

`When we mention "nearby," there exists a parameter known as the "window size" within the algorithm. Typically, a window size of 5 is used, encompassing 5 preceding words and 5 succeeding words (10 in total).`

The output probabilities will indicate the likelihood of finding each
vocabulary word in the vicinity of our input word. For instance, if we
feed the trained network the input word "coffee," the probabilities will
be higher for words like "mug" and "brew" compared to unrelated words
such as "elephant" and "umbrella."

To train the neural network for this task, we will provide it with word
pairs extracted from our training documents. It's okay if you still
think this is magic, stick with me till the end and you'll understand
how are related words able to cluster together in a high dimensional
space.

</div>

<div class="cell markdown" id="7clIt9mmE--V">

## Loading Data

The below command loads the data for you -

1.  Downloads the [text8 dataset](http://mattmahoney.net/dc/text8.zip);
    a file of cleaned up *Wikipedia article text* from Matt Mahoney.
2.  Unzips the data and places that data in the `data` folder in the
    home directory.

Execute the below command to load the text8 file into your data
directory: `data/text8`.

</div>

<div class="cell code" execution_count="1"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="_W0nlXHnEl78" outputId="91bccfb0-40a7-4bea-dc2d-571288a685f3">

``` python
!wget http://mattmahoney.net/dc/text8.zip && mkdir data && unzip text8.zip -d data
```

<div class="output stream stdout">

    --2023-06-21 06:28:39--  http://mattmahoney.net/dc/text8.zip
    Resolving mattmahoney.net (mattmahoney.net)... 34.198.1.81
    Connecting to mattmahoney.net (mattmahoney.net)|34.198.1.81|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 31344016 (30M) [application/zip]
    Saving to: ‘text8.zip’

    text8.zip           100%[===================>]  29.89M  --.-KB/s    in 0.1s    

    2023-06-21 06:28:39 (216 MB/s) - ‘text8.zip’ saved [31344016/31344016]

    Archive:  text8.zip
      inflating: data/text8              

</div>

</div>

<div class="cell markdown" id="gUEqCYlnSc-O">

# Let's take a look at the data

</div>

<div class="cell code" execution_count="2"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="AgJdy537SrMe" outputId="4cc4d13b-3cf7-42bb-fa47-37725ab89783">

``` python
# Open the file, and read its content into 'text'
with open('data/text8') as f:
    text = f.read()

print(text[:50])
```

<div class="output stream stdout">

     anarchism originated as a term of abuse first use

</div>

</div>

<div class="cell markdown" id="fvqDS5yITkp8">

## ## Inspecting Word Counts<a name="word_counts"></a>

</div>

<div class="cell code" execution_count="39" id="B28OzBVvWKqn">

``` python
from collections import Counter

# Convert text into a list of words
text_words = text.split()

# Use the Counter to count the number of occurences for each word
word_counts = Counter(text_words)

# Sorting the Counter Dict based on the count values (In descending order)
sorted_vocab = sorted(word_counts.items(), key=lambda pair: pair[1], reverse=True)
```

</div>

<div class="cell code" execution_count="41"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="T3QVB3oSWp66" outputId="5141bf76-3855-4351-e3a6-d5da6d567b98">

``` python
import numpy as np

# Convert the dictionary into two numpy arrays so we can do math on it easily.
words = np.asarray(list(word_counts.keys()))
word_counts = np.asarray(list(word_counts.values()))

# Total words in the training set.
# Make sure to sum with int64, otherwise it will overflow!
total_words = np.sum(word_counts, dtype=np.int64)

print('Number of words in vocabulary: {:,}\n'.format(len(words)))
print('Total word occurrences in Wikipedia: {:,}\n'.format(total_words))
```

<div class="output stream stdout">

    Number of words in vocabulary: 253,854

    Total word occurrences in Wikipedia: 17,005,207

</div>

</div>

<div class="cell markdown" id="-CS6gGJftvit">

Just out of curiosity, here are the most frequent and least frequent
words.

</div>

<div class="cell code" execution_count="45"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="QSxys8JZtwof" outputId="fd6ae7c0-e59f-445b-9d13-2ba48a81a9ab">

``` python
print('The 10 most frequent words:\n')
print('  --Count--    --Word--')

# For the first ten word counts...
for item in sorted_vocab[:10]:
    # Print the count with commas, and pad it to 12 characters.
    print('{:>12,}     {:}'.format(item[1], item[0]))
```

<div class="output stream stdout">

    The 10 most frequest words:

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

</div>

</div>

<div class="cell code" execution_count="48"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="F_TwqyjfxzNx" outputId="e91eb02c-a607-4391-8e44-43783bc714a9">

``` python
print('The 10 least frequent words:\n')
print('  --Count--    --Word--')

# For the first ten word counts...
for item in sorted_vocab[:-10:-1]:
    # Print the count with commas, and pad it to 12 characters.
    print('{:>12,}     {:}'.format(item[1], item[0]))
```

<div class="output stream stdout">

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

</div>

</div>

<div class="cell code" id="ZXcHni0NVHyQ">

``` python
import re
from collections import Counter


def preprocess(text):

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
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
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab
```

</div>
