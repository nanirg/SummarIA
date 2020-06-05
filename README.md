# SummarIA

SummarIA was born from the necessity of extracting key information from bureaucratic documents. Documents in legal contexts tend to have a lot of paragraphs with callbacks to other cases, to provide jurisdiction, which are often related o similar among themselves. It is key to assert which sentences provide distinct and new information, in order to build an ordered subselection of sentences that highlight the key ideas of a text.

## Method

So far, sentences are being ranked modeling them as an undirected weighted graph, and then applying PageRank. Similarity between two sentences is computed using pretrained [embeddings](https://spacy.io/models/es#es_core_news_md).


