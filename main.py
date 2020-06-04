import spacy
import numpy as np
from nltk.cluster import cosine_distance
import networkx as nx
from sympy.combinatorics.graycode import GrayCode
import math
from PSO import PSO
import requests

class Summarizer:

    def __init__(self,nlpmodel="es_core_news_md"):
        print("Loading spaCy model...")
        self.nlp = spacy.load(nlpmodel)
        self.processed_text = None
        self.simm_matrix = None
        print("Done!")

    def get_pairwise_similarity(self, vectortext1, vectortext2, stopwords=None):
        # Es una similaridad, no una distancia
        if stopwords is None:
            stopwords = []
        sent1 = [tok.text.lower() for tok in vectortext1]
        sent2 = [tok.text.lower() for tok in vectortext2]
        all_words = list(set(sent1 + sent2))
        all_words = [w for w in all_words if w not in stopwords]
        vector1 = np.zeros(len(all_words))
        vector2 = np.zeros(len(all_words))
        for w in sent1:
            vector1[all_words.index(w)] += 1
        for w in sent2:
            vector2[all_words.index(w)] += 1
        return 1 - cosine_distance(vector1, vector2)

    def do_process(self, text):
        self.processed_text = self.nlp(text)

    def build_simm_matrix(self, method = "word_freq",stopwords = []):
        pairwise_similarity = None
        sentences = [sent for sent in self.processed_text.sents]
        if method == "word_freq":
            def similarity_aux(a,b):
                return self.get_pairwise_similarity(a,b,stopwords)
            pairwise_similarity = similarity_aux
        if method == "word_embed":
            def similarity_spaCy(texta,textb):
                return texta.similarity(textb)
            pairwise_similarity = similarity_spaCy
        if method == "sentence_encoder":
            url = 'http://127.0.0.1:5000/results'
            r = requests.post(url, json={'text': self.processed_text.text})
            sentences = r.json()
            def similarity_tf(vec1,vec2):
                return 1 - cosine_distance(vec1,vec2)
            pairwise_similarity = similarity_tf

        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue
                similarity_matrix[idx1][idx2] = pairwise_similarity(sentences[idx1], sentences[idx2])

        self.simm_matrix = similarity_matrix

    def summarize(self,method = 'PageRank',coef=0.43):
        summary = []
        sentences = [sent for sent in self.processed_text.sents]
        if method == 'PageRank':
            ssgraph = nx.from_numpy_array(self.simm_matrix)
            scores = nx.pagerank(ssgraph)
            ranked_sentence = sorted(((scores[i], i) for i, _ in enumerate(sentences)), reverse=True)
            n_sentences_preserve = int(len(sentences) * coef)
            print(n_sentences_preserve)
            idx_preserve = [i for (_,i) in ranked_sentence[0:n_sentences_preserve]]
            summary = [(sentences[a], str(0+(a in idx_preserve))) for a in range(len(sentences))]
        elif method == "PSO":
            def within_diferentiation(simm_matrix,code):
                # REFACTOR
                indexes = [i for i in range(len(code)) if code[i] == '1']
                difs = np.zeros((len(code), len(code)))
                for i in indexes:
                    for j in indexes:
                        if i != j:
                            difs[i][j] = 1 - simm_matrix[i][j]
                            if math.isnan(difs[i][j]):
                                difs[i][j] = 1
                if np.count_nonzero(difs) == 0:
                    return 0
                return np.sum(difs) / np.count_nonzero(difs)
            def score(config):
                a = GrayCode(len(sentences))
                code = list(a.generate_gray())[config]
                sentence_list = [sentences[i].text for i in range(len(code)) if code[i] == '1']
                if sentence_list ==[]:
                    retention = 0
                else:
                    doc_propossed_summary =  self.nlp(" ".join(sentence_list))
                    retention = self.processed_text.similarity(doc_propossed_summary)
                diferentiation = within_diferentiation(self.simm_matrix,code)
                #print(diferentiation)
                return retention + diferentiation
            pso = PSO(n_pop=20, size =len(sentences))
            code, cum_scores = pso.findBest(score, epochs = 10)
            a = GrayCode(len(sentences))
            code = list(a.generate_gray())[code]
            summary = [(sentences[a].text , code[a]) for a in range(len(code))]
        return summary

if __name__ == '__main__':
    file = open('.\Texts\\testfile.txt', "r", encoding='utf8')
    text = file.read()
    summarizer = Summarizer()
    summarizer.do_process(text)
    summarizer.build_simm_matrix("word_embed")
    summary = summarizer.summarize()
    print(summary)
