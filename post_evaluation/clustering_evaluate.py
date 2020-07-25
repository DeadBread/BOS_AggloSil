from xlm.substs_loading import load_substs
from collections import defaultdict, Counter
from evaluatable import Evaluatable, GridSearch
from xlm.data_loading import load_data, load_target_words
from xlm.wsi import clusterize_search

import sys
import inspect
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import os
from pathlib import Path

from itertools import product
from joblib import Memory
import fire
import numpy as np
import pandas as pd
from pymorphy2 import MorphAnalyzer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

np.seterr(divide='ignore', invalid='ignore')

class Substs_loader:
    def __init__(self, data_name, lemmatizing_method, topk, max_examples=None, delete_word_parts=False, drop_duplicates=True):
        self.data_name = data_name
        self.lemmatizing_method = lemmatizing_method
        self.topk = topk
        self.max_examples = max_examples
        self.delete_word_parts = delete_word_parts
        self.drop_duplicates = drop_duplicates
        if lemmatizing_method is not None and lemmatizing_method!='none': 
            if 'ru' in data_name:
                self.analyzer = MorphAnalyzer()
            elif 'german' in data_name:
                self.analyzer = spacy.load("de_core_news_sm", disable=['ner', 'parser'])
            elif 'english' in data_name:
                self.analyzer = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        self.cache = {}

    def analyze(self, word):
        word = word.strip()
        if not word:
            return ['']
        if 'ru' in self.data_name:
            parsed = self.analyzer.parse(word)
            return sorted(list(set([i.normal_form for i in parsed])))
        else:
            spacyed = self.analyzer(word)
            lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
            return [lemma]

    def get_lemmas(self, word):
        if word not in self.cache:
            self.cache[word] = self.analyze(word)
        return self.cache[word]

    def get_single_lemma(self, word):
        return self.get_lemmas(word)[0]

    def preprocess_substitutes(self, x, exclude_lemmas=[], delete_word_parts=False):
        """
        1) leaves only topk substitutes without spaces inside
        2) applies lemmatization
        3) excludes unwanted lemmas (if any)
        4) returns string of space separated substitutes
        """
        if self.topk is not None:
            x = x[:self.topk]

        if delete_word_parts:
            words = [word.strip() for prob, word in x if word.strip() and ' ' not in word.strip() and word[0] == ' ']
        else:
            words = [word.strip() for prob, word in x if word.strip() and ' ' not in word.strip()]

        if self.lemmatizing_method == 'single':
            words = [self.get_single_lemma(word.strip()) for word in words]
        elif self.lemmatizing_method == 'all':
            words = [' '.join(self.get_lemmas(word.strip())) for word in words]
        else:
            assert self.lemmatizing_method == 'none', "unrecognized lemmatization method %s" % self.lemmatizing_method

        if exclude_lemmas:
            words = [s for s in words if s not in exclude_lemmas]
        return ' '.join(words)

    def get_subs(self, path1, path2):
        """
        loads subs from path1, path2 and applies preprocessing
        """
        subst1 = load_substs(path1, data_name=self.data_name + '_1', drop_duplicates=self.drop_duplicates )
        subst2 = load_substs(path2, data_name=self.data_name + '_2', drop_duplicates=self.drop_duplicates )

        subst1['substs'] = subst1['substs_probs'].apply(lambda x: self.preprocess_substitutes(x, delete_word_parts=self.delete_word_parts))
        subst2['substs'] = subst2['substs_probs'].apply(lambda x: self.preprocess_substitutes(x, delete_word_parts=self.delete_word_parts))

        subst1['word'] = subst1['word'].apply(lambda x: x.replace('ё', 'е'))
        subst2['word'] = subst2['word'].apply(lambda x: x.replace('ё', 'е'))

        if self.max_examples is not None:
            subst1 = subst1.sample(frac=1).groupby('word').head(self.max_examples)
            subst2 = subst2.sample(frac=1).groupby('word').head(self.max_examples)

        return subst1, subst2


class Clustering_Pipeline(Evaluatable):
    def __init__(self, data_name, output_directory, vectorizer_name = 'count_tfidf', min_df = 10, max_df = 0.6,  max_number_clusters = 12,
                 use_silhouette = True, k = 2, n = 5, topk = None, lemmatizing_method = 'none', binary = False,
                 dump_errors = False, max_examples = None, delete_word_parts = False, drop_duplicates=True,
                 path_1 = None, path_2 = None, subst1 = None, subst2 = None, stream=None, should_dump_results=True):
        """
        output_directory -- location where all the results are going to be written

        vectorizer_name - [count, tfidf, count_tfidf]
            count - pure count vectorizer
            tfidf - pure tfidf vectorizer
            count_tfidf - count vectorizer for both subcorpuses, tfidf transformer unique for each subcorpus

        min_df, max_df - vectorizer params

        max_number_clusters - max number of clusters. Ignored when use_silhuette is set,

        use_silhouette - if set, algorithm runs through different number of clusters and pick one
            that gives the highest silhouette score

        k, n - task hyperparameters for binary classification
            word is considered to have gained / lost it's sence if it appears in this sense no more than k times
            in one subcorpus and no less then n times in the other one

        topk - how many (max) substitutes to take for each example

        lemmatizing_method - [none, single, all] - method of substitutes lemmatization
            none - don't lemmatize substitutes
            single - replace each substitute with single lemma (first variant by pymorphy)
            all - replace each substitutete with set of all variants of its lemma (by pymorphy)

        path_1, path_2 - paths to the dumped substitutes files for both corporas. In case you don't want to generate then on the go
        subst1, subst2 - you can also pass the pre-loaded substitutes as dataframes

        """

        print(data_name, vectorizer_name, min_df , max_df,  max_number_clusters ,
                 use_silhouette , k , n , topk , lemmatizing_method, binary,
                 dump_errors , max_examples , delete_word_parts , drop_duplicates,
                 path_1 , path_2 , subst1 , subst2)

        super().__init__(output_directory, should_dump_results)

        self.stream = stream if stream is not None else sys.stdout

        self.data_name = data_name

        self.mem = Memory('clustering_cache', verbose=0)
        self.transformer = None
        self.k = int(k) if int(k) >= 1 else float(k)
        self.n = int(n) if int(n) >= 1 else float(n)
        self.max_number_clusters = max_number_clusters
        self.use_silhouette = use_silhouette
        self.min_df = int(min_df) if int(min_df) >= 1 else float(min_df)
        self.max_df = int(max_df) if int(max_df) >= 1 else float(max_df)

        self.topk = topk

        self.vectorizer_name = vectorizer_name
        self.substitutes_params = dict()

        self.subst1 = subst1
        self.subst2 = subst2

        self.path_1 = path_1
        self.path_2 = path_2
        self.transformer1 = None
        self.transformer2 = None

        self.nonzero_indexes = dict()
        self.decision_clusters = dict()
        self.template = self.path_1.split('/')[-1].split('_')[0]

        self.lemmatizing_method = lemmatizing_method
        self.binary = binary
        self.dump_errors = dump_errors
        self.max_examples = max_examples
        self.delete_word_parts = delete_word_parts
        self.substs_loader = Substs_loader(data_name, lemmatizing_method, topk, max_examples, delete_word_parts, drop_duplicates)

        self.log_df = pd.DataFrame(columns=['word', 'dist1', 'dist2'])

        if vectorizer_name == 'tfidf':
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=self.min_df, max_df=max_df,
                                              binary = self.binary)

        elif vectorizer_name in ['count', 'count_tfidf']:
            self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=self.min_df, max_df=max_df,
                                              binary = self.binary)
            if vectorizer_name == 'count_tfidf':
                self.transformer1 = TfidfTransformer()
                self.transformer2 = TfidfTransformer()
        else:
            assert False, "unknown vectorizer name %s" % vectorizer_name

        print(self.get_params())

    def get_params(self):
        """
        return dictionary of hyperparameters to identify the run
        """
        init = getattr(Clustering_Pipeline.__init__, 'deprecated_original', Clustering_Pipeline.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        exclude_params = ['output_directory', 'self','stream', 'subst1', 'subst2', 'needs_preparing']
        parameters = [p.name for p in init_signature.parameters.values()
                      if p.name not in exclude_params and p.kind != p.VAR_KEYWORD]

        values = [getattr(self, key, None) for key in parameters]
        res = dict(zip(parameters, values))
        res['template'] = self.template
        return res

    def _get_score(self, vec1, vec2):
        return cosine(vec1, vec2)

    def _get_vectors(self,word, subs1, subs2):
        self.vectorizer = self.vectorizer.fit(np.concatenate((subs1, subs2)))
        vec1 = self.vectorizer.transform(subs1).todense()
        vec2 = self.vectorizer.transform(subs2).todense()

        vec1_count = vec1
        vec2_count = vec2

        if self.transformer1 is not None and self.transformer2 is not None:
            vec1 = self.transformer1.fit_transform(vec1).todense()
            vec2 = self.transformer2.fit_transform(vec2).todense()

        bool_array_1 = ~np.all(np.array(vec1) < 1e-6, axis=1)
        bool_array_2 = ~np.all(np.array(vec2) < 1e-6, axis=1)
        self.nonzero_indexes[word] = (np.where(bool_array_1)[0], np.where(bool_array_2)[0])

        vec1 = vec1[bool_array_1]
        vec2 = vec2[bool_array_2]

        return vec1, vec2, vec1_count, vec2_count

    def _prepare(self, data_name1, df1, data_name2, df2):
        """
        load substitutes if none provided
        """
        if self.subst1 is None or self.subst2 is None:
            self.subst1, self.subst2 = self.substs_loader.get_subs(self.path_1, self.path_2)


    def clusterize(self, word, subs1_df, subs2_df):
        """
        clustering.
        subs1 and subs2 dataframes for the specific word on the input
        distributions of clusters for subs1 and subs2 on the output
        """
        subs1 = subs1_df['substs']
        subs2 = subs2_df['substs']

        print("started clustering %s - %d samples" % (word, len(subs1) + len(subs2)), file=self.stream)
        vec1, vec2, vec1_count, vec2_count = self._get_vectors(word, subs1, subs2)

        border = len(vec1)
        transformed = np.asarray(np.concatenate((vec1, vec2), axis=0))

        if self.use_silhouette:
            labels, _, _, distances = clusterize_search(word, transformed, ncs=list(range(2, 5)) + list(range(7, 15, 2)))
        else:
            labels, _, _, distances = clusterize_search(word, transformed, ncs=(self.max_number_clusters,))

        dist1 = []
        dist2 = []

        left = list(labels[:border])
        right = list(labels[border:])
        for i in sorted(list(set(labels.tolist()))):
            dist1.append(left.count(i))
            dist2.append(right.count(i))

        print(dist1, file=self.stream)
        print(dist2, '\n', file=self.stream)

        distribution_one = np.array(dist1)
        distribution_two = np.array(dist2)

        return distribution_one, distribution_two, left, right

    def solve(self, target_words, data_name1, df1, data_name2, df2):
        """
        main method
        target words - list of target words
        data_name1, data_name2 - names of the data, such as 'rumacro_1', 'rumacro_2'
        df1, df2 - dataframes with data. Can be None, that data will be loaded using given names
        (or will not be loaded at all if there's no need in generating substitutes)
        """

        self._prepare(data_name1, df1, data_name2, df2)
        distances = []
        binaries = []
        targets = []
        for word in target_words:
            subs1_w = self.subst1[self.subst1['word'] == word]
            subs2_w = self.subst2[self.subst2['word'] == word]
            if len(subs1_w) == 0 or len(subs2_w) == 0:
                print("%s - no samples" % word, file=self.stream)
                continue
            targets.append(word)
            distribution1, distribution2, labels1, labels2 = self.clusterize(word, subs1_w, subs2_w)

            index1 = subs1_w.index[self.nonzero_indexes[word][0]]
            index2 = subs2_w.index[self.nonzero_indexes[word][1]]

            self.subst1.loc[index1, 'predict_sense_id'] = labels1
            self.subst2.loc[index2, 'predict_sense_id'] = labels2

            if distribution1.size == 0 or distribution2.size == 0:
                print("for word %s zero examples in corporas - %d, %d" , file=self.stream%
                      (word, len(distribution1), len(distribution2)))
                distance = sum(distances) / len(distances)
                binary = 1
                distances.append(distance)
                binaries.append(binary)
                continue

            distance = self._get_score(distribution1, distribution2)
            binary = self.solve_binary(word, distribution1, distribution2)
            distances.append(distance)
            binaries.append(binary)

            print(word, ' -- ', distance, ' ', binary, file=self.stream)
        print(len(targets), "words processed", file=self.stream)
        self.log_df.to_csv(r'words_clusters.csv', index=False)

        return list(zip(targets, distances, binaries))

    def solve_binary(self, word, dist1, dist2):
        """
        solving binary classification subtask using the clustering results
        """
        for i, (count1, count2) in enumerate(zip(dist1, dist2)):
            if count1 <= self.k and count2 >= self.n or count2 <= self.k and count1 >= self.n:
                if self.dump_errors:
                    self.decision_clusters[word] = i
                return 1
        return 0

    def solve_for_one_word(self, word, stream = None):
        if stream is not None:
            self.stream = stream
        subs1_w = self.subst1[self.subst1['word'] == word]
        subs2_w = self.subst2[self.subst2['word'] == word]
        if len(subs1_w) == 0 or len(subs2_w) == 0:
            print("%s - no samples<br>" % word, file=self.stream)
            return
        distribution1, distribution2, labels1, labels2 = self.clusterize(word, subs1_w, subs2_w)

        index1 = subs1_w.index[self.nonzero_indexes[word][0]]
        index2 = subs2_w.index[self.nonzero_indexes[word][1]]

        self.subst1.loc[index1, 'labels'] = labels1
        self.subst2.loc[index2, 'labels'] = labels2

        if distribution1.size == 0 or distribution2.size == 0:
            print("for word %s zero examples in corporas - %d, %d" %
                (word, len(distribution1), len(distribution2)), file=self.stream)
            return None, None

        distance = self._get_score(distribution1, distribution2)
        binary = self.solve_binary(word, distribution1, distribution2)

        print(word, ' -- ', distance, ' ', binary, '<br>',  file=self.stream)
        return binary, distance

###########################################################
"""
hyperparameters that for the search grid
binary parameter 'use_silhuette' is not included as it's processed in a special way
"""
search_ranges = {
    'vectorizer_name' : ['tdidf', 'count', 'count_tfidf'],
    'min_df' : [1, 3, 5, 7, 10, 20, 30],
    'max_df': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    'k':list(range(2,6)) + list(range(6,22,3)),
    'n':list(range(3,7)) + list(range(7,30,3)),
    'max_number_clusters': [3, 4, 5, 7, 8, 10, 12, 15],
    'topk' : [15, 30, 50, 100, 150],
    'lemmatizing_method' : ['none', 'single', 'all']
}

class Clustering_Search(GridSearch):
    def __init__(self, output_directory, subst1_path, subst2_path, subdir = None, vectorizer_name = None, min_df = None,
                 max_df = None,  max_number_clusters = None,use_silhouette = None, k = None, n = None,
                 topk = None, lemmatizing_method=None, binary = False, dump_errors = False, max_examples = None,
                 delete_word_parts = False, drop_duplicates=True, should_dump_results=True):
        """
        subst1_path, subst2_path - paths to the substitutes. Two cases:
        1) subst1_path and subst2_path are directories containing substitutes dumps (in that case the search will be
            iterating through all substitutes files found in subst1_path/subdir, considering
            subst2_path/subdir has the same contents)
        2) subst1_path and subst2_path are full paths to the substitutes dump files. 'subdir' param is ignored

        output_directory -- location where all the results are going to be written

        vectorizer_name - [count, tfidf, count_tfidf]
            count - pure count vectorizer
            tfidf - pure tfidf vectorizer
            count_tfidf - count vectorizer for both subcorpuses, tfidf transformer unique for each subcorpus

        min_df, max_df - vectorizer params

        max_number_clusters - max number of clusters. Ignored when use_silhuette is set,

        use_silhouette - if set, algorithm runs through different number of clusters and pick one
            that gives the highest silhouette score

        k, n - task hyperparameters for binary classification
            word is considered to have gained / lost it's sence if it appears in this sense no more than k times
            in one subcorpus and no less then n times in the other one

        topk - how many (max) substitutes to take for each example

        lemmatizing_method - [none, single, all] - method of substitutes lemmatization
            none - don't lemmatize substitutes
            single - replace each substitute with single lemma (first variant by pymorphy)
            all - replace each substitutete with set of all variants of its lemma (by pymorphy)
        """
        super().__init__()
        self.vectorizer_name = vectorizer_name
        self.max_number_clusters = max_number_clusters
        self.use_silhouette = use_silhouette
        self.min_df = min_df
        self.max_df = max_df
        self.k = k
        self.n = n
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        self.topk = topk
        self.lemmatizing_method = lemmatizing_method
        self.binary = binary
        self.dump_errors = dump_errors
        self.should_dump_results = should_dump_results
        self.template = None
        self.max_examples = max_examples

        self.substitutes_params = dict()
        self.substs = dict()
        self.subst_paths = self.get_subst_paths(subst1_path, subst2_path, subdir)
        self.delete_word_parts = delete_word_parts
        self.drop_duplicates = drop_duplicates

    def get_substs(self, data_name, subst1_path, subst2_path, topk, lemmatizing_method):
        """
        loads substitutes from specific path, unless they already present
        """
        line1 = subst1_path + str(topk) + str(lemmatizing_method)
        line2 = subst2_path + str(topk) + str(lemmatizing_method)

        if line1 not in self.substs or line2 not in self.substs:
            substs_loader = Substs_loader(data_name, lemmatizing_method, topk, self.max_examples,
                                          self.delete_word_parts, drop_duplicates=self.drop_duplicates)
            self.substs[line1], self.substs[line2] = substs_loader.get_subs( subst1_path, subst2_path )

        return self.substs[line1], self.substs[line2]

    def get_subst_paths(self, subst_path1, subst_path2, subdir):
        """
        resolves substitutes path. Returns a list of path pairs to iterate through
        is paths are passed directly to the substitutes dump files, list only contains one element
        """

        if os.path.isfile(subst_path1) or '+' in subst_path1:
            assert os.path.isfile(subst_path2) or '+' in subst_path2, "inconsistent substitutes paths - %s, %s" % (subst_path1, subst_path2)
            self.template = subst_path1.split('/')[-1].split('_')[0]
            return [(subst_path1, subst_path2)]

        tmp_path1 = subst_path1 + '/' + subdir
        tmp_path2 = subst_path2 + '/' + subdir

        files = [i for i in os.listdir(tmp_path1) if i.split('.')[-1] != 'input']
        res = [(tmp_path1 + '/' + file, tmp_path2 + '/' + file) for file in files]
        return res

    def get_output_path(self):
        filename = self.output_directory + '/' + str(self.template) + "_clustering_parameters_search"
        return filename

    def create_evaluatable(self, data_name, params):
        if params['k'] >= params['n']:
            return None
        subst1_path = params['path_1']
        subst2_path = params['path_2']

        substs1, substs2 = self.get_substs(data_name, subst1_path, subst2_path,
                                           params['topk'], params['lemmatizing_method'])

        res = Clustering_Pipeline(data_name, self.output_directory, **params, binary = self.binary,
                                  dump_errors = self.dump_errors, should_dump_results=self.should_dump_results, max_examples = self.max_examples,
                                  delete_word_parts=self.delete_word_parts, drop_duplicates=self.drop_duplicates,
                                  subst1=substs1, subst2=substs2)
        return res

    def _create_params_list(self, lists):
        tmp_res = []
        params_names = list(search_ranges.keys())
        params_names.append('use_silhouette')
        if not self.use_silhouette or self.use_silhouette is None:
            lists['use_silhouette'] = [False]
            tmp_res += [dict(zip(params_names, values)) for values in product(*lists.values())]
        if self.use_silhouette or self.use_silhouette is None:
            lists['use_silhouette'] = [True]
            lists['max_number_clusters'] = [0]
            tmp_res += [dict(zip(params_names, values)) for values in product(*lists.values())]

        res = []
        for subst1_path, subst2_path in self.subst_paths:
            for params in tmp_res:
                params['path_1'] = subst1_path
                params['path_2'] = subst2_path
                res.append(params)
        return res

    def get_params_list(self):
        lists = dict()
        for param, values in search_ranges.items():
            val = getattr(self, param, [])
            assert val != [], "no attribute %s in the class" % param
            if val == None:
                lists[param] = search_ranges[param]
            else:
                lists[param] = [val]
        res = self._create_params_list(lists)
        return res

    def evaluate(self, data_name):
        """
        run the evaluation with given parameters
        checks that all the searchable parameters are set explicitly
        """
        list = self.get_params_list()
        if len(list) > 1:
            nones = []
            for param in search_ranges.keys():
                item = getattr(self, param, None)
                if item is None:
                    nones.append(param)
            print("not all parameters are set: %s" % str(nones))
            return 1
        evaluatable = self.create_evaluatable(data_name, list[0])
        evaluatable.evaluate()

    def solve(self, data_name, output_file_name = None):
        """
        run the evaluation with given parameters
        checks that all the searchable parameters are set explicitly
        """

        list = self.get_params_list()
        if len(list) > 1:
            nones = []
            for param in search_ranges.keys():
                item = getattr(self, param, None)
                if item is None:
                    nones.append(param)
            print("not all parameters are set: %s" % str(nones))
            return 1

        params = list[0]

        list = self.get_params_list()

        if len(list) > 1:
            nones = []
            for param in search_ranges.keys():
                item = getattr(self, param, None)
                if item is None:
                    nones.append(param)
            print("not all parameters are set: %s" % str(nones))
            return 1
        evaluatable = self.create_evaluatable(data_name,params)
        target_words = load_target_words(data_name)
        if target_words is None:
            target_words = evaluatable.subst1['word'].unique()
        else:
            target_words = [i.split('_')[0] for i in target_words]
        evaluatable.solve(target_words,  data_name + '_1', None,  data_name + '_2', None)

        if output_file_name is not None:
            dd = Path(self.output_directory)
            for n,df in enumerate((evaluatable.subst1, evaluatable.subst2)):
                df.loc[df.predict_sense_id.isnull(), 'predict_sense_id'] = -1
                df.to_csv(dd / (output_file_name + f'_{n+1}.csv'), sep='\t', index=False)

if __name__ == '__main__':
    fire.Fire(Clustering_Search)
