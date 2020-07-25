# BOS_AggloSil

This repo contains code to reproduce the results of cs2020 team on the task 1 of SemEval-2020 workshop

Required packages:
flask - 1.1.1
fire - 0.3.1
regex - 2020.5.14
pymorphy2 - 0.8
pytorch - 1.4.0
spacy - 2.1.3
pandas - 1.0.1
numpy - 1.18.1
scipy - 1.1.0
scikit-learn - 0.21.1
joblib - 0.14.1
matplotlib - 3.2.0
seaborn - 0.10.0

We used anaconda 4.8.3

### Substitutes
The generated substitutes files can be found here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3948252.svg)](https://doi.org/10.5281/zenodo.3948252)
To use these files one should unpack the archives.
The substitutes dumps that were used during the evaluation phase are marked with prefix "EVALUATION".

### Launching
The command line template to launch the programm looks like this:
```
./evaluate_final.sh <data_name> <output_directory> <substitutes_path1> <substitutes_path2> <lemmatizing_method> <vectorizer_name>
```
### Parameters meaning

* data_name - target language name - english, german, swedish or latin
* output_directory - the directory where results are going to be stored
* substitutes_path1, substitutes_path2 - the paths to the substitutes dump files (.npz) corresponding with old and new corpora respectively. 
If you desire to use the patterns combination you should add the prefix "+0.0+" to the end of the substitutes dump path. Example:

```
./\<mask\>\<mask\>-or-T-2ltr2f_topk200_fixspacesTrue.npz+0.0+
```

* lemmatizing_method - 'none' - no lemmatization of the substitutes. 'single' - every substitute is replaced by its most probable lemma (only works for English and German). 
* vectorizer name - 'count' to use default CountVectorizer, 'tfidf' to apply tf-idf.

command line example:
```
./evaluate_final.sh english ./res ./english_1/\<mask\>\<mask\>-or-T-2ltr2f_topk200_fixspacesTrue.npz+0.0+ ./english_2/\<mask\>\<mask\>-or-T-2ltr2f_topk200_fixspacesTrue.npz+0.0+ none count```
```
### Visualisation tool

the results visualisation tool for SCD and WSI tasks can be found [here](https://github.com/DeadBread/SCD_WSI_tool).
To launch it type:
```
python web/app.py --data_name=<data_name> --subst1_path=<subst1_path>
```
Mandatory parameters:
data_name - name of the dataset (language name for SemEval-2020 task-1 datasets)
subst1_path - path to the substitutes dump file (.npz) for corpus 1 (it is the only file in case of WSI task) For patterns combination use postfix "+0.0+"
Optional patameters:
subst1_path - path to the substitutes dump file (.npz) for corpus 2 (required for SCD task) For patterns combination use postfix "+0.0+"
vectorizer_name - string - 'count' to use default CountVectorizer, 'tfidf' to apply tf-idf.
min_df - int - vactorization parameter - minimal document frequency between the substitutes sets for the substitute to be included into the vocabulary.
max_df - int - maximal document frequency between the substitutes sets for the substitute to be included into the vocabulary.
number_of_clusters - int - Number of clusters. IS silhouette score is used to determine the optimal number of clusters than this parameter is ignored.
use_silhouette - bool - flag whether to use silhouette score to determine number of clusters
k - int / float - SCD parameter (is ignored in WSI mode) - the acceptable "noise level" fot decision cluster (maximal number of examples from the other corpus) to select the cluster as decision cluster.
n - int / float - SCD parameter (is ignored in WSI mode) - the required number of examples from one corpus in the cluster for it to be selected as decision cluster
topk - int - number of substitutes limitation
lemmatizing_method - string - whether is set to 'none' - no lemmatization or single - every substitute is replaced by its most probable lemma (in case of Russian language lemma is picked randomly).
If the value "all" is selected (only works for Russian language) each substitute is replaced by all possible variants of its lemmatization. 

