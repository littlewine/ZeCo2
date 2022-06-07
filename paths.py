import os
import platform

if platform.system() == 'Darwin':
    print('Using paths for Macbook')
    path_project_data = '/Users/amkrasakis/data/0sConvDR'
    path_anserini = '/Users/amkrasakis/libs/anserini'
    path_project_base = '/Users/amkrasakis/cqa-rewrite'

elif platform.system() == 'Linux':
    print('Using paths for slurm')
    path_anserini = '/home/akrasak/anserini' #TODO: change path to anserini (used for evaluation)
    path_project_data = '/ivi/ilps/personal/akrasak/data/0sConvDR' # TODO: replace with 'data/'
    path_project_base = '/home/akrasak/cqa-rewrite' # TODO: change with local path of current repo (where this file is located)

path_treceval = os.path.join(path_anserini,'tools','eval','trec_eval.9.0.4','trec_eval')

# Queries, qrels

path_queries = {'cast19':
                    {'full_conv': os.path.join(path_project_data,'queries/2019', 'full_conv.tsv'),
                       'raw': os.path.join(path_project_data,'queries/2019', 'raw.tsv'),
                       'human': os.path.join(path_project_data,'queries/2019', 'human.tsv'),
                       },
                'cast20':{
                    'json' :'/ivi/ilps/personal/akrasak/data/treccastweb/2020/2020_automatic_evaluation_topics_v1.0.json',
                    'full_conv': os.path.join(path_project_data, 'queries/2020',
                                              'full_conv_canonical_256.tsv'),

                    'full_conv_canonical': os.path.join(path_project_data, 'queries/2020', 'full_conv_canonical.tsv'),
                    'full_conv_canonical_256': os.path.join(path_project_data, 'queries/2020',
                                                            'full_conv_canonical_256.tsv'),
                    'full_conv_canonical_384': os.path.join(path_project_data, 'queries/2020',
                                                            'full_conv_canonical_384.tsv'),
                    'raw': os.path.join(path_project_data, 'queries/2020', 'raw.tsv'),
                    'human': os.path.join(path_project_data, 'queries/2020', 'human.tsv'),

                    'history_only': os.path.join(path_project_data, 'queries/2020', 'history_only.tsv'),
                    'canonical_only': os.path.join(path_project_data, 'queries/2020', 'canonical_only.tsv'),
                    'canonical_passage_pickle': os.path.join(path_project_data, 'queries/2020', 'canonical_passages.pickle'),
                    'qid_mapping': os.path.join(path_project_data, 'queries/2020', 'qid_mapping.pickle'),

                },
                'cast21':{
                    'full_conv': os.path.join(path_project_data, 'queries/2021',
                                                            'full_conv_canonical_256.tsv'),

                    'full_conv_canonical': os.path.join(path_project_data,'queries/2021', 'full_conv_canonical.tsv'),
                    'full_conv_canonical_256': os.path.join(path_project_data,'queries/2021', 'full_conv_canonical_256.tsv'),
                    'full_conv_canonical_384': os.path.join(path_project_data,'queries/2021', 'full_conv_canonical_384.tsv'),

                    'raw': os.path.join(path_project_data, 'queries/2021', 'raw.tsv'),
                    'human': os.path.join(path_project_data, 'queries/2021', 'human.tsv'),

                    'history_only': os.path.join(path_project_data, 'queries/2021', 'history_only.tsv'),
                    'canonical_only': os.path.join(path_project_data,'queries/2021', 'canonical_only.tsv'),

                    'json': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/2021_automatic_evaluation_topics_v1.0.json',
                },
}

path_qrels = {
    'cast19': os.path.join(path_project_data,'qrels','2019qrels.txt'),
    'cast19MARCO': os.path.join(path_project_data,'qrels','2019qrels_MARCO.txt'),
    'cast20': os.path.join(path_project_data,'qrels','2020qrels.txt'),
    'cast21': os.path.join(path_project_data,'qrels','qrels-docs.2021.txt')
}

# Collections
path_collections = "data/collection_samples"

# For CAsT21 I used the trecweb files provided from the organizers :
# paths_trecweb_cast21 = {'wapo': os.path.join(path_collections,'wapo.trecweb'),
#                         'marco': os.path.join(path_collections,'msmarco-docs.trecweb'),
#                         'kilt': os.path.join(path_collections,'kilt_knowledgesource.trecweb'),
#                         }

path_collection_int = {
                        # check examples in `data/collection_samples/msmarco-docs.tsv.sample.int`
                        'CAR':'/ivi/ilps/personal/akrasak/data/collections/car-wiki2020-01-01/Car_collection.tsv.int',
                       'marcoP': '/ivi/ilps/personal/akrasak/data/collections/msmarco-passage/collection.tsv',
                       'marcoD': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.int',
                       'KILT': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.int',
                       'WAPO': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.int',
                       }

path_collection_mappings = {
                            # check examples in `data/collection_samples/msmarco-docs.tsv.sample.intmapping`
                            'CAR':'/ivi/ilps/personal/akrasak/data/collections/car-wiki2020-01-01/Car_collection.tsv.intmapping',
                            'marcoP': '/ivi/ilps/personal/akrasak/data/collections/msmarco-passage/collection.tsv.intmapping',
                            'marcoD': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.intmapping',
                            'KILT': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.intmapping',
                            'WAPO': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.intmapping',
                           }

# Colbert paths
path_colbert_checkpoint = '/ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn' #TODO: download colbert checkpoint and point to this path `colbert-400000.dnn`

# FAISS index paths (generated after ColBERT indexing)
path_index_root = '/ivi/ilps/personal/akrasak/data/faiss_indexes/' # TODO: replace with parent FAISS index folder
index_names = {'marcoP': 'MSMARCO.L2.32x200k.180len', # the FAISS index of MSMarco should be @ `faiss_indexes/MSMARCO.L2.32x200k.180len/`
               'CAR':'CAR.FirstP.L2.32x200k.180len',
               'marcoD': 'marco.FirstP.L2.32x200k.180len',
               'KILT': 'kilt.FirstP.L2.32x200k.180len',
               'WAPO': 'wapo.FirstP.L2.32x200k.180len',
               }

# path rankings: those are the output ranking files (needed for the analysis section)
path_rankings_noexp = {
                'cast19': {'raw': 'data/rankings/cast19-raw-noexp.trecrun',
                           'ctx': 'data/rankings/cast19-lastturn-noexp.trecrun',
                           'allHist': 'data/rankings/cast19-allhist-noexp.trecrun',
                           'human': 'data/rankings/postprocessed_run.trecrun'
                           },
                'cast20': {'raw': 'data/rankings/cast20-raw-noexp.trecrun',
                           'ctx': 'data/rankings/cast20-last_turn-noexp.trecrun',
                           'allHist': 'data/rankings/cast20-allhist-noexp.trecrun',
                           'human': 'data/rankings/cast20-human-noexp.trecrun',
                           },
                'cast21': {'raw': 'data/rankings/cast21-raw-noexp.trecrun',
                           'ctx': 'data/rankings/cast21-lastturn-noexp.trecrun',
                           'allHist': 'data/rankings/cast21-allhist-noexp.trecrun',
                           'human': 'data/rankings/cast21-human-noexp.trecrun',
                           }
}
