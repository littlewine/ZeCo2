import os
import platform

if platform.system() == 'Darwin':
    print('Using paths for Macbook')
    path_data_folder = '/Users/amkrasakis/data/'
    # path_treccast20 = '/Users/amkrasakis/data/treccastweb/2020'
    path_canard = '/Users/amkrasakis/data/CANARD_Release'
    path_index = '/Users/amkrasakis/data/indexes/index-cast2019'
    path_project_data = '/Users/amkrasakis/data/0sConvDR'
    path_anserini = '/Users/amkrasakis/libs/anserini'
    path_project_base = '/Users/amkrasakis/cqa-rewrite'

elif platform.system() == 'Linux':
    print('Using paths for slurm')
    path_data_folder = '/ivi/ilps/personal/akrasak/data/'
    path_anserini = '/home/akrasak/anserini'
    path_index = '/ivi/ilps/personal/akrasak/data/indexes/cast2019/index-cast2019'
    path_project_data = '/ivi/ilps/personal/akrasak/data/0sConvDR'
    # path_treccast20 = '/ivi/ilps/personal/akrasak/data/treccastweb/2020'
    path_project_base = '/home/akrasak/cqa-rewrite'

path_treceval = os.path.join(path_anserini,'tools','eval','trec_eval.9.0.4','trec_eval')

# Queries, qrels

# path_queries_21 = {'json' :'/ivi/ilps/personal/akrasak/data/treccastweb/2021/2021_automatic_evaluation_topics_v1.0.json',
#                     'queries_with_canonical': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical.tsv',
#                     'queries_with_canonical_256': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_256.tsv',
#                     'queries_with_canonical_200': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_200.tsv',
#                     'queries_with_canonical_186': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_186.tsv',
#                    'queries_with_canonical_384': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_384.tsv',
#
#                    'queries_canonical_last': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_canonical_last.tsv',
#                    'queries_history_only': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_history_only.tsv',
#
#                    'raw': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_raw.tsv',
#                    'raw_SEP': '/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_raw_SEP.tsv',
#                    }

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
path_collections = "/ivi/ilps/personal/akrasak/data/collections"
paths_trecweb_cast21 = {'wapo': os.path.join(path_collections,'wapo.trecweb'),
                        'marco': os.path.join(path_collections,'msmarco-docs.trecweb'),
                        'kilt': os.path.join(path_collections,'kilt_knowledgesource.trecweb'),
                        }

path_collection_int = {'CAR':'/ivi/ilps/personal/akrasak/data/collections/car-wiki2020-01-01/Car_collection.tsv.int',
                       'marcoP': '/ivi/ilps/personal/akrasak/data/collections/msmarco-passage/collection.tsv',
                       'marcoD': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.int',
                       'KILT': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.int',
                       'WAPO': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.int',
                       }

path_collection_mappings = {'CAR':'/ivi/ilps/personal/akrasak/data/collections/car-wiki2020-01-01/Car_collection.tsv.intmapping',
                            'marcoP': '/ivi/ilps/personal/akrasak/data/collections/msmarco-passage/collection.tsv.intmapping',
                            'marcoD': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.intmapping',
                            'KILT': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.intmapping',
                            'WAPO': '/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.intmapping',
                           }

# Colbert paths
path_colbert_checkpoint = '/ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn'
path_index_root = '/ivi/ilps/personal/akrasak/data/faiss_indexes/'

index_names = {'marcoP': 'MSMARCO.L2.32x200k.180len',
               'CAR':'CAR.FirstP.L2.32x200k.180len',
               'marcoD': 'marco.FirstP.L2.32x200k.180len',
               'KILT': 'kilt.FirstP.L2.32x200k.180len',
               'WAPO': 'wapo.FirstP.L2.32x200k.180len',
               }

# path rankings
path_rankings_noexp = {
                'cast19': {'raw': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast19-raw-noexp/python_pipeline.py/2021-11-26_17.28.54/postprocessed_run.trecrun',
                           'ctx': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast19-lastturn-noexp/python_pipeline.py/2021-11-26_19.38.37/postprocessed_run.trecrun',
                           'allHist': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast19-allhist-noexp/python_pipeline.py/2021-11-29_11.18.49/postprocessed_run.trecrun',
                           'human': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast19-human-noexp/python_pipeline.py/2021-12-06_17.32.17/postprocessed_run.trecrun'
                           },
                'cast20': {'raw': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast20-raw-noexp/python_pipeline.py/2021-12-06_10.00.38/postprocessed_run.trecrun',
                           'ctx': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast20-last_turn-noexp/python_pipeline.py/2021-12-06_13.44.50/postprocessed_run.trecrun',
                           'allHist': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast20-allhist-noexp/python_pipeline.py/2021-12-07_12.04.15/postprocessed_run.trecrun',
                           'human': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast20-human-noexp/python_pipeline.py/2021-12-06_18.38.18/postprocessed_run.trecrun',
                           },
                'cast21': {'raw': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast21-raw-noexp/python_pipeline.py/2021-11-30_13.39.36/postprocessed_run.trecrun',
                           'ctx': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast21-lastturn-noexp/python_pipeline.py/2021-11-29_13.15.12/postprocessed_run.trecrun',
                           'allHist': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast21-allhist-noexp/python_pipeline.py/2021-11-30_23.34.34/postprocessed_run.trecrun',
                           'human': '/home/akrasak/cqa-rewrite/ColBERT/experiments/cast21-human-noexp/python_pipeline.py/2021-12-06_19.44.48/postprocessed_run.trecrun',
                           }
}
