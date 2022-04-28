import os
os.environ["JAVA_HOME"] = "/home/akrasak/anaconda3/envs/pyserini"

from jnius import autoclass

# testing anserini stuff
JString = autoclass('java.lang.String')
JString('Hello world')

os.chdir("/home/akrasak/cqa-rewrite/ColBERT")
from pyserini import collection, index

# for syspath in ['/home/akrasak/apache-maven-3.6.3/bin',
#                  "/home/akrasak/jdk-11.0.7/bin:",
#                  '/home/akrasak/anaconda3/envs/pyserini/bin',
#                 ]:
#     sys.path.append(syspath)

path_collections = "/ivi/ilps/personal/akrasak/data/collections"
paths_trecweb_cast21 = {'wapo': os.path.join(path_collections,'wapo.trecweb'),
                        'marco': os.path.join(path_collections,'msmarco-docs.trecweb'),
                        'kilt': os.path.join(path_collections,'kilt_knowledgesource.trecweb'),
                        }

output_path = os.path.join(path_collections,"wapo.anserini.tsv")
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"removed {output_path}")

collection_obj = collection.Collection('TrecwebCollection', paths_trecweb_cast21['wapo'])
generator = index.Generator('DefaultLuceneDocumentGenerator')

for (i, fs) in enumerate(collection_obj):
    for (j, doc) in enumerate(fs):
        parsed = generator.create_document(doc)
        docid = parsed.get('id')            # FIELD_ID
        passageid = parsed.get('id')        # PASSAGE_ID
        docpid = f"{docid}-{passageid}"
        raw = parsed.get('raw')             # FIELD_RAW
        contents = parsed.get('contents').strip()   # FIELD_BODY
        print('{} {} -> {} {}...'.format(i, j, docid, contents.strip().replace('\n', ' ')[:50]))

        #TODO: export tsv

        #TODO: export mapping

        break
