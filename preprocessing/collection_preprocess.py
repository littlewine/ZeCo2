from regexXML import Attr, Tag
from transformers import BertTokenizer
import pickle
from bs4 import BeautifulSoup

count_tokens = True

# Convert XML to tsv

def clean_xmltag(txt, xml_tag_start, xml_tag_end):
    return txt.split(xml_tag_start)[-1].split(xml_tag_end)[0]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

collection_path = '/ivi/ilps/personal/akrasak/data/collections/msmarco-docs.trecweb'
output_path = collection_path.replace("trecweb", "tsv")
if collection_path==output_path:
    raise ValueError("output path should be != than original collection path ")

title_length = dict()
whole_passage_length = dict()

# Define tags for parsing
doc_re = Tag("DOC")
docid_re = Tag("DOCNO")
doctitle_re = Tag("TITLE")
passage_re = Tag("passage")

# with open(collection_path, 'r') as f:
#     marco_sample = f.read()
# doc_entity = doc_re.search(marco_sample)
# doc_id = docid_re.search(doc_entity.group()).group()
# doc_id = clean_xmltag(doc_id, xml_tag_start='<DOCNO>', xml_tag_end='</DOCNO>')
#
# doc_title =  doctitle_re.search(doc_entity.group()).group()
# doc_title = clean_xmltag(doc_title, xml_tag_start='<TITLE>', xml_tag_end='</TITLE>')
# print(f"{doc_id} title length: {len(tokenizer.encode(doc_title))}")

with open(collection_path, "r") as f:
    for doc_entity in doc_re.finditer_from_file(f):
        doc_id = docid_re.search(doc_entity.group()).group()
        doc_id = clean_xmltag(doc_id, xml_tag_start='<DOCNO>', xml_tag_end='</DOCNO>')

        doc_title = doctitle_re.search(doc_entity.group()).group()
        doc_title = clean_xmltag(doc_title, xml_tag_start='<TITLE>', xml_tag_end='</TITLE>')
        title_length[doc_id] = len(tokenizer.encode(doc_title))

        # for passage in doc
        for passage in passage_re.finditer(doc_entity.group("inner")):
            # doc_id =
            passage_id = int(passage.group('attr').split("id=")[1]) #TODO: fix parsing here: breaks when
            # docid == MARCO_D1555982-1
            # passage == '<passage id=1>\n= 3.22x10+13 m Source (s):http://en.wikipedia.org/wiki/Stefan_bolt...schmiso · 1 decade ago0 18 Comment Schmiso, you forgot a 4 in your answer. Your link even says it: L = 4pi (R^2)sigma (T^4). Using L, luminosity, as the energy in this problem, you can find the radius R by doing sqrt (L/ (4pisigma (T^4)). Hope this helps everyone. Caroline · 4 years ago4 1 Comment (Stefan-Boltzmann law) L = 4pi*R^2*sigma*T^4 Solving for R we get: => R = (1/ (2T^2)) * sqrt (L/ (pi*sigma)) Plugging in your values you should get: => R = (1/ (2 (11,000K)^2)) *sqrt ( (2.7*10^32W)/ (pi * (5.67*10^-8 W/m^2K^4))) R = 1.609 * 10^11 m? · 3 years ago0 1 Comment Maybe you would like to learn more about one of these? \n</passage>'

            passage_txt = passage.group('inner')
            doc_passage_id = f"{doc_id}-{passage_id}"
            title_passage_txt = f"{doc_title}. {passage_txt}"
            # print(f"{doc_passage_id} passage length: {len(tokenizer.encode(title_passage_txt))}")
            # Fix some things in text
            title_passage_txt = title_passage_txt.replace("\n",' ')
            soupped_content = BeautifulSoup(title_passage_txt, 'html.parser')

            whole_passage_length[doc_passage_id] = len(tokenizer.encode(title_passage_txt))
            if whole_passage_length[doc_passage_id]>180:
                print(doc_passage_id,' length: ', whole_passage_length[doc_passage_id])
            # print(passage_id, title_passage_txt)
            # print("**********")
        #TODO: fix character escaping issue (\n, \') etc.
        # Write out docs to file
        with open(output_path, 'a') as filewriter:
            filewriter.write(f"{doc_passage_id} \t {soupped_content}")

# Write token statistics
token_stats = {'docs': whole_passage_length,
               'titles': title_length}
with open(output_path+".lengths", 'w') as file:
    file.write(json.dumps(token_stats))  # use `json.loads` to do the reverse

print("Finished.")