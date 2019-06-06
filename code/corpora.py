from lxml import etree
from nltk.corpus import wordnet as wn
from my_utils import *

def parse(corpora_path: str, bn2wn_mapping_path: str, outfile_path: str, c_type: str = "precision") -> None:
    """
    :param corpora_path; Full path to the corpora_path to be parsed
    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora
    :param outfile_path; Full path to write the sentences extracted from the corpora
    :param c_type; Corpora type "precision" or "coverage"
    :return None

    THIS FUNCTION HANDLES ONLY EUROSENSE CORPORA
    """
    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_path)

    sentence = ""
    context = etree.iterparse(corpora_path, events=('start', 'end'))

    anchors = list()
    lemmas = list()
    bn_keys = list()
    coherenceScores = list()
    sentence_count = 0

    with open(outfile_path, 'w', encoding='utf-8') as output_file:
        for event, elem in context:
            if (event == 'start'):
                if (elem.tag == 'text' and elem.attrib['lang'] == 'en'):
                    sentence = elem.text
                    
                elif (elem.tag == 'annotation' and elem.attrib['lang'] == 'en'):
                    """
                    For each annotation, this block;
                    * Uniquesly populates lists for later "replace" to avoid multiple replacements
                    * prioritizes the longer annotations for the PRECISION dataset, then
                    * prioritizes annotations with higher coherenceScore for the COVERAGE dataset, then
                    * cross-checks each annotations with the wordnet equivalent to verify that it's CORRECTLY annotated
                    """
                    if (elem.attrib['anchor'].lower() not in anchors):
                        found = False
                        for k, anchor in enumerate(anchors):
                            new_anchors = [i.lower() for i in elem.attrib['anchor'].replace(" ","-").split("-")]
                            if (anchor in new_anchors or elem.attrib['anchor'].lower() in anchor.replace(" ","-").split("-")):
                                found = True

                                if (c_type == "precision" and len(anchor) < len(elem.attrib['anchor'])):
                                    if (elem.text in bn2wn_mapping):
                                        offset = bn2wn_mapping[elem.text]
                                        wn_synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]) ).lemma_names()
                                        if (elem.attrib['lemma'].lower() in wn_synset):
                                            anchors[k] = elem.attrib['anchor'].lower()
                                            lemmas[k] = elem.attrib['lemma'].replace(" ","_").lower()
                                            bn_keys[k] = elem.text

                                elif (c_type == "coverage" and elem.attrib['coherenceScore'] > coherenceScores[k]):
                                    if (elem.text in bn2wn_mapping):
                                        offset = bn2wn_mapping[elem.text]
                                        wn_synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]) ).lemma_names()
                                        if (elem.attrib['lemma'].lower() in wn_synset):
                                            anchors[k] = elem.attrib['anchor'].lower()
                                            lemmas[k] = elem.attrib['lemma'].replace(" ","_").lower()
                                            bn_keys[k] = elem.text
                                            coherenceScores[k] = elem.attrib['coherenceScore']

                        if (not found):
                            if (elem.text in bn2wn_mapping):
                                offset = bn2wn_mapping[elem.text]
                                wn_synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]) ).lemma_names()
                                if (elem.attrib['lemma'].lower() in wn_synset):
                                    anchors.append(elem.attrib['anchor'].lower())
                                    lemmas.append(elem.attrib['lemma'].replace(" ","_").lower())
                                    bn_keys.append(elem.text)
                                    coherenceScores.append(elem.attrib['coherenceScore'])

            elif (event == 'end'):
                if (elem.tag == 'sentence'):
                    if (sentence):
                        sentence = replace_words(sentence, anchors, lemmas, bn_keys)
                        """
                        Encoding as utf-8 here because I got some errors writing some characters
                        This will cause the sentence to be written as strings of bytes; b'sentence'
                        Since the type when read back from the file will be string and not bytes,
                        I used string list indexing to eliminate the leading b' and the trailing ' 
                        """
                        output_file.write("{}\n".format(sentence.encode('utf-8')))
                        sentence_count += 1
                        print("{:,d} sentences extracted...".format(sentence_count), end="\r")
                        
                        sentence = ""
                        anchors = list()
                        lemmas = list()
                        bn_keys = list()
                    
            #Freeing up memory before parsing the next tag
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    print ("\nDone!")