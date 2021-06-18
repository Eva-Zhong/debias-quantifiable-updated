"""Preprocess the SQuAD dataset with CoreNLP tokenization and tagging"""
import argparse
import json
import pickle
import regex
from os import mkdir, makedirs
from os.path import join, exists, dirname
from typing import Iterable, List, Dict
from tqdm import tqdm

import numpy as np

from debias import config
from debias.datasets.squad import AnnotatedSquadParagraph, SquadQuestion
from debias.preprocessing.corenlp_client import CoreNLPClient
from debias.utils import py_utils
from debias.utils.process_par import Processor, process_par
from debias.utils.py_utils import get_containing_spans


class SquadAnnotator(Processor):
  """Builds `AnnotatedSquadParagraph` objects from the SQuAD paragraphs as loaded from JSON."""

  def __init__(self, port, intern=False, resplit=True):
    self.port = port
    self.intern = intern
    self.resplit = resplit
    if self.resplit:
      resplit = r"\p{Pd}\p{Po}\p{Ps}\p{Pe}\p{S}\p{Pc}"
      resplit = "([" + resplit + "]|'')"
      split_regex = r"(?![\.,'])" + resplit
      self.split_regex = regex.compile(split_regex)

  def process(self, data: Iterable[Dict]) -> List[AnnotatedSquadParagraph]:
    client = CoreNLPClient(port=self.port)
    out = []
    i = 0
    skipped_entries = []

    ### When we use our customized data, some of the entries cannot be properly parsed for
    ### this experiment, so we ignore these entries
    print("len data", len(data))
    print("data")

    newdata = []

    remove_idx = [57, 73, 80, 94, 95, 99, 319, 320, 324, 455, 465, 483, 484, 502, 509, 512, 523, 528, 541, 543, 584,
                  592, 596, 597, 602, 613, 615, 620, 637, 638, 642, 648, 650, 660, 662, 668, 675, 679, 723, 785, 789,
                  797, 798, 799, 839, 856, 858, 859, 860, 888, 897, 902, 905, 907, 908, 909, 941, 954, 956, 1031, 1049,
                  1056, 1058, 1065, 1077, 1082, 1092, 1103, 1107, 1116, 1117, 1120, 1121, 1127, 1130, 1133, 1135, 1140,
                  1238, 1320, 1329, 1330, 1337, 1339, 1343, 1363, 1365, 1367, 1368, 1389, 1393, 1399, 1400, 1413, 1414,
                  1435, 1456, 1529, 1594, 1603, 1608, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1626, 1629, 1631, 1632,
                  1633, 1634, 1637, 1639, 1653, 1689, 1756, 1811, 1862, 1877, 1926, 1975]
    #for i in remove_idx:
      #del data[i]

    '''
    counter = 0
    for para in data:
        if counter in remove_idx:
            print(counter)
            print(para)
            newdata.append(para)
        counter += 1
    '''
    print("Removing indices which can't be annotated from json file")

    #with open('../../data/squad-experiment1-bias-80-percent.json', 'w', encoding='utf8') as json_file:
    #   json.dump(newdata, json_file)

    # print(len(newdata))
    # newdata = tqdm(newdata, desc="annotate", ncols=80)


    for para in data:
      # print("\n********** process data *********")
      passage = para['context']

      offset = 0
      while passage[offset].isspace():
        offset += 1
      query_result = client.query_ner(passage[offset:])
      if not query_result:
        skipped_entries.append(i)
        i += 1
        continue
      annotations = query_result["sentences"]

      # print("annotations", annotations)
      if self.resplit:
        # We re-split the CORENLP tokens on some punctuation tags, since we need pretty aggressive tokenization
        # in ensure (almost) all answers span are contained within tokens
        words, pos, ner, inv = [], [], [], []
        sentence_lens = []
        on_len = 0
        for sentences in annotations:
          if len(sentences["tokens"]) == 0:
            raise RuntimeError()
          for token in sentences["tokens"]:
            p, n = token["pos"], token["ner"]
            s, e = (token["characterOffsetBegin"], token["characterOffsetEnd"])
            if len(token["originalText"]) != (e - s):
              # For some reason (probably due to unicode-shenanigans) the character offsets
              # we get make are sometime incorrect, we fix it here
              offset -= (e - s) - len(token["originalText"])
            s += offset
            e += offset

            w = passage[s:e]

            if w == "''" or w == '``':
              split = [w]
            else:
              split = [x for x in self.split_regex.split(w) if len(x) > 0]

            if len(split) == 1:
              words.append(w)
              pos.append(p)
              ner.append(n)
              inv.append((s, e))
            else:
              words += split
              ner += [n] * len(split)
              pos += ['SEP' if self.split_regex.match(x) else p for x in split]

              for w in split:
                inv.append((s, s + len(w)))
                s += len(w)
              if s != e:
                raise RuntimeError()

          sentence_lens.append(len(words) - on_len)
          on_len = len(words)
      else:
        raise NotImplementedError()

      inv = np.array(inv, np.int32)
      sentence_lens = np.array(sentence_lens, np.int32)
      if sum(sentence_lens) != len(words):
        raise RuntimeError()

      questions = []
      for question in para["qas"]:
        try:
          q_tokens = py_utils.flatten_list([x["tokens"] for x in client.query_tokenize(question["question"])["sentences"]])
        except:
          print("client.query_tokenize failure\n")
          skipped_entries.append(i)
          i += 1
          continue
        answer_spans = []
        answers_text = []
        for answer_ix, answer in enumerate(question['answers']):
          answer_raw = answer['text']
          answer_start = answer['answer_start']
          answer_stop = answer_start + len(answer_raw)
          # print(answer_raw)
          if passage[answer_start:answer_stop] != answer_raw:
            raise RuntimeError()
          word_ixs = get_containing_spans(inv, answer_start, answer_stop)
          answer_spans.append((word_ixs[0], word_ixs[-1]))
          answers_text.append(answer_raw)

        questions.append(SquadQuestion(
          question["id"], [x["word"] for x in q_tokens],
          answers_text, np.array(answer_spans, dtype=np.int32),
        ))

      out.append(AnnotatedSquadParagraph(
        passage, words, inv, pos, ner, sentence_lens, questions))
      i += 1

    print("SKIPPED ENTRIES:", skipped_entries)
    return out


def cache_docs(source_file, output_file, port, n_processes):
  annotator = SquadAnnotator(port, True)
  with open(source_file, "r") as f:
    docs = json.load(f)["data"]

  paragraphs = py_utils.flatten_list(x["paragraphs"] for x in docs)

  annotated = process_par(paragraphs, annotator, n_processes, 30, "annotate")

  with open(output_file, "wb") as f:
    pickle.dump(annotated, f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("source_file", help="SQuAD source file")
  parser.add_argument("output_file", help="Output pickle file to dump the annotated paragraphs")
  parser.add_argument("--port", type=int, default=9000, help="CoreNLP port")
  parser.add_argument("--n_processes", "-n", type=int, default=1)
  args = parser.parse_args()
  cache_docs(args.source_file, args.output_file, args.port, args.n_processes)


if __name__ == "__main__":
  main()
