import json, re
import pickle
import random
import random
from random import randrange
import nltk

from random import shuffle

url = "https://raw.githubusercontent.com/Eva-Zhong/debias/main/data/squad-corenlp/squad-dev-experiment.json"

# response = requests.get(url).read().decode('utf-8')
# html = response.read().decode('utf-8')
# json_data = re.match('rdsData\((.*?)\)$', response.content).group(1)


# import urllib.request, json
# with urllib.request.urlopen(url) as json_data:
    # data = json.loads(json_data.read().decode('utf-8'))



# Opening JSON file
f = open('train-v1.1.json',)

# returns JSON object as
# a dictionary
data = json.load(f)

'''
1 - keep only interrogatives in the question
2 - shuffle words in questions
3 - shuffle words in paragraphs
'''
experiment = 1

# Iterating through the json
# list
# for i in data['emp_details']:
    # print(i)


interrogatives =["where","who","why","when","which","where","how"]

for i in range(len(data['data'])):
    entry = data['data'][i]
    # print(entry['paragraphs'])
    for j in range(len(entry['paragraphs'])):
        # print("new p")
        # print(p)
        p = entry['paragraphs'][j]
        # print(p)
        # keep only interrogative words in question
        if experiment == 1:
            ### Adjust the perentage of bias in the document
            bias_percentage = 0.6

            for k in range(len(p['qas'])):
                randomfloat = random.random()
                qa = p['qas'][k]
                if randomfloat < bias_percentage:
                    question = qa['question']
                    words = question.split()
                    newwords = [w for w in words if w.lower() in interrogatives]
                    newquestion = " ".join(newwords)
                    data['data'][i]['paragraphs'][j]['qas'][k]['question'] = newquestion

        # Shuffle words in question
        elif experiment == 2:
            for k in range(len(p['qas'])):
                qa = p['qas'][k]
                question = qa['question']
                words = question.split()
                random.shuffle(words)
                newquestion = " ".join(words)
                data['data'][i]['paragraphs'][j]['qas'][k]['question'] = newquestion

        # Shuffle words in context
        #1. Locate correct answer
        #2. Remove answer text
        #3. Shuffle the rest
        #4. insert answer string into random positions; update answer_position
        elif experiment == 3:
            context = p['context']
            true_answers = []

            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for a in answers:
                    idx = a['answer_start']
                    cur_qas_answer.append(a['text'])
                    context = context[:idx] + context[idx+len(cur_qas_answer):]
                true_answers.append(cur_qas_answer)

            words = context.split()
            random.shuffle(context)

            # Insert answer into random positions
            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for q in range(len(answers)):
                    a = true_answers[k][q]
                    new_idx_in_list = randrange(len(words))
                    words = words[:new_idx_in_list] + [a] + words[new_idx_in_list:]
                    c = " ".join(words[:(new_idx_in_list+1)])

            newcontext = " ".join(words)
            data['data'][i]['paragraphs'][j]['context'] = newcontext

            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for q in range(len(answers)):
                    a = true_answers[k][q]
                    new_idx_in_context = newcontext.find(a)
                    # print(newcontext[new_idx_in_context:(new_idx_in_context + 50)])
                    data['data'][i]['paragraphs'][j]['qas'][k]['answers'][q]['answer_start'] = new_idx_in_context

        # Shuffle sentences in context
        elif experiment == 4:
            context = p['context']
            context = context.replace(". ", ".")
            data['data'][i]['paragraphs'][j]['context'] = context

            true_answers = []

            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                newcontext = context

                for q in range(len(answers)):
                    a = answers[q]
                    idx = a['answer_start']
                    text = a['text']


                    if ("Wernher" in text):
                        print(text, text in context)
                        print("*** ", text)
                        print(context)

                    while ". " in text:
                        # print(". in answer", text)
                        newtext = text.replace(". ", ".")
                        data['data'][i]['paragraphs'][j]['qas'][k]['answers'][q]['text'] = newtext
                        if ("Wernher" in text):
                            print(newtext, newtext in context)
                            print(context)
                            print("*** ", text)
                        text = newtext

                    cur_qas_answer.append(text)
                true_answers.append(cur_qas_answer)

            # Shuffle sentences
            sentences = context.split(". ")
            # if "Wernher" in context:
                # print("Wernher", newcontext)
            random.shuffle(sentences)
            newcontext = ". ".join(sentences)+"."
            data['data'][i]['paragraphs'][j]['context'] = newcontext

            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for q in range(len(answers)):
                    a = true_answers[k][q]
                    new_idx_in_context = newcontext.find(a)
                    data['data'][i]['paragraphs'][j]['qas'][k]['answers'][q]['answer_start'] = new_idx_in_context


        # Shuffle sentences in context
        elif experiment == 5:
            context = p['context']
            sentences = context.split(". ")
            newsentences = sentences

            true_answers = []
            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar

                for aa in range(len(answers)):
                    a = answers[aa]
                    idx = a['answer_start']
                    text = a['text']
                    cur_qas_answer.append(text)
                    if (aa > 0 and text in context):
                        continue

                    for ss in range(len(sentences)):
                        s = sentences[ss]
                        if text in s:
                            idx = s.find(text)
                            rest = s[:idx] + s[idx+len(text):]

                            words = rest.split()
                            if (len(words) > 0):
                                random.shuffle(words)
                                new_idx_in_list = randrange(len(words))
                                words = words[:new_idx_in_list] + [text] + words[new_idx_in_list:]
                                newsentence = " ".join(words)
                                sentences[ss] = newsentence
                        else:
                            words = s.split()
                            if (len(words) > 0):
                                random.shuffle(words)
                                newsentence = " ".join(words)
                                sentences[ss] = newsentence

                true_answers.append(cur_qas_answer)

            newcontext = ". ".join(sentences)
            data['data'][i]['paragraphs'][j]['context'] = newcontext
            # print(newcontext)

            # Shuffle words in a sentence except for the last word

            '''
            sentences = context.split(". ")
            newsentences = []
            for s in sentences:
                words = s.split()
                l = len(words)
                if l > 0:
                    words_shuffle = words[:l]
                    random.shuffle(words_shuffle)
                    newsentence = words_shuffle + [words[-1]]
                    newsentences.append(" ".join(newsentence))
            '''

            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for q in range(len(answers)):
                    a = true_answers[k][q]
                    new_idx_in_context = newcontext.find(a)
                    # print(newcontext[new_idx_in_context:(new_idx_in_context + 50)])
                    data['data'][i]['paragraphs'][j]['qas'][k]['answers'][q]['answer_start'] = new_idx_in_context

       # Replace words with other words of the same POS Tag
        elif experiment == 6:
            context = p['context']
            tokens = nltk.word_tokenize(context)
            true_answers = []
            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for a in answers:
                    idx = a['answer_start']
                    cur_qas_answer.append(a['text'])
                true_answers.append(cur_qas_answer)

            # Shuffle words in a sentence except for the last word
            sentences = context.split(". ")
            newsentences = []
            for s in sentences:
                words = s.split()
                l = len(words)
                if l > 0:
                    words_shuffle = words[:l]
                    random.shuffle(words_shuffle)
                    newsentence = words_shuffle + [words[-1]]
                    newsentences.append(" ".join(newsentence))

            newcontext = ". ".join(newsentences)
            data['data'][i]['paragraphs'][j]['context'] = newcontext
            print(newcontext)
            print(context)

            for k in range(len(p['qas'])):
                cur_qas_answer = []
                qa = p['qas'][k]
                answers = qa['answers']  # only need to process according to the first answer; all are similar
                for q in range(len(answers)):
                    a = true_answers[k][q]
                    new_idx_in_context = newcontext.find(a)
                    # print(newcontext[new_idx_in_context:(new_idx_in_context + 50)])
                    data['data'][i]['paragraphs'][j]['qas'][k]['answers'][q]['answer_start'] = new_idx_in_context




with open("train-experiment1-60-percent-bias.json", "w") as write_file:
    json.dump(data, write_file)

#filename = 'train-experiment1-80-percent-bias.pkl'
#with open(filename, 'wb') as f:
#    pickle.dump(data["data"], f)

#with open(filename, "rb") as f:
#    pickle.load(f)
