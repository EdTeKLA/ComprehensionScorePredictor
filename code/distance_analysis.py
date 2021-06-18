from sentence_transformers import SentenceTransformer, util
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

grade = '5'
corpus_path = f"../subtest_txt/gr{grade}_paragraphs.txt"
questions_path = f"../subtest_txt/gr{grade}_questions.txt"
scores_path = "../Gates.ReadComp_By-Item_Gr3-5(CM).xlsx"

questions = []

with open(questions_path,'r') as fp:
    while True:
        sub_questions = []
        next_line = fp.readline()
        while next_line and next_line != '\n':
            q = next_line
            a = fp.readline()
            sub_question = q + ' ' + a
            sub_question = sub_question.replace('\n', '')
            sub_questions.append(sub_question)
            next_line = fp.readline()
        questions.append(sub_questions)
        if not next_line:
            break

sub_tests = []
with open(corpus_path,'r') as fp:
    sub_test = fp.readline()
    while sub_test:
        sub_tests.append(sub_test)
        # print(len(sub_test))
        sub_test = fp.readline()

model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')

#Change the length to 200
model.max_seq_length = 700

distances = []

for i, sub_questions in enumerate(questions):
    print("text:", sub_tests[i+1])
    sub_test_embed = model.encode(sub_tests[i+1], show_progress_bar=True)
    sub_questions_embed = model.encode(sub_questions)
    for j, question in enumerate(sub_questions_embed):
        print('question:', sub_questions[j])
        cos_sim = util.pytorch_cos_sim(question, sub_test_embed)
        print(cos_sim)
        distances.append(cos_sim)

df = pd.read_excel(scores_path)
score_per_question_means = []
for q_num in range(1,len(distances)+1):
    total_score = 0
    total_students = 0
    for i in df.index:
        if df[f"Gr{grade}.RC.Gates_"+"{:02d}".format(q_num)][i] == 1:
            total_score +=1
            total_students += 1
        if df[f"Gr{grade}.RC.Gates_"+"{:02d}".format(q_num)][i] in (0,2):
            total_students += 1
            
    score_per_question_means.append(total_score/total_students)

print(score_per_question_means)

cos_line, = plt.plot(np.arange(1,len(distances)+1),distances)#, label='cosine similarity')
score_line = plt.plot(np.arange(1,len(score_per_question_means)+1), score_per_question_means)#,lable = 'average score')
# np.arange(1,len(score_per_question_means)+1), 
# np.arange(1,len(distances)+1),
plt.legend()
plt.show()
# print("Max Sequence Length:", model.max_seq_length)

# # #Our sentences we like to encode
# # sentences = ['This framework generates embeddings for each input sentence',
# #     'Sentences are passed as a list of string.',
# #     'The quick brown fox jumps over the lazy dog.']

# sentences = sub_tests[:2]
# sentences.append("Why didn't Eliza's mother hear the baby?She was with Eliza.,She was with Andrew.,She was outside.,She was in her room.")
# #Sentences are encoded by calling model.encode()
# embeddings = model.encode(sentences, show_progress_bar=True)

# cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[2])
# print(cos_sim)
#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding.shape)
#     print("")