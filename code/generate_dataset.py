import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics

# Make sure the current working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

target_path = "../data/gr3_score.csv"
scores_path = "../Gates.ReadComp_By-Item_Gr3-5(CM).xlsx"
corpus_path = "../subtest_txt/gr3_paragraphs.txt"


sub_tests = []
with open(corpus_path,'r') as fp:
    sub_test = fp.readline()
    while sub_test:
        sub_tests.append(sub_test)
        sub_test = fp.readline()


df = pd.read_excel(scores_path)
sub_test_number = df.columns[1:]

questions_ranges = [(1,5), (6,8), (9,13), (14,16), (17,21), (22,27), (28,30),
                   (31,35), (36,40), (41,43), (44,48)]

score_data = {}
for i in range(len(questions_ranges)):
    score_data[i+1] = []
    score_data["skip"+str(i+1)] = []

df = pd.read_excel(scores_path)
for i in df.index:
    for q_index, q_range in enumerate(questions_ranges):
        total_score = 0
        total_skip = 0
        for j in range(q_range[0],q_range[1]):
            if df["Gr3.RC.Gates_"+"{:02d}".format(j)][i] not in [0,2]:
                total_score +=1
            if df["Gr3.RC.Gates_"+"{:02d}".format(j)][i] == 2:
                total_skip += 1
        score_data[q_index+1].append(total_score/(q_range[1]-q_range[0]+1))
        score_data["skip"+str(q_index+1)].append(total_skip/(q_range[1]-q_range[0]+1))

points = []
for i in range(len(questions_ranges)):
    points.append(sum(score_data[i+1])/len(score_data[i+1]))

# Showing skip percentage during tests
# for i in range(len(questions_ranges)):
#     points.append(statistics.mean(score_data["skip"+str(i+1)]))

# plt.figure()
# plt.plot(points,"o-")
# plt.ylabel("Skips %")
# plt.xlabel("sub-test number")
# plt.xticks(np.arange(0, 12, step=1))
# plt.show()

# Rearrange data and store each entry with student id, the text they read, and
# the percentage score they got on that sub-test
text = []
score = []
for i in range(len(score_data[1])):
    for subtext_index in range(1,len(questions_ranges)+1):
        text.append(sub_tests[subtext_index])
        score.append(score_data[subtext_index][i])

data = {'text': text,
        'score': score,}

target_df = pd.DataFrame(data, columns = ['text', 'score'])

print(target_df)

target_df.to_csv(target_path, index=False)