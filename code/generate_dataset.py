import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics

# Make sure the current working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

graph_subtest_mean = False
graph_subtest_skips = False
generate = True
generate_no_skill = False
grade = "3"
target_path = f"../data/gr{grade}/gr{grade}_score.csv"
scores_path = "../Gates.ReadComp_By-Item_Gr3-5(CM).xlsx"
corpus_path = f"../subtest_txt/gr{grade}_paragraphs.txt"


sub_tests = []
with open(corpus_path,'r') as fp:
    sub_test = fp.readline()
    while sub_test:
        sub_tests.append(sub_test)
        sub_test = fp.readline()


df = pd.read_excel(scores_path)
sub_test_number = df.columns[1:]

#gr3
questions_ranges = [(1,5), (6,8), (9,13), (14,16), (17,21), (22,27), (28,30),
                   (31,35), (36,40), (41,43), (44,48)]

#gr4
# questions_ranges = [(1,4), (5,8), (9,13), (14,16), (17,19), (20,25), (26,29),
#                    (30,34), (35,39), (40,44), (45,48)]

# # # gr5
# questions_ranges = [(1,3), (4,6), (7,11), (12,14), (15,19), (20,25), (26,30),
#                    (31,34), (35,38), (39,43), (44,48)]

score_data = {}
for i in range(len(questions_ranges)):
    score_data[i+1] = []
    score_data["skip"+str(i+1)] = []

df = pd.read_excel(scores_path)

for i in df.index:
    for q_index, q_range in enumerate(questions_ranges):
        total_score = 0
        total_skip = 0
        for j in range(q_range[0],q_range[1]+1):
            if df[f"Gr{grade}.RC.Gates_"+"{:02d}".format(j)][i] == 1:
                total_score +=1
            if df[f"Gr{grade}.RC.Gates_"+"{:02d}".format(j)][i] == 2:
                total_skip += 1
        
        score_data[q_index+1].append(total_score/(q_range[1]-q_range[0]+1))
        score_data["skip"+str(q_index+1)]\
        .append(total_skip/(q_range[1]-q_range[0]+1))

if graph_subtest_mean:
    points = []
    for i in range(len(questions_ranges)):
        # points.append(sum(score_data[i+1])/len(score_data[i+1]))
        points.append(statistics.median(score_data[i+1]))
    plt.figure()
    plt.plot(points,"o-")
    plt.ylabel("mean score %")
    plt.xlabel("sub-test number")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 12, step=1))
    plt.show()

# Showing skip percentage during tests
if graph_subtest_skips:
    skip_points = []
    for i in range(len(questions_ranges)):
        skip_points.append(statistics.mean(score_data["skip"+str(i+1)]))

    plt.figure()
    plt.plot(skip_points,"o-")
    plt.ylabel("Skips %")
    plt.xlabel("sub-test number")
    plt.xticks(np.arange(0, 12, step=1))
    plt.show()

# Rearrange data and store each entry with student id, the text they read, and
# the percentage score they got on that sub-test
if generate:
    text = []
    score = []
    for i in range(len(score_data[1])):
        for subtext_index in range(1,len(questions_ranges)+1):
            text.append(sub_tests[subtext_index])
            score.append(score_data[subtext_index][i])

    data = {'text': text,
            'score': score,}
    
    if generate_no_skill:
        target_df = pd.DataFrame(data, columns = ['text', 'score'])
        target_df.to_csv(target_path, index=False)

    else:
        df2 = pd.read_excel("../data/gr3/gr3_features.xlsx")
        # iterate through every variable except for id and raw score
        for col in df2.columns[1:-1]:
            values = []
            for i in df2.index:
                for _ in range(11):
                    values.append(df2[col][i])# each student has 11 different
                    # subtest with the same skill variables
            data[col] = values
        for key in data.keys():
            print(key,len(data[key]))
        
        data['skills'] = []
        print(df2.columns)
        for i in df2.index:
            skill = []
            for col in df2.columns[1:-1]:
                skill.append(df2[col][i])
            for _ in range(11):
                data["skills"].append(skill)
            
        target_df = pd.DataFrame(data, columns = ['text', 'score', 'skills'])
        target_df.to_csv(target_path, index=False)


