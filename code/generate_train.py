import os
import pandas as pd

# Make sure the current working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

corpus_path = "../subtest_txt/gr3_paragraphs.txt"
target_path = "../data/gr3_test_to_score.csv"
scores_path = "../scores.xlsx"

sub_tests = []
with open(corpus_path,'r') as fp:
    sub_test = fp.readline()
    while sub_test:
        sub_tests.append(sub_test)
        sub_test = fp.readline()

# print(len(sub_tests))

df = pd.read_excel(scores_path,sheet_name="Gr3")
sub_test_number = df.columns[1:]

# stores column name to its subtest index and number of questions in the subtest
column_to_subtest = {}
count = 1
for number_range in sub_test_number:
    a,b = [int(i) for i in number_range.split()]
    column_to_subtest[number_range] = (count,b-a+1)
    count += 1

# Rearrange data and store each entry with student id, the text they read, and
# the percentage score they got on that sub-test
id =[]
text = []
score = []
for i in df.index:
    for number_range in sub_test_number:
        id.append(df["ID"][i])
        subtext_index, num_questions = column_to_subtest[number_range]
        text.append(sub_tests[subtext_index])
        score.append(df[number_range][i]/(num_questions*2))

data = {'id': id,
        'text': text,
        'score': score,}

target_df = pd.DataFrame(data, columns = ['id', 'text', 'score'])

print(target_df)

target_df.to_csv(target_path, index=False)
