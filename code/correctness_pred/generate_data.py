import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


# Make sure the current working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

grade = "3"
target_path = f"../../data/question_correctness/.csv"
scores_path = "../../Gates.ReadComp_By-Item_Gr3-5(CM).xlsx"
corpus_path = f"../../subtest_txt/gr{grade}_paragraphs.txt"

sub_tests = []
with open(corpus_path,'r') as fp:
    sub_test = fp.readline()
    while sub_test:
        sub_tests.append(sub_test)
        sub_test = fp.readline()


df = pd.read_excel(scores_path)
sub_test_number = df.columns[1:]

print(sub_test_number)

questions_ranges = [(1,5), (6,8), (9,13), (14,16), (17,21), (22,27), (28,30),
                   (31,35), (36,40), (41,43), (44,48)]

model = SentenceTransformer('paraphrase-mpnet-base-v2')

#Change the length to 200
model.max_seq_length = 500

sub_tests_embed = model.encode(sub_tests, show_progress_bar=True)

print(sub_tests_embed.shape)