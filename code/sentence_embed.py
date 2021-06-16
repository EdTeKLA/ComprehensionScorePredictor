from sentence_transformers import SentenceTransformer, util

grade = '3'
corpus_path = f"../subtest_txt/gr{grade}_paragraphs.txt"
model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')

#Change the length to 200
model.max_seq_length = 400

print("Max Sequence Length:", model.max_seq_length)

sub_tests = []
with open(corpus_path,'r') as fp:
    sub_test = fp.readline()
    while sub_test:
        sub_tests.append(sub_test)
        # print(len(sub_test))
        sub_test = fp.readline()

# #Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']

sentences = sub_tests[:2]
sentences.append("Why didn't Eliza's mother hear the baby?She was with Eliza.,She was with Andrew.,She was outside.,She was in her room.")
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences, show_progress_bar=True)

cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[2])
print(cos_sim)
#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding.shape)
#     print("")