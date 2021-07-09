'''
This short script generate a custom embedding torchtext compatible format.
The embedding maps numbers as a string to their corresponding value as a float.
'''

text = ''
for i in range(100):
    text += f'{i} {i} 1 1\n'
    print(text)

with open("number.txt","w") as fp:
    fp.write(text)
    