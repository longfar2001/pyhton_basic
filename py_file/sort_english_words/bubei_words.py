#%%
import markdown
with open('E:/Markdown_file/English_word.md','r') as md_file:
    md_text=md_file.read()

word=[]
for line in md_text:
    words=line.strip()
    word.append(words)

word=[jinghao.replace("#","") for jinghao in word]  # 去掉#
word=['#' if str(s)=='' else s for s in word]

word_all=''.join(word).split("#")
word_all = [s for s in word_all if s]
word_all.sort(reverse=False)
word_all=[s for s in word_all if not s.isupper()]
word_all_all=list(set(word_all))
word_all_all.sort(reverse=False)
with open("E:/Markdown_file/English_word.txt","w") as file:
    for item in word_all_all:
        file.write(item + '\n')
#%%
import markdown
with open('E:/Markdown_file/阅读538.txt','r') as md_file:
    md_text=md_file.read()

word=[]
for line in md_text:
    words=line.strip()
    word.append(words)

word=[jinghao.replace("#","") for jinghao in word]  # 去掉#
word=['#' if str(s)=='' else s for s in word]

word_all=''.join(word).split("#")
word_all = [s for s in word_all if s]

word_all=[s for s in word_all if not s.isupper()]

with open("E:/Markdown_file/阅读538_word.txt","w") as file:
    for item in word_all:
        file.write(item + '\n')
# %%
with open('E:/Markdown_file/王陆.txt','r') as txt_file:
    txt_text=txt_file.read()
word=[]
for line in txt_text:
    words=line.strip()
    word.append(words)

word=[jinghao.replace("#","") for jinghao in word]  # 去掉#
word=['#' if str(s)=='' else s for s in word]

word_all=''.join(word).split("#")
word_all = [s for s in word_all if s]

word_all=[s for s in word_all if not s.isupper()]
word_all.sort(reverse=False)
word_all_wl=list(set(word_all))
word_all_wl.sort(reverse=False)
with open("E:/Markdown_file/王陆_word.txt","w") as file:
    for item in word_all_wl:
        file.write(item + '\n')
#%%