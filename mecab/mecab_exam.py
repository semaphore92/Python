import MeCab

m = MeCab.Tagger()
out = m.parse("안녕하세요")
print(out)