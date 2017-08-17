import cPickle

ranges = cPickle.load(open("Chars.p", "r"))
notPickle = open('Chars.txt', 'w')

for (imagepath, lpranges) in ranges:
    notPickle.write(imagepath + ": \n")
    for r in lpranges:
        notPickle.write(','.join(str(r)) + ";\n")

notPickle.close()