import itertools

def getPermutedListOfSix():
    initialPermutation = itertools.product(range(2), repeat=6)
    resultList = [i for i in [*initialPermutation] if sum(i) > 0]
    print(resultList)

def getBCs():
    in_BCs, out_BCs = getPermutedListOfSix(), getPermutedListOfSix()
    return (in_BCs, out_BCs) # up and sides