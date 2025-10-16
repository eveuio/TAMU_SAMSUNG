

def avgAmbientTemp(loadCurrentPercent):
    #TODO: lightly loaded transformers 10 degrees above ambient room temp, heavily loaded transformers 30 degrees above ambient room temp
    ambientTemp = 23.8889 + 10 + loadCurrentPercent*(40.556-23.8889)
    return ambientTemp
