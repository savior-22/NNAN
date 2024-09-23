a = [
    {"Jodie1": "123"},
    {"Jodie2": "456"},
    {"Jodie3": "789"},
    ]
with open('1.txt', 'w') as f:
    for i in range(len(a)):
        for key, values in a[i].items():
            print(key+","+values+"\r")
            f.write(key+","+values+"\r")