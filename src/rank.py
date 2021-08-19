'''
rank.py ranks the models by the best validation accuracy
'''

import os

# get all the log files
models = []
for path, dirs, files in os.walk("../models"):
    for f in files:
        if f.endswith(".txt"):
            models.append(os.path.join(path,f))

# search all the log files for model types
categories = set()
for model in models:
    with open(model) as f:
        log = f.readlines()
        for line in log:
            loc = line.find("Model Name: ")
            if loc > -1:
                categories.add(line[loc+12:].rstrip())
                break

# rank models in each type
categories = list(categories)
trained_models = [[] for i in range(len(categories))]
for model in models:
    with open(model) as f:
        log = f.readlines()
        model_name = ""
        session_name = ""
        acc = 0
        for line in log:
            # get the model type
            loc = line.find("Model Name: ")
            if loc > -1:
                model_name = line[loc+12:].rstrip()
            # get the session name
            loc = line.find("Session Name: ")
            if loc > -1:
                session_name = line[loc+14:].rstrip()
            # get the final best accuracy
            loc = line.find("Best Accuracy: ")
            if loc > -1:
                acc = float(line[loc+15:-5].rstrip())

        for i,model in enumerate(categories):
            if model_name == model:
                trained_models[i].append((session_name,acc))

# print nicely
for i,model in enumerate(categories):
    print(model)
    sorted_models = sorted(trained_models[i],key = lambda x: x[1],reverse=True)
    for tm in sorted_models:
        print(tm)
