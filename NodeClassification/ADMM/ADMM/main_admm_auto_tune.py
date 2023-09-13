import os
import numpy as np
learning_rate = [0.01, 0.001]
prune_ratio = 30
ADMM_times = [2,3,4,5,6,7]
Total_epochs = [10,30,40,50,60]
target_accuracy = 0.76
count = 0
highest_acc = 0

for i in range(len(learning_rate)):
    for j in range(len(ADMM_times)):
        for k in range(len(Total_epochs)):
            lr = learning_rate[i]
            admm = ADMM_times[j]
            epoch = Total_epochs[k]
            #linux
            #os.system('rm '+"log"+str(count)+".txt")
            #windows
            os.system('del '+"log"+str(count)+".txt")
            os.system("python train-auto-admm-tuneParameter.py"
                      +" --target_acc="+str(target_accuracy)
                      +" --prune_ratio="+str(prune_ratio)
                      +" --count=" + str(count)
                      +" --learning_rate="+str(lr)
                      +" --ADMM="+str(admm)
                      +" --epochs="+str(epoch)
                      +" >>log"+str(count)+".txt")
            f = open("log" + str(count) + ".txt")
            for line2 in f:
                if "Finally Test set results" in line2:
                    res = line2.split()
                    if float(res[7]) > highest_acc:
                        highest_acc = float(res[7])
            count+=1

print("highest accuracy only train with pruned adjacency + weights: ", highest_acc)
