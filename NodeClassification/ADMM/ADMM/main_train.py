import os
import numpy as np
learning_rate = [0.001, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095, 0.0098]
adj_index = np.arange(60).tolist() #eg. 60. before that you need to find the largest number in your adj.npy
                                   #and change 60 to the number
highest_acc = 0
acc_list = []
for j in range(len(adj_index)):
    index = adj_index[j]
    # linux
    #os.system('rm '+"log"+str(index)+".txt")
    #windows
    os.system('del ' + "log" + str(index) + ".txt")
for i in range(len(learning_rate)):
    for j in range(len(adj_index)):
        lr = learning_rate[i]
        index = adj_index[j]
        if os.path.exists('adj_'+ str(index)+'.npy'):
            os.system("python train.py"
                      +" --learning_rate="+str(lr)
                      +" --adj_index="+str(index)
                      +" >>log"+str(index)+ "_"+str(lr)+".txt")
            f = open("log" + str(index) + "_"+str(lr)+".txt")
            for line2 in f:
                if "Finally Test set results" in line2:
                    res = line2.split()
                    acc_list.append(float(res[7]))
                    if float(res[7]) > highest_acc:
                        highest_acc = float(res[7])
        else:
            print("no log and npy file in this adj_"+str(index))

print("acc list: ",acc_list)
print("highest accuracy only train with pruned adjacency: ", highest_acc)
