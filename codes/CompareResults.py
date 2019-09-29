from tkinter import *
import os,numpy as np
from ResultAnalyser import ResultsComparison
from collections import Counter

def showResults():
    idx = [(v.get()==1) for v in var]
    selected = list(np.array(log_names)[idx])
    
    idx_dann = [(s.split(' ')[0] in list(np.array(label_text)[idx])) for s in log_names_dann ]
    selected_dann = list(np.array(log_names_dann)[idx_dann])
    print("selected ", selected)
    print("dann  ", selected_dann)
    if(len(selected)>0):
        if(len(selected_dann)>0):
            res = ResultsComparison(selected, selected_dann)
        else:
            res = ResultsComparison(selected)
        res.show()
    else:
         print("None selected ")   
def refreshList(var,labels,checkbuttons):
    log_names = os.listdir(log_dir)
    log_names_dann = os.listdir(log_dir_dann)
    label_text_dann = [x.split(' ')[0] for x in os.listdir(log_dir)]
    label_text = [x.split(' ')[0] for x in os.listdir(log_dir)]
    colors = color_similarity(label_text)
    var.clear()
    labels.clear()
    checkbuttons.clear()
    for i,text in enumerate(label_text):
        var1 = IntVar()
        lb = Label(window,text=text,bg=colors[i],font =('Verdana', 15),width=40).grid(column=1,row=i)
        lbt = Checkbutton(window, bg="white",variable=var1).grid(column=0,row=i)
        labels.append(lb)
        checkbuttons.append(lbt)
        var.append(var1)
        del lb,lbt,var1
def color_similarity(label_text):
    labs = [x.split('_')[0]+'_'+x.split('_')[1] if(len(x.split('_'))>1) else x for x in label_text ]
    col = list(np.arange(255,155,-100//len(Counter(labs))))
    label_color = np.array(['#%02x%02x%02x' % (255,255,255)]*len(labs))
    for k,c in zip(list(Counter(labs).keys()),col):
        idx = [l==k for l in labs]
        label_color[idx] = '#%02x%02x%02x' % (c, c, c)
    return list(label_color)


log_dir = '../../Adversarial Heart Sound Results/logs/'
log_dir_dann = log_dir+'dann/'

window = Tk()
window.configure(background='black')
window.title("Welcome to LikeGeeks app")
window.minsize(300, 300) 

log_names = os.listdir(log_dir)
log_names_dann = os.listdir(log_dir_dann)

print(log_names)
print(log_names_dann)


label_text = [x.split(' ')[0] for x in os.listdir(log_dir)]

var = []
labels = []
checkbuttons = []
refreshList(var,labels,checkbuttons)
button = Button(window, text='Show Result Comparison ', fg='white',bg='grey',
                font=('Verdana',15), command=showResults)
refreshButton = Button(window, text='Refresh', fg='white',bg='grey',
                font=('Verdana',12), command=lambda:refreshList(var,labels,checkbuttons))
button.grid(column=1)
refreshButton.grid(column=1)
window.mainloop()