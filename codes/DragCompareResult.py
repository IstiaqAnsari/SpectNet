import sys
if sys.version_info[0] == 2:
    from Tkinter import *
else:
    from tkinter import *
from TkinterDnD2 import *
from tkinter import *
import os,numpy as np
from ResultAnalyser import ResultsComparison
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
global tex

global dirs
dirs = []

def showResults2():
    
    print("selected ", dirs)
    if(len(dirs)>0):
        res = ResultsComparison(dirs)
        fig,ax = plt.subplots(figsize=(8,5))
        res.show(fig=fig,ax=ax,shownow=(len(dirs)>1))
        if(len(dirs)==1):
            fig2,ax2 = plt.subplots(figsize=(8,5))
            lossplot(dirs[0],fig=fig2,ax=ax2,limy=(0,1))
    else:
         print("None selected ") 
def newEntry(event):
    s = (event.data)
    dirs.append(s[1:-1])
    updateTextList(s[1:-1])
def updateTextList(x):
    model = x[x.index('logs')+len('logs/'):].split('/')[0]
    dann = 'dann' if 'dann' in x else  'No_dann'
    log = x.split('/')[-1]
    [tr,test]=log.split(' ')[0].split('_')
    x = model.ljust(15)+dann.ljust(10)+'Train - '+tr.ljust(len('abcdefghi  '))+'Tester - '+test

    tex.insert(END, x + '\n')
def refreshList():
    dirs.clear()
    tex.delete('1.0', END)
    

def lossplot(filepath,fig=None,ax=None,figsize=(8,5),limy=None,params = [],plots = []):
    
    filepath = os.path.join(filepath,'training.csv')
    if(os.path.isfile(filepath)):
        df = pd.read_csv(filepath)
        df.sort_values(by='epoch',ascending=True,inplace = True)
        if(fig==None):
            fig, ax = plt.subplots(figsize=figsize)
        losses = [x for x in df.keys() if('loss' in x)]
        for x in losses:
            ax.plot(np.array(df[x]),label=x)
        for x in plots:
            ax.plot(df[x],label=x)
        ax.set_xlabel('Epochs',fontdict={'size':12})
        ax.legend()
        if(limy is not None):
            x1,x2,y1,y2 = plt.axis()
            ax.axis((x1,x2,limy[0],limy[1]))
        plt.show()
        
    else:
        print("NO csv file found")


window = TkinterDnD.Tk()
window.configure(background='black')
window.title("Welcome to LikeGeeks app")
window.minsize(500, 500) 


window.drop_target_register(DND_ALL)
window.dnd_bind('<<Drop>>',func = newEntry)


tex = Text(window)

button = Button(window, text='Show Result Comparison ', fg='white',bg='grey',
                font=('Verdana',15), command=showResults2)
refreshButton = Button(window, text='Refresh', fg='white',bg='grey',
                font=('Verdana',12),command=refreshList)

button.grid(row=1,column=0)
refreshButton.grid(row=2,column = 0)

tex.config(font=('Arial', 12))
tex.grid(row=3)

window.mainloop()
