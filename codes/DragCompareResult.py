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

global tex

global dirs
dirs = []

def showResults2():
    
    print("selected ", dirs)
    if(len(dirs)>0):
        res = ResultsComparison(dirs)
        res.show()
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
    domains = log.split(' ')[0].split('_')
    if(len(domains)>=2):
        [tr,test]=domains
    else:
        tr = domains[0]
        test = domains[0]
    x = model.ljust(15)+dann.ljust(10)+'Train - '+tr.ljust(len('abcdefghi  '))+'Tester - '+test
    tex.insert(END, x + '\n')

def refreshList():
    dirs.clear()
    tex.delete('1.0', END)


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
