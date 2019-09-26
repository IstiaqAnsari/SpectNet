from tkinter import *
import os,numpy as np
from ResultAnalyser import ResultsComparison

def showResults():
    idx = [(v.get()==1) for v in var]
    selected = list(np.array(log_names)[idx])
    print(selected)
    res = ResultsComparison(selected)
    res.show()
    window.destroy()

log_dir = '../../Adversarial Heart Sound Results/logs/'

window = Tk()
window.configure(background='black')
window.title("Welcome to LikeGeeks app")
window.minsize(300, 300) 

log_names = os.listdir(log_dir)
label_text = [x.split(' ')[0] for x in os.listdir(log_dir)]
var = []
labels = []
checkbuttons = []
for i,text in enumerate(label_text):
    var1 = IntVar()
    lb = Label(window,text=text,bg="white",font =('Verdana', 15),width=40)
    lb.grid(column=1,row=i)
    lbt = Checkbutton(window, bg="white",variable=var1)
    lbt.grid(column=0,row=i)
    labels.append(lb)
    checkbuttons.append(lbt)
    var.append(var1)
    del lb,lbt,var1
    
button = Button(window, text='Show Result Comparison ', fg='white',bg='grey',
                font=('Verdana',15), command=showResults)
button.grid(column=1)
window.mainloop()