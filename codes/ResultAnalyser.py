import numpy as np, matplotlib.pyplot as plt, pandas as pd,os

class Results():
    def __init__(self,log_name):
        self.log_name = log_name 
        self.df = None
        self.dft = None # Tuned result
        self.tune = False
        self.log_dir = '../../Adversarial Heart Sound Results/logs/'
        self.metrics = ['val_macc','val_F1','val_precision','val_sensitivity','val_specificity']
        self.read()
    def read(self):
        os.listdir(self.log_dir)
        wow = [x for x in os.listdir(self.log_dir) if self.log_name in x]
        how = [x.split(' ')[0] for x in wow]
        now = list(set(how))
        now.sort()
        print("Log not tuned " ,wow[how.index(now[0])])
        self.df = pd.read_csv(self.log_dir+wow[how.index(now[0])]+'/training.csv')
        self.df.sort_values(by=['val_macc','val_F1'],ascending=False,inplace = True)
        self.df = dict(self.df.iloc[0][self.metrics])
        if(len(now)>1):
            print("Tuned log ",wow[how.index(now[1])])
            self.dft = pd.read_csv(self.log_dir+wow[how.index(now[1])]+'/training.csv')
            self.dft.sort_values(by=['val_macc','val_F1'],ascending=False,inplace = True)
            self.dft = dict(self.dft.iloc[0][self.metrics])
            self.tune = True
    def show(self,width = 0.35,figsize=(8,5)):
        x = np.arange(len(self.metrics))
        fig, ax = plt.subplots(figsize=figsize)
        labels = list(self.df.values())
        rect1 = ax.bar(x-width/2,labels,width,label='Results')
        self.autolabel(rect1,ax)
        title = "Trainer: " +self.log_name.split('_')[0]+"\n Tester: "+ self.log_name.split('_')[1]
        if(self.tune):
            labels2 = list(self.dft.values())
            rect2 = ax.bar(x+width/2,labels2,width,label='Tuned')
            self.autolabel(rect2,ax)
            title = title + "\nWith Tuning"
        ax.set_ylabel('Scores')
        ax.set_xticks(x)
        ax.set_title(title)
        ax.set_xticklabels(self.metrics)
        ax.legend()
        fig.tight_layout()
        plt.show()
        
    def autolabel(self,rects,ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


###              """class to hold single Result"""
class Result():
    def __init__(self,log):
        self.log_dir = '../../Adversarial Heart Sound Results/logs/'
        self.metrics = ['val_macc','val_F1','val_precision','val_sensitivity','val_specificity']
        self.log_name = log.split(' ')[0]
        self.trainer = self.log_name.split('_')[0]
        self.tester = self.log_name.split('_')[1]
        self.df = pd.read_csv(self.log_dir+log+'/training.csv')
        self.df.sort_values(by=['val_macc','val_F1'],ascending=False,inplace = True)
        self.df = dict(self.df.iloc[0][self.metrics])

## Compare any number of result 
class ResultsComparison():
    def __init__(self,logs):
        self.log_dir = '../../Adversarial Heart Sound Results/logs/'
        self.metrics = ['val_macc','val_F1','val_precision','val_sensitivity','val_specificity']
        self.logs = logs
        self.data = []
        self.read()
    def read(self):
        self.data = [Result(l) for l in self.logs]
        print(len(self.data))
    def show(self,width = 0.35,figsize=(8,5)):
        x = np.arange(len(self.metrics))
        print(x)
        fig, ax = plt.subplots(figsize=figsize)
        plot_number = len(self.data)
        width = .8/plot_number
        idx = np.arange(-(plot_number-1)/2,(plot_number-1)/2+1,1)
        print(idx)
        for i,d in enumerate(self.data):
            print(d.log_name)
            labels = list(d.df.values())
            rect1 = ax.bar(x+idx[i]*width,labels,width,label=d.log_name)
            self.autolabel(rect1,ax)
            #self.autolabel(rect1,ax)
        ax.set_xticks(x)
        ax.set_title("Result comparison ")
        ax.set_xticklabels(self.metrics)
        ax.legend()
        fig.tight_layout()
        plt.show()
    def autolabel(self,rects,ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')