import numpy as np, matplotlib.pyplot as plt, pandas as pd,os

class Results():
    def __init__(self,log_name):
        self.log_name = log_name 
        self.df = None
        self.dft = None # Tuned result
        self.tune = False
        self.log_dir = '../../Heartnet_Results/logs/'
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
        self.macc = None
        self.f1 = None
        for x in self.df.keys():
            if('macc' in x):
                self.macc = x
            if('F1' in x):
                self.f1 = x
        
        self.df.sort_values(by=self.macc if(self.f1 is None) else [self.macc,self.f1],ascending=False,inplace = True)
        self.df = dict(self.df.iloc[0][self.metrics])
        if(len(now)>1):
            print("Tuned log ",wow[how.index(now[1])])
            self.dft = pd.read_csv(self.log_dir+wow[how.index(now[1])]+'/training.csv')
            self.dft.sort_values(by=self.macc if(self.f1 is None) else [self.macc,self.f1],ascending=False,inplace = True)
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
    def __init__(self,log,dann=False,find = False):
        #find is a bool, if true then the log name is searched using the log when the log is only the fold name
        # not the full path . result comparison calls it with the full path. so no need of find variable 
        # but must in model loading which is set in heartnet.getattentionModel
        self.log_dir = '../../Heartnet_Results/logs/'
        self.metrics = ['macc','F1','precision','sensitivity','specificity']
        self.log_name = log.split(' ')[0].split('/')[-1]
        if(dann):
            self.log_dir = self.log_dir+'dann/'
            self.log_name = self.log_name + " dann"
        if(find):
            self.metrics.append('model_path')
            for x in os.listdir(self.log_dir):
                if log in x:
                    log = x
                    break
        self.trainer = self.log_name.split('_')[0]
        self.tester = self.log_name.split('_')[1]
        if('logs' in log): # Means full path given
            self.log_name = log.split('/')[-1].split(' ')[0]
            if('dann' in log):
                self.log_name = self.log_name + " dann"
            self.df = pd.read_csv(log+'/training.csv')
        else:
            self.df = pd.read_csv(self.log_dir+log+'/training.csv')
        self.macc = None
        self.f1 = None
        for x in self.df.keys():
            if('macc' in x):
                self.macc = x
            if('F1' in x):
                self.f1 = x
        self.metdic = {}
        for m in self.metrics:
            for k in self.df.keys():
                if(m in k):
                    self.metdic[m] = k
#         print(list(self.metdic.values()))
        self.df.sort_values(by=self.macc if(self.f1 is None) else [self.macc,self.f1],ascending=False,inplace = True)
#         print(self.df.iloc[0])
        self.df = (self.df.iloc[0][list(self.metdic.values())])
        print(self.df)
#         self.df = dict(self.df.iloc[0]['val_macc'])
## Compare any number of result 
class ResultsComparison():
    def __init__(self,logs,logs_dann=None):
        if(logs_dann is not None):
            self.log_dir_dann = '../../Heartnet_Results/logs/dann/'
        self.log_dir = '../../Heartnet_Results/logs/'
        self.metrics = ['macc','F1','precision','sensitivity','specificity']
        self.logs = logs
        self.logs_dann = logs_dann
        self.data = []
        self.read()
        self.baseline = pd.read_csv('../miscellaneous/baseline_results.csv')
    def read(self):
        self.data = [Result(l) for l in self.logs]
        if(self.logs_dann is not None):
            self.data = self.data+[Result(l,True) for l in self.logs_dann]
    def show(self,fig=None,ax=None,width = 0.35,figsize=(8,5),shownow=True):
        x = np.arange(len(self.metrics))
        if(fig==None):
            fig, ax = plt.subplots(figsize=figsize)
        plot_number = len(self.data)
        width = .8/plot_number
        idx = np.arange(-(plot_number-1)/2,(plot_number-1)/2+1,1)
        for i,d in enumerate(self.data):
            labels = list(d.df.values)
            rect1 = ax.bar(x+idx[i]*width,labels,width,label=d.log_name)
            self.autolabel(rect1,ax)
            #self.autolabel(rect1,ax)
        ax.set_xticks(x)
        ax.set_title("Orre kop ")
       
        ax.set_xticklabels(self.data[0].df.keys())
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