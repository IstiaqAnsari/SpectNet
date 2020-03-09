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
    def __init__(self,log,dann=False,find = False):
        #find is a bool, if true then the log name is searched using the log when the log is only the fold name
        # not the full path . result comparison calls it with the full path. so no need of find variable 
        # but must in model loading which is set in heartnet.getattentionModel
        self.log_dir = '../../Adversarial Heart Sound Results/logs/'
        self.metrics = ['val_macc','val_F1','val_precision','val_sensitivity','val_specificity']
        self.log_name = log.split(' ')[0]
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
        self.df.sort_values(by=['val_macc','val_F1'],ascending=False,inplace = True)
        self.df = dict(self.df.iloc[0][self.metrics])

## Compare any number of result 
class ResultsComparison():
    def __init__(self,logs,logs_dann=None):
        if(logs_dann is not None):
<<<<<<< HEAD
            self.log_dir_dann = '../../Heartnet_Results/logs/dann/'
        self.log_dir = '../../Heartnet_Results/logs/'
        self.metrics = ['macc','F1','precision','sensitivity','specificity']
=======
            self.log_dir_dann = '../../Adversarial Heart Sound Results/logs/dann/'
        self.log_dir = '../../Adversarial Heart Sound Results/logs/'
        self.metrics = ['val_macc','val_F1','val_precision','val_sensitivity','val_specificity']
>>>>>>> parent of 51edc1c... result showing edited  key.
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
        print(x)
        if(fig==None):
            fig, ax = plt.subplots(figsize=figsize)
        plot_number = len(self.data)
        width = .8/plot_number
        idx = np.arange(-(plot_number-1)/2,(plot_number-1)/2+1,1)
        macc_avg = 0
        for i,d in enumerate(self.data):
            macc_avg = macc_avg + d.df['val_macc']
            print(d.log_name)
            labels = list(d.df.values())
            rect1 = ax.bar(x+idx[i]*width,labels,width,label=d.log_name)
            self.autolabel(rect1,ax)
            #self.autolabel(rect1,ax)
        print(macc_avg/plot_number)
        ax.set_xticks(x)
<<<<<<< HEAD
        ax.set_title("Orre kop ")
       
        ax.set_xticklabels(self.data[0].df.keys())
=======
        ax.set_title("Results Comparison")
        ax.set_xticklabels(self.metrics)
>>>>>>> parent of 51edc1c... result showing edited  key.
        ax.legend()
        fig.tight_layout()
        if(shownow):
            plt.show()
    def autolabel(self,rects,ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')