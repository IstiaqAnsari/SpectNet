%%Extract cardiac segments from heart sound signals
%%
Ns1=200;%max of s1 to sis = 180
Nsis = 800;%max of sis to s2 = 780
Ns2=200;%max of s2 to dis = 160
Ndis=1700;%max of dis to next s1 = 1680
% s1m=2000;%min of s1 to sis = 60
% sism = 2000;%min of sis to s2 = 20
% s2m=2000;%min of s2 to dis = 40
% dism=2000;%min of dis to next s1 = 140
% wrong = 0;
% right = 0;
%%
for folder_idx=0:0
    disp("folder idx");
    disp(folder_idx);
    if folder_idx==10
        continue
    end
clearvars -except folder_idx Ns1 Nsis Ns2 Ndis
clc
%% Initialize Parameters
% folder_idx=4; %index for training folder [0 to 5]
max_audio_length=60;    %seconds
N=60;                   %order of filters
sr=1000;                %resampling rate
nsamp = 2500;           %number of samples in each cardiac cycle segment
s1=[];
systole=[];
s2 = [];
diastole = [];
trainX = [];
trainY=[];

file_name=[];
train_files = [];
states=[];
train_parts = [];
sig_quality = []
%% Initialize paths

datapath=['/media/mhealthra2/Data/heart_sound/Heart_Sound/Physionet/training/training-' 'a'+folder_idx '/'];
labelpath=['/media/mhealthra2/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/20160725_Reference with signal quality results for training set/' 'training-' 'a'+folder_idx '/REFERENCE_withSQI.csv'];
savedir='/media/mhealthra2/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/individual_fold_4_segments/';
exclude_text='/media/mhealthra2/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/Recordings need to be removed in training-e.txt';
addpath(genpath('/media/mhealthra2/Data/heart_sound/Adversarial-Heart-Sound-Classification/miscellaneous/cristhian.potes-204/'));
d=dir([datapath,'*.wav']);
num_files=size(d,1);

%% Initialize filter bank
% 
% Wn = 45*2/sr; % lowpass cutoff
% b1 = fir1(N,Wn,'low',hamming(N+1));
% Wn = [45*2/sr, 80*2/sr]; %bandpass cutoff
% b2 = fir1(N,Wn,hamming(N+1));
% Wn = [80*2/sr, 200*2/sr]; %bandpass cutoff
% b3 = fir1(N,Wn,hamming(N+1));
% Wn = 200*2/sr; %highpass cutoff
% b4 = fir1(N,Wn,'high',hamming(N+1));

%% Feature extraction using Springers Segmentation

load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;

%% Importing labels
labels=importlabel(labelpath); % first column normal(-1)/abnormal(1) second column good(1)/bad(0)
label_pointer=1; % label saving index
%% Import list of files to be excluded

exclude = importlist(exclude_text);
ftype = repmat('.wav',[length(exclude),1]); 
exclude = strcat(exclude,ftype); % list of files to be excluded from training-e
%%
check = 1  ; 
label_pointer = check;
for file_idx=check:num_files
    fprintf('File number %i\n', file_idx)

%% Importing signals
    if folder_idx==4    % if dataset is training-e
        if sum(cell2mat(strfind(exclude,d(file_idx).name))) % if file is found in exclude list
            continue;
        end
    end
    
    %%
    fname=[datapath,d(file_idx).name];
    [PCG,Fs1] = audioread(fname);
    if length(PCG)>max_audio_length*Fs1
        PCG = PCG(1:max_audio_length*Fs1); % Clip signals to max_audio_length seconds
    end
        
%% Pre-processing (resample + bandpass + spike removal)

    % resample to 1000 Hz
    PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); 
    % filter the signal between 25 to 400 Hz
    PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
    PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
    % remove spikes
    PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);
    
    fprintf('label pointer %i\nLabels\n',file_idx);
    disp(labels(file_idx,:));
    fprintf('%s\n',d(file_idx).name);
    label_pointer = label_pointer+1;
    sav_name = strrep(fname,'/training/','/training_resampled/');
    fprintf('%s\n',sav_name);
    audiowrite(sav_name,PCG_resampled,sr);
%% Dividing signals into filter banks

%     trainY=[trainY;labels(label_pointer,:)];               % Class labels for each cardiac cycle
% 
%     file_name=[file_name;string(d(file_idx).name)]; % matrix containing the filename
%     trainX=[trainX;PCG_resampled]; % matrix containing all cardiac cycles
% 
%     X = [X;x];
%     states=[states;idx_states]; % matrix containing 
%                                 %index of states of each cardiac cycle
%     label_pointer=label_pointer+1;  % point at label for the next recording
%                                     % increasing with each loop
end
    wav_name = convertStringsToChars(file_name);
    sig_quality = labels(:,2);
%% Save Data
    sname=[savedir 'fold_' 'aa'+folder_idx '.mat',];
    disp(['Saving ' sname])
    trainY = Y;
    %save(sname, 'X', 'Y', 'states', 'file_name');
    %save(sname, 's1','s2','systole','diastole', 'trainY', 'train_files','train_parts','sig_quality','wav_name','-v7.3');
end






