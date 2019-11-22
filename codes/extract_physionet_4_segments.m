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
Y=[];
file_name=[];
train_files = [];
states=[];
train_parts = [];
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
for file_idx=1:num_files
    disp(file_idx);
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
    
%% Run springer's segmentation

    assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,... 
                    springer_options.audio_Fs,... 
                    Springer_B_matrix, Springer_pi_vector,...
                    Springer_total_obs_distribution, false);
         %%      
    [idx_states , last_idx]=get_states(assigned_states); %idx_states ncc x 4 matrix 
                                % containing starting index of segments 
                                
    %% calculate max length of segs
%     sub = idx_states(1:end-1,4) - idx_states(2:end,1);
%     wrong = wrong + sum(sub>0);
%     right = right + sum(sub<0);
%     
%     state = idx_states;
%     state(:,1:3) = idx_states(:,2:4);
%     state(1:end-1,4) = idx_states(2:end,1);
%     state(end:end) = last_idx;
%     if(state(end:end)-idx_states(end:end)<0)
%         state = state(1:end-1,:);
%         idx_states = idx_states(1:end-1,:);
%     end
%     sub = state - idx_states;
% 
%     s1 = max(s1,max(sub(:,1)));
%     sis = max(sis,max(sub(:,2)));
%     s2 = max(s2,max(sub(:,3)));
%     dis = max(dis,max(sub(:,4)));
%     
%     s1m = min(s1m,min(sub(:,1)));
%     sism = min(sism,min(sub(:,2)));
%     s2m = min(s2m,min(sub(:,3)));
%     dism = min(dism,min(sub(:,4)));
%     [s1 sis s2 dis s1m sism s2m dism]
    
%% Dividing signals into filter banks
    clear PCG
    PCG = PCG_resampled;

    nfb = 4;
    ncc = size(idx_states,1);
    train_parts = [train_parts;ncc];
    s = nan(ncc,Ns1);
    sis = nan(ncc,Nsis);
    ss = nan(ncc,Ns2);
    dis = nan(ncc,Ndis);
    take_last_beat = 0;
    
    for row=1:ncc
        for fb=1:nfb
            take_last_beat = 0;
            if row == ncc % for the last complete cardiac cycle
                if(last_idx>idx_states(row,4))
                    take_last_beat = 1;
                    ts1 = PCG(idx_states(row,1):idx_states(row,2)-1);
                    tsis = PCG(idx_states(row,2):idx_states(row,3)-1);
                    ts2 = PCG(idx_states(row,3):idx_states(row,4)-1);
                    tdis = PCG(idx_states(row,4):last_idx-1);
%                     tmp = PCG(idx_states(row,1):last_idx-1);
                end
            else
                ts1 = PCG(idx_states(row,1):idx_states(row,2)-1);
                tsis = PCG(idx_states(row,2):idx_states(row,3)-1);
                ts2 = PCG(idx_states(row,3):idx_states(row,4)-1);
                tdis = PCG(idx_states(row,4):idx_states(row+1,1)-1);
%                 tmp = PCG(idx_states(row,1):idx_states(row+1,1)-1);
            end
%             N = nsamp-length(tmp); % append zeros at the end of cardiac cycle
            ns1 = Ns1-length(ts1);
            nsis = Nsis-length(tsis);
            ns2 = Ns2-length(ts2);
            ndis = Ndis-length(tdis);
%             x(row,:) = [tmp; zeros(N,1)];
            s(row,:) = [ts1; zeros(ns1,1)];
            sis(row,:) = [tsis; zeros(nsis,1)];
            ss(row,:) = [ts2; zeros(ns2,1)];
            dis(row,:) = [tdis; zeros(ndis,1)];
            
        end
        file_name=[file_name;string(d(file_idx).name)]; % matrix containing the filename
                                                % of each cardiac cycle
        Y=[Y;labels(label_pointer,:)];               % Class labels for each cardiac cycle
    end
    file_name=[file_name;string(d(file_idx).name)]; % matrix containing the filename
%     X=[X;x]; % matrix containing all cardiac cycles
    s1 = [s1;s];
    systole = [systole;sis];
    s2 = [s2;ss];
    diastole = [diastole;dis];
    
    states=[states;idx_states]; % matrix containing 
                                %index of states of each cardiac cycle
    label_pointer=label_pointer+1;  % point at label for the next recording
                                    % increasing with each loop
end
    wav_name = convertStringsToChars(file_name);
%% Save Data
    sname=[savedir 'fold_' 'a'+folder_idx '.mat',];
    disp(['Saving ' sname])
    trainY = Y;
    %save(sname, 'X', 'Y', 'states', 'file_name');
    save(sname, 's1','s2','systole','diastole', 'trainY', 'train_files','train_parts','wav_name','-v7.3');
end






%% function to extract state index
function [idx_states,last_idx] = get_states(assigned_states)
    indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

    if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end

    indx2                = indx(K:end); % K controls the starting cycle
                                        % of the segment. Starting cycle
                                        % is always kept 1 through the 
                                        % switch cases (!)
                                        
    rem                  = mod(length(indx2),4);
    last_idx             = length(indx2)-rem+1;
    indx2(last_idx:end) = []; % clipping the partial segments in the end
    idx_states           = reshape(indx2,4,length(indx2)/4)'; % idx_states 
                            % reshaped into a no.segments X 4 sized matrix
                            % containing state indices
end