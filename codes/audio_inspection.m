addpath('../miscellaneous/cristhian.potes-204');

PCG = data;
Fs1 = fs;

springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;
% resample to 1000 Hz
PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); 
% filter the signal between 25 to 400 Hz
PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
% remove spikes
PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);
sound(PCG_resampled,1000);
plot(PCG_resampled)
