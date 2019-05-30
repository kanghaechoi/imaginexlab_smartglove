function [b1,a1] =butterworth_5hz(fs)
%% 5 Hz low-pass filter

fmax = fs/2; %Nyquist sampling rate
LPF_cutoff = 5/fmax;
order = 5;
[b1,a1] = butter(order, LPF_cutoff, 'low');
end