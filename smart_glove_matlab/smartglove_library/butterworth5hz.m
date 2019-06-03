function [B1,A1] =butterworth5hz(FS)
%% 5 Hz low-pass filter

fMax = FS/2; %Nyquist sampling rate
LPFCutOff = 5/fMax;
ORDER = 5;
[B1,A1] = butter(ORDER, LPFCutOff, 'low');
end