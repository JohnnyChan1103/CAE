clear all;
clc;
path = 'D:\Develop\Dataset\clean_testset_wav\clean_testset_wav\';
outputpathAmp = 'D:\Develop\Dataset\TESTSET_CLEAN_STFT\Amp\';
outputpathAng = 'D:\Develop\Dataset\TESTSET_CLEAN_STFT\Ang\';
%nfile = "D:\Develop\Dataset\noisy_trainset_wav\p226_005.wav";

audio  = dir([path '*.wav']);
wlen = 256;
for i = 1:length(audio)
    info = audioinfo([path audio(i).name]);
    [y,Fs]=audioread([path audio(i).name]);
    [p,q] = rat(16000/Fs);
    y = resample(y,p,q);
    size(y);
    %% Short-time Fourier Transform
    [s,f,t]=spectrogram(y,wlen,[],[],Fs/4); %Hamming window by default
    %[outputpathAmp audio(i).name(1:end-4) '.csv']
    %size(s)
    %size(f)
    %size(t)
    writematrix(abs(s),[outputpathAmp audio(i).name(1:end-4) '.csv'])
    writematrix(angle(s),[outputpathAng audio(i).name(1:end-4) '.csv'])
    i
end
