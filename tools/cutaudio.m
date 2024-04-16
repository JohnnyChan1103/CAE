clear all;
clc;
path = 'D:\Develop\Dataset\clean_trainset_wav\';
outputpath = 'D:\Develop\Dataset\TRAINSET_CLEAN\';
%nfile = "D:\Develop\Dataset\noisy_trainset_wav\p226_005.wav";

audio  = dir([path '*.wav']);
for i = 1:length(audio)
    info = audioinfo([path audio(i).name])
    [y,Fs]=audioread([path audio(i).name]);
    if size(y,1)>65536
        y=y(1:65536, :);
    else
        y=[y;zeros(65536-size(y,1),1)];
    end
    audiowrite([outputpath audio(i).name],y,Fs)
end
