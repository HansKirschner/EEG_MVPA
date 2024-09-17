function [ Odata ] = EEG_SVM_Predict( indata, inSVM, Trials, time_w, varargin)
%%
%Function that runs matlab integrated SVM on an EEG dataset.
%
%INPUT:
%'indata'        reflecting dataset 2 = the dataset to be classified, either EEG structure or path to eeg file that will be openend.
%'inSVM'        output of 'AGF_EEG_SVM_TF' structure, the SVM that is used to classify 'indata', i.e., the 'training' dataset.
%'trial_ind'    indexes the trials of interest in 'indata'.
%'time_w'       time window (ms) that should be classified in dataset 2 (note: which SVM is used at which point in time is defined
%                   by input argument 'WhichSVM'). If empty, use exact same time as input SVM.
%
%OPTIONAL INPUT:
%
%'disp'         0 = quite mode, 1 = text output.
%'GT'           Provide ground truth of data: this has to be a vector of length(Trials) coding the actual
%                   categories per trial.
%'WhichSVM'     This determines which SVM from the input dataset will be used for classification of the second dataset (time = SVM from each time).
%               {'time' 'best' '##'} where '##' = a specific time in the SVM dataset (e.g., '220' = 220 ms, still has to be a string). 
%'bin_size'     If 'time_w' is not empty, this these arguments determine bin and stepsize for the analysis. Warning: bin_size
%'stepsize'         should be equal to the bin_size used in the generation of the SVMs used here!
%'scaling'      Scaling applied to the data. Default: scaling as in training dataset (4)!
%                   1 = min_max_1 (scales data range to -1 : 1)
%                   2 = z-score data.
%'scale_dim'    If scaling is set, this defines the scaling dimension. 
%                   0 = the whole dataset (all channels and time-points
%                   decide scaling, this leaves the data relations unchanged, default).
%                   1 = scale each channel over time (this changes relationship between channels, as all will have the same scale).
%'ScalingValues' Can be set to the exact scaling values as returned by the previous SVM (Field: info.ScalingValues) or any value. 
%                   However, if scaling is set to be channel wise, it has to have the exact same number of rows as the dataset has channels!
%
%%%%%OUTPUT%%%%
%Odata              All the data from the SVM analysis.

%Warning: Currently only supports datasets with ONE cluster of electrodes!!

%Set defaults
FStepSize       = inSVM.info.stepsize;
Fbins           = inSVM.info.binsize;
Fdisp           = 1;
scaling         = 4;
scale_dim       = inSVM.info.scale_dim;
GT              = [];
WhichSVM        = {'time'}; %default: use the SVM from training SVM at each point in time.
Tstr            = '';
Ecluster        = inSVM.info.ecluster;
if isfield(inSVM, 'TF')
    TF              = inSVM.TF;
else
    TF = [];
end
TFdims          = 1; %Dummy variable, overwritten if TF is active
ScalingValues   = [];
FitPosteriorOn  = 0; %Can be set to 1 manually: calculates true posterior probability (this takes long when more than one training SVM is used (~2 s / SVM)

nargs = nargin-4;
if nargs > 1 
  if ~(round(nargs/2) == nargs/2)
    error('Odd number of input arguments??')
  end
end;
for i = 1:2:length(varargin)
    Param = varargin{i};
    if ~isstr(Param)
      error('Flag arguments must be strings')
    end
    Param = lower(Param);
    switch Param     
        case 'gt'
            GT=varargin{i+1};
            if size(GT,2)==size(Trials,2)
                GT=GT'; %transpose ground truth
            elseif numel(GT) ~= numel(Trials)
                disp(['Error: Ground truth has to be the same length as the number of trials. numel(GT) = ' num2str(length(GT)) ' and numel(Trials) = ' num2str(length(Trials))]); return;
            end;
        case 'disp'
            Fdisp=varargin{i+1};
        case 'whichsvm'
            WhichSVM=varargin{i+1};
        case 'bin_size'
                Fbins=varargin{i+1};
        case 'stepsize'
            FStepSize=varargin{i+1};
        case 'scaling'
            scaling=varargin{i+1}; 
        case 'scale_dim'
            scale_dim=varargin{i+1}; 
        case 'scalingvalues'
            ScalingValues=varargin{i+1};
        otherwise
            disp(['Unknown argument ' varargin{i} '...'])
            pause
    end;
end; 

%%
Tstr            = '';
clc

for HideTimes = 0
    if strcmp(WhichSVM, 'time') %same time as SVM is specified
        if isempty(time_w) 
            time_w      = unique([min(inSVM.info.times) max(inSVM.info.times)]);
            FStepSize   = inSVM.info.stepsize;
            Fbins       = inSVM.info.binsize;
            Time_in_DP  = find([indata.times]==time_w(1)) : FStepSize : find([indata.times]==time_w(end));
            if isempty(Time_in_DP)
                disp('Error: Time windows of training SVM and new dataset do not align...'); return;
            end;
            PlotTime    = indata.times(Time_in_DP);
            TimeSVM     = 1:length(Time_in_DP);
            TimeSVMms   = inSVM.info.times(TimeSVM);
            Tstr        = [Tstr 'Using time window of input SVM (' num2str(PlotTime(1)) ' to ' num2str(PlotTime(end)) ' ms) with a stepsize of ' num2str(FStepSize) ' and a bin size of ' num2str(Fbins) '\n'];
        else
            Time_in_DP  = find([indata.times]==time_w(1)) : FStepSize : find([indata.times]==time_w(end));
            PlotTime    = indata.times(Time_in_DP);
            TimeSVM     = find(ismember([inSVM.info.times], PlotTime)==1);
            TimeSVMms   = inSVM.info.times(TimeSVM);
            Tstr        = [Tstr 'Using part of SVM time window from ' num2str(PlotTime(1)) ' to ' num2str(PlotTime(end)) ' ms with a stepsize of ' num2str(FStepSize) ' and a bin size of ' num2str(Fbins) '\n'];
        end;
    elseif strcmp(WhichSVM, 'best') %use the SVM from the highest accuracy of all possible SVMs
        if isempty(time_w)
            disp('Argument ''timew'' must be specified when best time is used...');return;
        end;
        Time_in_DP  = find([indata.times]==time_w(1)) : FStepSize : find([indata.times]==time_w(end));
        PlotTime    = indata.times(Time_in_DP);
        %EclustName  = fieldnames(inSVM.Accuracy);
        AccSVM      = inSVM.Accuracy;
        [x,TimeSVM] = max(AccSVM);
        TimeSVMms   = inSVM.info.times(TimeSVM);
        Tstr        = [Tstr 'Using the SVM from peak accuracy (' num2str(x) ' at ' num2str(TimeSVMms) ' ms) to classify all times of the new dataset 2.\n'];
    else  %use specific SVM time window
        if isempty(time_w)
            disp('Argument ''timew'' must be specified when specific time is used...');return;
        end;
        Time_in_DP  = find([indata.times]==time_w(1)) : FStepSize : find([indata.times]==time_w(end));
        PlotTime    = indata.times(Time_in_DP);
        try
            Tinput      = str2num(WhichSVM);
            TimeSVM     = find([inSVM.info.times]==Tinput);
            TimeSVMms   = inSVM.info.times(TimeSVM);
            if isempty(TimeSVMms)
                disp(['Error: Specified time (' WhichSVM ' ms) not found in training SVM dataset...']);return
            end;
        catch
            disp('Error: Input argument WhichSVM is not recognized and cannot be converted to a number... aborting')
            disp(['Input is: ' WhichSVM])
            return;
        end;
        Tstr        = [Tstr 'Using specified time from training SVM (' num2str(TimeSVMms) ' ms) to classify all times of the new dataset 2 from ' num2str(PlotTime(1)) ' to ' num2str(PlotTime(end)) ' ms.\n'];
    end;
end;

%TW for smoothing
Tbef = floor(Fbins/2);
Tafter = ceil(Fbins/2);

%Exchange data with ica activation if set
if inSVM.info.ICA==1
    [indata]=eeg_checkset(indata); %recompute ICA activation
    indata.data=[];
    indata.data=indata.icaact;
    %Replace channel lables
    for i = 1 : length(indata.chanlocs)
        indata.chanlocs(i).labels = ['IC_no' num2str(i)];
    end;
    Tstr = [Tstr 'Using IC activity instead of regular EEG.\n'];
end;

%find unique electrodes specified in different clusters to save time and perform TF only here
Etotal = [];
for Ecount = 1 : size(Ecluster,2)
    Etotal = [Etotal Ecluster{Ecount}];
end;
Etotal = unique(Etotal);

%performe time-frequency decomposition of the EEG data before running the SVM
%this replaces the time-domain data completely
if ~isempty(TF)
   [Sensor_TF, Odata, TFdims, TFtimesMs, AngleInfo] = AGF_SVM_TF(indata, TF, PlotTime, Fdisp, Trials, Etotal);
end;

%%
%In order to scale EEG input in different ways, EEG data has to be pre-calculated
scaleT = ' ';
SensorInformation=nan(length(PlotTime),length(Trials),length(Etotal),TFdims); %4D time trial channel frequency (if set). Last dimension is always time domain EEG at first position!

%Smooth time domain data (with Fbins) and translate everything into one big array 'SensorInformation'
for FreqCount = 1 : TFdims
    K=0;
    for bc = find([indata.times]==time_w(1)) : FStepSize : find([indata.times]==time_w(2))
        K=K+1;
        SensorInformation(K,:,:,FreqCount) = squeeze(mean(indata.data(Ecluster{Ecount},[bc-Tbef:bc+Tafter-1],Trials),2))'; %First entry is always time domain EEG
    end;
    if ~isempty(TF)
        SensorInformation(:,:,:,FreqCount+1)=Sensor_TF(:,:,:,FreqCount); 
    end;
end;
clear Sensor_TF;
%%
if ~isempty(TF)
    if TF.angle==1 %add angle information which should not be scaled
        SensorInformation(:,:,:,(FreqCount+1:FreqCount*3)+1) = AngleInfo(:,:,:,:);
    end;  
end;

%This index simply adresses which information is stored in 'SensorInformation' matrix: 1 = EEG, 2 = EEG spectral power, 3 = EEG phase angels
if TFdims > 1 && TF.angle==1 
    WhatIsWhere = [1 ones(1,TFdims).*2 ones(1,TFdims*2).*3]; 
elseif TFdims > 1 && TF.angle~=1 
    WhatIsWhere = [1 ones(1,TFdims).*2];
else
    WhatIsWhere = 1;
end;
%%
if scaling~=0
    Tstr = [Tstr '******************SCALING*****************\n'];
    if scaling ~= 4
        if scale_dim~=3
            %scale normal EEG
            [SensorInformation(:,:,:,1), SaveScale, Tstr ] = AGF_SVM_SCALE(SensorInformation(:,:,:,1), scale_dim, scaling, [Tstr 'EEG ' ]);
            %scale power
            if sum(WhatIsWhere==2)~=0 && TF.scalepower~=0
                [SensorInformation(:,:,:,WhatIsWhere==2), SaveScale, Tstr ] = AGF_SVM_SCALE(SensorInformation(:,:,:,WhatIsWhere==2), scale_dim, scaling, [Tstr 'Power ' ]);
            end;
        elseif scale_dim==3 %scale all information that is available (including power and angle)
            [SensorInformation, SaveScale, Tstr ] = AGF_SVM_SCALE(SensorInformation, 0, scaling, [Tstr 'EEG ' ]);
        end;
    else
        if isempty(ScalingValues) %Use scaling values from training dataset
            ScalingValues = inSVM.info.ScalingValues;
        end;
        if inSVM.info.scaling==1 %input dataset was min max 1 scaling
            T = SensorInformation - min(min(min(min(SensorInformation))));  %subtract lowest value
            SensorInformation = T./(max(max(max(max(T))))/2)-1; clear T
            SensorInformation = demean(SensorInformation);
        elseif inSVM.info.scaling==2 %input dataset was z scored
            SensorInformation = (SensorInformation - ScalingValues(1))./ScalingValues(2);
        else
            
        end;
        Tstr = [Tstr 'Data is scaled the same way as training SVM across all trials and electrodes.\n'];
    end;
    Tstr = [Tstr '\n'];    
end;

%%
%Perform security check if dimensions match!
if length(TimeSVM)>1
    if size(SensorInformation,1) ~= length(TimeSVM)
        disp('Error: Size of training SVM does not match time from classification dataset...'); return;
    end;
    if size(SensorInformation,1) ~= length(TimeSVM)
        disp('Error: Size of training SVM does not match time from classification dataset...'); return;
    end;
end;

if Fdisp %display information to user if set
    clc
    Tstr = ['Data is binned to ' num2str(Fbins) ' data points using a step size of ' num2str(FStepSize) ' data points.\n\n' Tstr];
    Tstr = ['Input dataset epoch from ' num2str(indata.xmin) ' to ' num2str(indata.xmax) ' ms.\n' Tstr];
    Tstr = ['Time window to apply training SVM to is ' num2str(PlotTime(1)) ' to ' num2str(PlotTime(end)) ' ms.\n' Tstr];
    fprintf(Tstr);
end;
%%
%Build a vector index for the data matrix
if ~isempty(TF)
    Fnames={'EEG' 'xPower' 'xPhase'};
    for c = 1 : length(TF.combination)
        Ivector = []; NameVector='';
        for c2 = 1 : 3
            if TF.combination{c}(c2)==1
                Ivector = [Ivector find(WhatIsWhere==c2)];
                NameVector = [NameVector Fnames{c2}];
            end;
        end;
        if strcmp(NameVector(1),'x')
            NameVector = NameVector(2:end);
        end;
        NameVectorAll{c} = NameVector;
    end;
else
    Ivector = 1;
end;

%%
%%%%%%MAIN ANALYSIS%%%%%%%%
%Allocate
PosteriorClassProb = NaN(length(Time_in_DP), length(Trials));
C                  = NaN(length(Time_in_DP), length(Trials));
PercentClassOne    = NaN(1,length(Time_in_DP));
Accuracy           = [];
%Set the first SVM (does not have to be updated if one is used for all datapoints)

UseSVM = inSVM.Full_SVM_Class{1}{TimeSVM(1)}; %Load first SVM from training dataset
if FitPosteriorOn %estimate the transformation function from scores (standard) to posterior probabilities (should not matter much if data is subsampled!)
    [UseSVM] = fitPosterior(UseSVM);
end;

for DP = 1 : length(Time_in_DP) %loop through datapoints
    if length(TimeSVM)>1 %Update loaded SVM if different SVMs are used
        UseSVM = inSVM.Full_SVM_Class{1}{TimeSVM(DP)};
        if FitPosteriorOn 
            [UseSVM] = fitPosterior(UseSVM);
        end;
    end;
    
    %Use this SVM to classify all trials at this time
    [C(DP,:), A] = predict(UseSVM, reshape(squeeze(SensorInformation(DP,:,:,Ivector)),size(SensorInformation,2),size(SensorInformation,3)*length(Ivector))); %C = class lable, A = posterior probability of this lable (A(:,1) = negative posterior probability)
    %[C(DP,:), A] = predict(UseSVM, squeeze(SensorInformation(DP,:,:,Ivector))); 
    PosteriorClassProb(DP,:)=A(:,2); 
    PercentClassOne(DP) = sum(C(DP,:))/numel(C(DP,:));    
    if ~isempty(GT) %If a ground truth is provided, also return accuracy
        Accuracy(DP) = 1-sum(GT~=C(DP,:))/numel(GT);
    end;
end;

Odata.info.times            = PlotTime;
Odata.info.Lables           = inSVM.info.Lables;
Odata.info.TrainingSVMtimes = TimeSVMms;
Odata.info.stepsize         = FStepSize;
Odata.info.binsize          = Fbins;
Odata.info.scaling          = scaling;
Odata.info.scale_dim        = scale_dim;
Odata.info.ecluster         = Ecluster;
Odata.info.TF               = TF;
Odata.accuracy              = Accuracy;             %accuracy is calculated if a ground truth is provided (otherwise meaningless)
Odata.Predicted_Class       = C;                    %the predicted class for each trial and point in time
%Odata.PercentClassOne      = PercentClassOne;      %Percentage of trials that is classified into class 1 at each point in time
Odata.PostProb_Class        = PosteriorClassProb;   %Usually the score of each trial (0 = boundary between classes). Can be set to reflect true posterior probability (not standard)
Odata.GT_Prediction         = GT;

return;