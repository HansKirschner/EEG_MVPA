function [ Odata, Accuracy, Weights ] = EEG_SVM_TF( indata, Trials, UseY, time_w, varargin)
%%
%Function that runs matlab integrated SVM on an EEG dataset.
%
%INPUT:
%'indata'       either EEG structure or path to eeg file that will be
%               openend.
%'trial_ind'    indexes the trials of interest.
%'time_w'       time window (ms) that should be classified.
%
%OPTIONAL INPUT:
%
%'DoAvg'            Should trials be avaraged (0 = no (default), 1 = yes compute ERP for training and test set, 2 = compute ERPs only for training-set)
%'AvgOverN'         How many trials should be avagrage
%'bin_size'         Signal is averaged over that many datapoints (default = 0).
%'stepsize'         Will jump that many datapoints (default = 1).
%'disp'             0 = quite mode, 1 = text output, 2 = plotting of searchlight as well as text (default).
%'TF'               Calculate TF transformation before running SVM. This has to have distinct fields:
%
%                   TF.frequencies  = frequency range
%                   TF.stepnumber   = number of steps within that range
%                   TF.cyclenumber  = number of cycles (will scale with the frequency (FWHM of gaussian) 
%                                     can also be a range (e.g. [3 13]), so that the width changes
%                                     as a function of frequency
%                   TF.space        = 'linear' or 'log'
%                   TF.bands        = name for bands over which should be
%                                     averaged before(!) regression {'alpha' 'beta'}
%                   TF.bandfreq     = frequencies for those bands (e.g. {[8 12] [12 23]}), length must equal TF.bands field.
%                   TF.basetime     = baseline used for TF subtraction (most be included in the time-range!)
%                   TF.basetype
%                       'subtract'  - subtracts each individual trials - baseline within each frequency
%                       'percent'   - subtracts each individual trials - baseline and divides by the mean baseline over all trials (determined prior to subtraction)
%                   TF.combination  = This field determines different combinations of EEG, Power, and angle:
%                           code as a cell array of logical vectors that index each element. 
%                           {[1 0 1] [0 1 1]} indicates: two SVMs are run separately, the first one uses EEG time domain and angle information,
%                           the second one power and angle information. This setting effects the output...
%'ecluster'         Cell array that contains electrode numbers (e.g.
%                       'ecluster', {[22 30 40:50]}). Data is reduced to these electrodes (also for scaling!).
%'Lables'           Class lables (smaller to bigger, e.g. {'green' 'red'}.
%'n_folding'        How many times should data be shuffled? (default = 10)
%'splitratio'       Ratio between training and test set (default = 0.1, 1 = no cross validation)
%'ret_SVM'          [0/1] when active, returns the SVM at every datapoint used. However, this only works
%                   if argument 'splitratio' is one, i.e., the returned SVM is the best possible at this 
%                   datapoint (default: 0).
%'minimum_test'     Define a minimum for the test sample size (will change
%                   splitratio accordingly, e.g. when 10 then ratio will be
%                   changed to ensure at least 10 entries in test set).
%'Activate_ICA'     Run SVM in ICA space. This replaces electrodes with independent components.
%'Default_sample'   Set to n to use this as the highest sample size
%                   (unless one category has less than n trials)
%'scaling'          Scaling applied to the data. Default: 0 (no scaling). 
%                   1 = min_max_1 (scales data range to -1 : 1)
%                   2 = z-score data.
%'scale_dim'        If scaling is set, this defines the scaling dimension. 
%                   0 = the whole dataset (all channels and time-points
%                   decide scaling, this leaves the data relations unchanged, default).
%                   1 = scale each channel over time (this changes relationship between channels, as all will have the same scale).
%'searchlight'      [0/1/2] activate searchlight mode. 0 = off, 1 = with
%                   plot, 2 = without plot.
%'light_time'       1 (default) time is peak of accuracy from general
%                   analysis, otherwise specify array of time points in ms
%                   e.g. [0:50:300] or [250].
%'UseParallel'      Set to 1 to use parallel computing toolbox and parfor loops.
%'light_mode'       Searchlight mode: Can be an array that contains: 
%                   1 = each electrode separately, 2 = contralateral
%                   electrodes in pairs, 3 = cluster of n_neighbor (see
%                   below) electrodes, 4 = cluster of all electrodes within
%                   e_dist (see below, not yet included). Thus, could be [1 3].
%'n_neighbor'       Number of neighboring electrodes for searchlight (default: 8).
%'e_dist'           Distance of electrodes for searchlight (default: 1).      
%'search_fold'      Number of folds for searchlight analysis (sometimes it makes sense to use more folds here, as 
%                   data is much more noisy at single electrodes or to use fewer folds as it can take quite a while...).   
%'add_arguments'    Additional arguments to be passed along to the fitcsvm function call {'Name1', 'Value1', 'Name2', Value2' etc.}
%
%%%%%OUTPUT%%%%
%Odata              All the data from the SVM analysis.
%Accuracy           (time, sonser) Accuracy of the cross validated classifier
%Weights            (time, sensor) The weights of the electrodes that is used for classification. This can easiy be used to calculate posterior 
%                   probabilities for other situation that a certain activation pattern belongs to one of both calsses.
% 


%Set defaults
DoAvg           = 0;
avgoverN        = [];
FStepSize       = 1;
Fbins           = 0;
activate_ica    = [];
IterNum         = 1e8;
n_folding       = 10;
splitratio      = 0.1;
minimum_test    = [];
Fdisp           = 2;
default_sample  = [];
scaling         = 0;
scale_dim       = 0;
TF              = 0;
oldschool       = 0;
searchlight     = 0;
light_time      = 1;
light_mode      = [1 2 3];
n_neighbor      = 8;
e_dist          = 1;
search_fold     = 0;
TFdims          = 1;
ret_SVM         = 0; 
Lables          = {'class1' 'class2'}; %standard lables
SaveScale       = [NaN]; %This remembers how the data was scaled (1: value to subtract; 2: valaue to divide)
add_arguments   = {};
DispSTR         = ''; %String stores warning messages
UseParallel     = 0; %Do not use parallel computing toolbox to parallize loops


Fcmap = load([pwd '/EEG_MVPA/helpers/AGF_cmap.mat']);
Fcmap=Fcmap.AGF_cmap(33:end,:);

%%
for hide = 1:1 %Get Variables from varargin array
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
            case 'doavg'
                DoAvg=varargin{i+1};
            case 'avgovern'
                avgoverN=varargin{i+1};
            case 'bin_size'
                Fbins=varargin{i+1};
            case 'stepsize'
                FStepSize=varargin{i+1};
            case 'disp'
                Fdisp=varargin{i+1};
            case 'tf'
                TF=varargin{i+1};
            case 'frequency'
                Ffrequency=varargin{i+1};
                disp(['Returns power envelope for frequency/frequencies: ' Ffrequency{:}])
            case 'ecluster'
                Ecluster=varargin{i+1};
            case 'n_folding'
                n_folding=varargin{i+1};
            case 'iternum'
                IterNum=varargin{i+1};
            case 'splitratio'
                splitratio=varargin{i+1};
            case 'minimum_test'
                minimum_test=varargin{i+1};  
            case 'activate_ica'
                activate_ica=varargin{i+1};     
            case 'default_sample'
                default_sample=varargin{i+1}; 
            case 'scaling'
                scaling=varargin{i+1}; 
            case 'scale_dim'
                scale_dim=varargin{i+1}; 
            case 'searchlight'
                searchlight=varargin{i+1}; 
            case 'light_time'
                light_time=varargin{i+1}; 
            case 'light_mode'
                light_mode=varargin{i+1}; 
            case 'n_neighbor'
                n_neighbor=varargin{i+1}; 
            case 'e_dist'
                e_dist=varargin{i+1}; 
            case 'search_fold'
                search_fold=varargin{i+1}; 
            case 'ret_svm'
                ret_SVM=varargin{i+1};   
            case 'lables'
                Lables=varargin{i+1};   
            case 'add_arguments'
                add_arguments=varargin{i+1}; 
            case 'useparallel'
                UseParallel=varargin{i+1};
            otherwise
                disp(['Unknown argument ' varargin{i} '...'])
                pause
        end;
    end; 
end;

if ~isempty(TF)
    TF.angle        = TF.combination{1}(3);
    TF.smoothing    = 0;
    TF.scalepower   = 1;
end

if n_folding>1 && ret_SVM
    DispSTR = [DispSTR 'Warning: Will not return full SVM, only last SVM from n folding procedure.\nSet n_folding to 1 to get the full SVM for each data point.\n'];
end;

%%
%Transpose Lables if dimensions are wrong
if size(UseY,1)<size(UseY,2)
    UseY=UseY';
end;

%Equalize number of items
U = unique(UseY);
U1 = length(find([UseY]==U(1)));
U1INDEX = find([UseY]==U(1));
U2 = length(find([UseY]==U(2)));
U2INDEX = find([UseY]==U(2));
if U1 < U2
    SmallerN = U1;
else
    SmallerN = U2;
end;

if default_sample
    if SmallerN > default_sample %lowest category has more items than default sample size
        SmallerN = default_sample;
    end;
end;

n_cat_low = ceil(SmallerN*splitratio);
n_cat_high = SmallerN - n_cat_low;

%Check if too few items?
if minimum_test
    while n_cat_low < minimum_test
        splitratio = splitratio + 0.01;
        n_cat_low = ceil(SmallerN*splitratio);
        DispSTR = [DispSTR 'Splitratio has been increased to ' num2str(splitratio) ' to ensure minimum test sample size (now: ' num2str(n_cat_low) ').\n'];
    end
end;

Time_in_DP = find([indata.times]==time_w(1)) : FStepSize : find([indata.times]==time_w(2));
PlotTime=indata.times(Time_in_DP);

Tbef = floor(Fbins/2);
Tafter = ceil(Fbins/2);


if ~isempty(activate_ica)
    if activate_ica==1
        if isempty(indata.icaact)
            indata.icaact = eeg_getica(indata);
        end
        indata.data=[];
        indata.data=indata.icaact;
        %Replace channel lables
        for i = 1 : length(indata.chanlocs)
            indata.chanlocs(i).labels = ['IC_no' num2str(i)];
        end;
    end;
end;

%find unique electrodes specified in different clusters to save time and perform TF only here
Etotal = [];
for Ecount = 1 : size(Ecluster,2)
    Etotal = [Etotal Ecluster{Ecount}];
end;
Etotal = unique(Etotal);

%%
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
        SensorInformation(K,:,:,FreqCount) = squeeze(mean(indata.data(Ecluster{Ecount},[bc-Tbef:bc+Tafter-1],Trials),2))';
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
    DispSTR = [DispSTR '******************SCALING*****************\n'];
    if scale_dim~=3
        %scale normal EEG
        [SensorInformation(:,:,:,1), SaveScale, DispSTR ] = AGF_SVM_SCALE(SensorInformation(:,:,:,1), scale_dim, scaling, [DispSTR 'EEG ' ]);
        %scale power
        if sum(WhatIsWhere==2)~=0 && TF.scalepower~=0
            [SensorInformation(:,:,:,WhatIsWhere==2), SaveScale, DispSTR ] = AGF_SVM_SCALE(SensorInformation(:,:,:,WhatIsWhere==2), scale_dim, scaling, [DispSTR 'Power ' ]);
        end;
    elseif scale_dim==3 %scale all information that is available (including power and angle)
        [SensorInformation, SaveScale, DispSTR ] = AGF_SVM_SCALE(SensorInformation, 0, scaling, [DispSTR 'EEG ' ]);
    end;
    DispSTR = [DispSTR '\n'];    
end;

%%
n_iterations = n_folding * K;
if ~isempty(TF)
    if ~isempty(TF.combination)
        n_iterations=n_iterations*length(TF.combination);
    end;
end;
n_features = size(SensorInformation,3);
if numel(size(SensorInformation))==4
    n_features = n_features*size(SensorInformation,4);
end;

if Fdisp
    clc
    fprintf(DispSTR)
    disp(['Dataset has ' num2str(U1) ' entries for ' num2str(U(1)) ' and ' num2str(U2) ' entries for ' num2str(U(2)) '.'])
    disp(['Reducing to ' num2str(SmallerN) ' random items using ' num2str(n_folding) '-fold crossvalidation.'])
    disp(['Testsize is set to ' num2str(splitratio) 'x trainingsize and contains ' num2str(2*n_cat_low) ' observations.'])
    disp(' ')
    if DoAvg == 1
        disp(['Averaging trials (n = ' num2str(avgoverN) ') belonging to the same exemplar before decoding'])
        disp(' ')
    elseif DoAvg == 2
        disp(['Averaging trials (n = ' num2str(avgoverN) ') belonging to the same exemplar before decoding only for training'])
        disp(' ')
    end
    disp(['Input dataset epoch from ' num2str(indata.xmin) ' to ' num2str(indata.xmax) ' ms.'])
    disp(['Timewindow for analysis is ' num2str(time_w(1)) ' to ' num2str(time_w(2)) ' ms.'])
    disp(['Data is binned to ' num2str(Fbins) ' data points using a step size of ' num2str(FStepSize) ' data points.'])
    disp(' ');disp(['Total number of SVMs to be run: ' num2str(n_iterations) ' using up to ' num2str(n_features) ' features per SVM.'])
    disp(scaleT)
    disp(' ')
end;
%%
for n_fold = 1 : n_folding %fold the data
    %Shuffle trials
    U1_Shuffle  = U1INDEX(randperm(U1));
    U2_Shuffle  = U2INDEX(randperm(U2));
    
    %Equalize length
    U1_Shuffle  = U1_Shuffle(1:SmallerN);
    U2_Shuffle  = U2_Shuffle(1:SmallerN);
    %Split into training and Test
    U1_Test     = U1_Shuffle(1:n_cat_low);
    U2_Test     = U2_Shuffle(1:n_cat_low);
    
    if splitratio==1
        %Return the best SVM possible (i.e., use information of all trials without cross validation)
        U1_Train    = U1_Test;
        U2_Train    = U2_Test;
        TrainingTrials(n_fold,:)  = [U1_Train' U2_Train']';
        TestTrials(n_fold,:)      = TrainingTrials(n_fold,:);
    else
        U1_Train    = U1_Shuffle(n_cat_low+1:SmallerN);
        U2_Train    = U2_Shuffle(n_cat_low+1:SmallerN);
        TrainingTrials(n_fold,:)  = [U1_Train' U2_Train']';
        TestTrials(n_fold,:)      = [U1_Test' U2_Test']';
    end;
end;

if search_fold && searchlight
    for n_fold = 1 : search_fold %fold the data
        %Shuffle trials
        U1_Shuffle  = U1INDEX(randperm(U1));
        U2_Shuffle  = U2INDEX(randperm(U2));

        %Equalize length
        U1_Shuffle  = U1_Shuffle(1:SmallerN);
        U2_Shuffle  = U2_Shuffle(1:SmallerN);

        %Split into training and Test
        U1_Test     = U1_Shuffle(1:n_cat_low);
        U2_Test     = U2_Shuffle(1:n_cat_low);
        U1_Train    = U1_Shuffle(n_cat_low+1:SmallerN);
        U2_Train    = U2_Shuffle(n_cat_low+1:SmallerN);
        SearchlightTraining(n_fold,:)  = [U1_Train' U2_Train']';
        SearchlightTest(n_fold,:)      = [U1_Test' U2_Test']';
    end;
else
    SearchlightTraining = TrainingTrials;
    SearchlightTest = TestTrials;
end;

Odata.info.times            = PlotTime;
Odata.info.n_cat            = [U1 U2];
Odata.info.n_train_test     = [n_cat_low n_cat_high];
Odata.info.splitratio       = splitratio;
Odata.info.n_folding        = n_folding;
Odata.info.stepsize         = FStepSize;
Odata.info.binsize          = Fbins;
Odata.info.scaling          = scaling;
Odata.info.scale_dim        = scale_dim;
Odata.info.ecluster         = Ecluster;
Odata.info.Elables          = {indata.chanlocs(Etotal).labels};
Odata.info.TrainingTrials   = TrainingTrials;
Odata.info.TestTrials       = TestTrials;
Odata.info.ScalingValues    = SaveScale;
Odata.info.Lables           = Lables;
Odata.info.ICA              = activate_ica;

%%
%%%%%%MAIN ANALYSIS%%%%%%%%
if ~isempty(TF)
    Fnames={'EEG' 'xPower' 'xPhase'};
    if iscell(TF.combination)
        %Build a vector index for the data matrix
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
        
        %Run analysis either in parallel or not
        if UseParallel == 1
            parfor pc = 1 : length(TF.combination)
                disp(['Running set no ' num2str(pc) ' of ' num2str(length(TF.combination)) ' SVMs using ' NameVector ' information which has ' num2str(length(Ivector)*size(SensorInformation,3)) ' features.']); disp(' ');
                [Accuracy(pc,:,:), ~, ~, FullSVM{pc},PT_Info] = AGF_EASY_SVM( TrainingTrials, TestTrials, SensorInformation(:,:,:,Ivector), UseY, IterNum, oldschool, Fdisp, ret_SVM, add_arguments,DoAvg,avgoverN);
            end;
        else
            for c = 1 : length(TF.combination)
                disp(['Running set no ' num2str(c) ' of ' num2str(length(TF.combination)) ' SVMs using ' NameVector ' information which has ' num2str(length(Ivector)*size(SensorInformation,3)) ' features.']); disp(' ');
                [Accuracy(c,:,:), ~, ~, FullSVM{c},PT_Info] = AGF_EASY_SVM( TrainingTrials, TestTrials, SensorInformation(:,:,:,Ivector), UseY, IterNum, oldschool, Fdisp, ret_SVM, add_arguments,DoAvg,avgoverN);
            end;
        end;
        Odata.info.names    = NameVectorAll;
        Odata.Full_SVM_Class = FullSVM;
        Odata.Accuracy       = nanmean(Accuracy,3); %average over n_fold, first entry is combination of EEG information (EEG, Power, Phase).
        Weights              = NaN; %weights across different dimensions of EEG information do not make much sense on their own, not returned (can be read out though).
        Weights_at_peak      = NaN;
    else
        disp('Field ''TF.combinations'' has to be a cell array (see help for more information).');return;
    end;
else

    %[Accuracy(1,:,:), ~, Weights(1,:,:,:), FullSVM{1},PT_Info] = AGF_EASY_SVM( TrainingTrials, TestTrials, SensorInformation, UseY, IterNum, oldschool, Fdisp, ret_SVM, add_arguments,DoAvg,avgoverN);
    [Accuracy(1,:,:), ~, Weights, FullSVM{1},PT_Info] = AGF_EASY_SVM( TrainingTrials, TestTrials, SensorInformation, UseY, IterNum, oldschool, Fdisp, ret_SVM, add_arguments,DoAvg,avgoverN);
   
    Odata.Full_SVM_Class = FullSVM;
    Odata.Accuracy       = nanmean(Accuracy,3); %average over n_fold, first entry is combination of EEG information (EEG, Power, Phase)
    Odata.info.names     = {'EEG_alone'};
    Odata.PT_Info        = PT_Info;
    %Tolerance      = nanmean(SaveToler,2); %average over n_fold for tolerance
    %     if length(size(Weights))==4
    %         Weights = squeeze(nanmean(Weights,3)); %average over n_fold for weights
    %     else
    %         Weights = squeeze(Weights); %average over n_fold for weights
    %     end;
    Weights = squeeze(nanmean(Weights,3));
    [~,peak] = max(Odata.Accuracy);
    Weights_at_peak = Weights(:,peak);

end;
%%
%Run searchlight if active
squeezefac = 1.2; %squeeze electrode locations somewhat to the vertex
f_xsize= 1400;
f_ysize= 1400;
if searchlight
    for clust = 1 : size(Ecluster,2)
        [MA,y]=max(nanmean(Accuracy(clust,:,:),3));
        TitStr = ['Peak time = ' num2str(PlotTime(y)) ' ms with an accuracy of: ' num2str(round(MA*1000)/10) '%.'];
        
        if Fdisp
            disp(TitStr)
        end;
        if light_time~=1 %get accuracy peak
            for lc = 1 : length(light_time)
                light_time2(lc) = find(PlotTime == light_time(lc));
            end;
            y = [y light_time2]; %also perform searchlight at other time points pre-specified by input
        end;
        
        Odata.lightinfo.times       = PlotTime(y);
        Odata.lightinfo.PeakAcc     = num2str(round(MA*1000)/10);
        Odata.lightinfo.light_mode  = light_mode;
        Odata.lightinfo.n_fold      = search_fold;
        Odata.lightinfo.n_neighbor  = NaN;
        
        if Fdisp & clust==1
            disp(['Performing searchlight analysis over ' num2str(size(SearchlightTraining,1)) ' folds with settings: '])
            if ~isempty(light_mode(light_mode==1)); disp(['Single electrode search']);end;
            if ~isempty(light_mode(light_mode==2)); disp(['Contralateral electrode search']);end;
            if ~isempty(light_mode(light_mode==3)); disp(['Cluster search with ' num2str(n_neighbor) ' nearest neighboring electrodes']);end;
        end;
        
        for T = 1 : length(y)
            clear glob
            ax =  [indata.chanlocs(Ecluster{clust}).X]; ax(abs(ax)<0.01)=0;
            ay =  [indata.chanlocs(Ecluster{clust}).Y]*squeezefac; ay(abs(ay)<0.01)=0;
            az =  [indata.chanlocs(Ecluster{clust}).Y];
            if max(ax)>5
                ax=ax/100; ay=ay/100; az=az/100;
            end;
            
            if Fdisp == 2
                close all; figure; hFig = figure(1);
                set(hFig, 'Position', [100 100 f_xsize f_ysize])%
                set(gca, 'xlim', [-1.1 1.1])
                set(gca, 'ylim', [-1.1 1.1])
                set(gca,'xcolor','w','ycolor','w','xtick',[],'ytick',[],'box','off');
            elseif Fdisp == 1
                disp(['Searchlight at ' num2str(PlotTime(y(T))) 'ms.'])
            end;
            g = 0;
            if ~isempty(light_mode(light_mode==1)) %searchlight through all single electrodes
                Light_AccM=nan(length(Ecluster{clust}),1); g = g+1;
                if Fdisp==2
                    title(['Searchlight results - single electrodes. ' TitStr])
                end;
                
                for c = 1:length(Ecluster{clust}) %run through each electrode separately
                    
                    [Light_acc] = AGF_EASY_SVM( SearchlightTraining, SearchlightTest, SensorInformation(y(T),:,c), UseY, IterNum, oldschool, 0, [], add_arguments,DoAvg,avgoverN);
                    Light_AccM(c) = mean(Light_acc); %average accuracy for current electrode
                    if c > 1
                        %delete([h(c-1) h1(c-1)])
                    end;
                    if Fdisp==2
                        h(c) = text(-ay(c),ax(c),az(c),[indata.chanlocs(Ecluster{clust}(c)).labels],'HorizontalAlignment','center','VerticalAlignment','middle');
                        ht(c) = text(-ay(c),ax(c)-0.03,az(c),[ num2str(round(1000*Light_AccM(c))/10) '%'],'HorizontalAlignment','center','VerticalAlignment','middle');
                        pause(0.000001)
                    end;
                end;
                if Fdisp == 2
                    pause(0.2)
                    delete([h(1:c) ht(1:c)]); clear h ht
                end;
                glob(g,:) = Light_AccM; Odata.Searchlight.single_electrodes(T,:,:) = glob(g,:);
            end;

            if ~isempty(light_mode(light_mode==2)) %searchlight through contralateral electrodes
                taken=[]; Light_AccM2=nan(length(Ecluster{clust}),1);g = g+1;
                if Fdisp==2
                    title(['Searchlight results - contralateral electrodes. ' TitStr])
                end;
                for c = 1:length(Ecluster{clust}) %run through each electrode separately
                    if ay(c) ~= 0 & isempty(find([taken]==c)) & ~isempty(find([ay]==-ay(c) & [ax]==ax(c)))%lateral electrode
                        E2 = find([ay]==-ay(c) & [ax]==ax(c));
                        taken = [taken c E2];
                        [Light_acc] = AGF_EASY_SVM( SearchlightTraining, SearchlightTest, SensorInformation(y(T),:,[c E2]), UseY, IterNum, oldschool, 0, [], add_arguments,DoAvg,avgoverN);
                        Light_AccM2(c) = mean(Light_acc); %average accuracy for current electrode
                        Light_AccM2(E2) = Light_AccM(c);
                        if Fdisp==2
                            h(c) = text(-ay(c),ax(c),az(c),[indata.chanlocs(Ecluster{clust}(c)).labels],'HorizontalAlignment','center','VerticalAlignment','middle');
                            ht(c) = text(-ay(c),ax(c)-0.03,az(c),[ num2str(round(1000*Light_AccM(c))/10) '%'],'HorizontalAlignment','center','VerticalAlignment','middle');
                            h2(c) = text(-ay(E2),ax(E2),az(E2),[indata.chanlocs(Ecluster{clust}(E2)).labels],'HorizontalAlignment','center','VerticalAlignment','middle');
                            ht2(c) = text(-ay(E2),ax(E2)-0.03,az(E2),[ num2str(round(1000*Light_AccM(E2))/10) '%'],'HorizontalAlignment','center','VerticalAlignment','middle');
                            pause(0.01)
                        end;
                    end;
                end;
                if Fdisp == 2
                    pause(0.2)
                    delete([h(:) ht(:) h2(:) ht2(:)])
                end;
                glob(g,:) = Light_AccM2; Odata.Searchlight.contralateral_electrodes(T,:,:) = glob(g,:);
            end;
            
            if ~isempty(light_mode(light_mode==3)) %searchlight through cluster of electrodes
                Odata.lightinfo.n_neighbor  = n_neighbor;
                taken=[]; Light_AccM3=nan(length(Ecluster{clust}),length(Ecluster{clust}));g = g+1;
                if Fdisp==2
                    title(['Searchlight results - clustered electrodes. ' TitStr])
                end;
                for c = 1:length(Ecluster{clust})
                    %Get closes electrodes
                    [~,~,~,~, enum] = agf_matchchans(indata.chanlocs(Ecluster{clust}),indata.chanlocs(Ecluster{clust}(c)),'noplot');
                    PE              = enum(2:n_neighbor+1);
                    [Light_acc] = AGF_EASY_SVM( SearchlightTraining, SearchlightTest, SensorInformation(y(T),:,PE), UseY, IterNum, oldschool, 0, [], add_arguments,DoAvg,avgoverN);
                    Light_AccM3(c,c) = mean(Light_acc); %average accuracy for current cluster
                    if Fdisp == 2
                        if c > 1
                            delete([h1(c-1) h(1:length(PE))])
                        end;
                        h1(c) = text(-ay(c),ax(c),az(c),[indata.chanlocs(Ecluster{clust}(c)).labels],'HorizontalAlignment','center','VerticalAlignment','middle','color','k','fontWeight','bold');
                        ht(c) = text(-ay(c),ax(c)-0.03,az(c),[ num2str(round(1000*Light_AccM(c))/10) '%'],'HorizontalAlignment','center','VerticalAlignment','middle');
                    end;
                    for i = 1 : length(PE)
                        if Fdisp == 2
                            h(i)    = text(-ay(PE(i)),ax(PE(i)),indata.chanlocs((Ecluster{clust}(PE(i)))).labels,'HorizontalAlignment','center','VerticalAlignment','middle','color','g');
                        end;
                        Light_AccM3(c,PE(i)) = mean(Light_acc);
                    end;
                    pause(0.00001)
                end;
                
                if Fdisp == 2
                    pause(0.2)
                    delete([h1(:) ht(:)]); delete(h(:))
                end;
                glob(g,:) = nanmean(Light_AccM3,2); Odata.Searchlight.clustered_electrodes(T,:,:) = glob(g,:);
            end;
            
            %Create global accuracy map from all run comparisons
            Odata.Searchlight.combined(T,:) = nanmean(glob);
            if Fdisp==2
                title(['Searchlight results - average map over all methods. ' TitStr])
                M = ceil(max(nanmean(glob))*10)/10;
                M = max(nanmean(glob));
                topoplot(nanmean(glob), indata.chanlocs(Ecluster{clust}), 'maplimits',[0.5 M], 'style', 'map');%'maplimits',[1-M M], 
                colormap(Fcmap); hcb=colorbar;
                pause(0.5)
            end;
        end;
    end;
end;

Odata.Weights                   = Weights;
Odata.WeightsWeights_at_peak    = Weights_at_peak;

if isempty(TF)
    [MA,y]=max(nanmean(Accuracy(1,:,:),3));
    Odata.peakDecoding              = PlotTime(y);
    Odata.peakAccuracy              = round(MA*1000)/10;
else
    [MA,y]=max(Accuracy);
    Odata.peakDecoding              = PlotTime(y);
    Odata.peakAccuracy              = round(MA*1000)/10;
end

% if ~isempty(activate_ica)
%     IC_Averaged_Accuracy = nanmean(squeeze(nanmean(ICA_accuracy(:,Mark_ICA,n_fold),3)),2);
%     [x,y] = sort(IC_Averaged_Accuracy,'descend');
%     Odata.ICA.Averaged_Accuracy = x;
%     Odata.ICA.Number = y;
%     Odata.ICA.Accuracy = squeeze(nanmean(ICA_accuracy(:,:,n_fold),3));
% end;

return;