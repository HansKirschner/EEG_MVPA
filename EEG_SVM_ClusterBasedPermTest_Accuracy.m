function EEG_SVM_ClusterBasedPermTest_Accuracy(loadpath,NPermutations,threshold,FigNr)
%% Function that plots results of decoding accuracy using the EEG_SVM toolbox
% 
% Input: 
% loadpath      --> path to data (output files of decoding analyses
% NPermutations --> Should be at least 1000 for reasonably stable results.
% threshold     --> threshold for significance
% FigNr         --> Number for the plot
%
% Cluster-based permutation analysis
% Core question: What is the likelihood of the data (i.e., the t-mass) if the data was
% a random sample from the null distribution?

files = dir([loadpath '*.mat']);
Nsub = length(files);

% load first subject to extract info for pre-allocation
VP                  = load([loadpath files(1).name]);
FN                  = fieldnames(VP);
Info                = VP.(FN{1});
Ntp                 = length(Info.info.times);
Nitr                = Info.info.n_folding;
chancelvl           = .5;
TimeWindow          = Info.info.times;
AverageAccuracy_H1  = nan(Nsub,Ntp);

CorrectedWeightMap  = nan(Nsub,length(Info.info.ecluster{1}));
Searchlight         = nan(Nsub,length(Info.info.ecluster{1}));
PeakTime            = nan(Nsub,1);
PeakAccuracy        = nan(Nsub,1);

chanlocs            = Info.chanlocs;

AverageAccuracy_H0 = nan(NPermutations,Nsub,Ntp);


tic
% 1. How many clusters are expected under H0?
disp('Loading data and calculating empirical H0... might be slow...')
for sub = 1:Nsub

    % Load data and use a little hack to make the function work for
    % different inputs
    VP = load([loadpath files(sub).name]);
    FN = fieldnames(VP);
    Info  = VP.(FN{1});
    Ntp = length(Info.info.times);

    %extract some info for across subject ploting
    AverageAccuracy_H1(sub,:) = Info.Accuracy;
    CorrectedWeightMap(sub,:) = Info.CorrectedWeights_at_peak;
    Searchlight(sub,:)        = Info.Searchlight.combined;
    PeakTime(sub)             = Info.peakDecoding;
    PeakAccuracy(sub)         = Info.peakAccuracy;

    for permt = 1: NPermutations

        DecodingAccuracy = zeros(Nitr,Ntp);

        for itr = 1:Nitr
            % assign random target ID for Permutation Testing
            Answer = squeeze(shuffle(Info.PT_Info.Truth(1,itr,:)));
            for tp = 1:Ntp
                prediction = squeeze(Info.PT_Info.model(tp,itr,:)); % this is predictions from models
                Err = Answer - prediction;
                ACC = mean(Err==0);
                DecodingAccuracy(itr,tp) = ACC; % average decoding accuracy
            end
        end

        AverageAccuracy_H0(permt,sub,:) = mean(DecodingAccuracy,1); % average across iteration

    end %End of subject
    clear Info

    if floor(rem(sub,(Nsub/10))) == 0
        progress = round(100*(sub/Nsub));
        whatToPrint = strcat(num2str(progress),'% is done.');
        disp(whatToPrint)
        toc
        tic
    end
end

disp('Now comparing the t-mass of the observed data against the null distribution...')

for permt = 1:NPermutations

    AverageAccuracy_H0_temp = squeeze(AverageAccuracy_H0(permt,:,:));
   
    Ps = nan(2,size(AverageAccuracy_H0_temp,2));
    [~,P_PT,~,STATS_PT] =  ttest(AverageAccuracy_H0_temp,chancelvl,'tail','right'); %Test Against chance-leve
    Ps(1,:) = STATS_PT.tstat;
    Ps(2,:) = P_PT;

    % find significant points
    candid = Ps(2,:) <= threshold;
    candid_marked = zeros(1,length(candid));
    candid_marked(1,1) = candid(1,1);
    candid_marked(1,length(candid)) = candid(1,length(candid));

    %remove orphan time points
    for i = 2:Ntp-1
        if candid(1,i-1) == 0 && candid(1,i) ==1 && candid(1,i+1) ==0
            candid_marked(1,i) = 0;
        else
            candid_marked(1,i) = candid(1,i);
        end
    end

    % combine whole time range with relevent time & significant information
    clusters = candid_marked; % significant or not
    clusterT = Ps(1,:);  % t values

    %%find how many clusters there are, and compute summed T of each cluster
    tmp = zeros(10,25); % creates a matrix with arbitrary size (n cluster x size of each cluster)
    cl = 0;
    member = 0;
    for i = 2:length(clusters)-1
        if clusters(i-1) ==0 && clusters(i) == 1 && clusters(i+1) == 1
            cl = cl+1;
            member = member +1;
            tmp(cl,member) = i;
        elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 0
            if i == 2
                cl = cl +1;
                member = member +1;
                tmp(cl,member) = i;
                member = 0;
            else
                member = member +1;
                tmp(cl,member) = i;
                member = 0;
            end
        elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 1
            if i ==2
                cl = cl+1;
                member = member +1;
                tmp(cl,member) = i;
            else
                member = member +1;
                tmp(cl,member) = i;
            end
        else
        end
    end

    HowManyClusters = cl;
    a = tmp(1:cl,:);
    eachCluster = a(:,logical(sum(a,1)~=0));

    %now, compute summed T of each cluster
    dat_clusterSumT = zeros(HowManyClusters,1);
    for c = 1:HowManyClusters
        dat_clusterSumT(c,1) = sum(clusterT(eachCluster(c,eachCluster(c,:) ~=0)));
    end
    if size(dat_clusterSumT,1) > 0 % if there is at least one signifiant cluster
        permutedT(1,permt) = max(dat_clusterSumT);
    else
        permutedT(1,permt) = 0;
    end
   
end % end of simulation

permutedT = sort(permutedT);

% 2. Perform cluster mass analyses part 2 - how many clusters in the original
% time series?
% Do t-test at each time point of interest
% Find contiguous time points of significant t values
% Sum up the t values within the time points (t-mass)

[H_H0,P_2,~,STATS_2] = ttest(AverageAccuracy_H1,chancelvl,'tail','right');

Ps_2 = nan(2,Ntp);
Ps_2(1,:) = STATS_2.tstat;
Ps_2(2,:) = P_2;

% find significant time points
candid = Ps_2(2,:) <= threshold;

%remove orphan time points
candid_woOrphan = candid;
candid_woOrphan(1,1) = candid(1,1);
for i = 2:(size(tp,2)-1)
    if candid(1,i-1) == 0 && candid(1,i) ==1 && candid(1,i+1) ==0
        candid_woOrphan(1,i) = 0;
    else
        candid_woOrphan(1,i) = candid(1,i);
    end
end

% combine whole time range with relevent time & significant information
clusters = candid_woOrphan;
clusterT = Ps_2(1,:);

%%find how many clusters are there, and compute summed T of each cluster
tmp = zeros(10,25); % creates a matrix with arbitrary size (n cluster x size of each cluster)
cl = 0;
member = 0;
for i = 2:length(clusters)-1
    if clusters(i-1) ==0 && clusters(i) == 1 && clusters(i+1) == 1
        cl = cl+1;
        member = member +1;
        tmp(cl,member) = i;
    elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 0
        if i == 2
            cl = cl +1;
            member = member +1;
            tmp(cl,member) = i;
            member = 0;
        else
            member = member +1;
            tmp(cl,member) = i;
            member = 0;
        end
    elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 1
        if i ==2
            cl = cl+1;
            member = member +1;
            tmp(cl,member) = i;
        else
            member = member +1;
            tmp(cl,member) = i;
        end
    else
    end
end
HowManyClusters = cl;
a = tmp(1:cl,:); % subset significant clusters
eachCluster = a(:,logical(sum(a,1)~=0)); % cut the size at the maximum cluster

%now, compute summed T of each cluster
dat_clusterSumT = nan(HowManyClusters,1);
for c = 1:HowManyClusters
    dat_clusterSumT(c,1) = sum(clusterT(eachCluster(c,eachCluster(c,:) ~=0)));
end

% Compare the t-mass of the observed data against the null
% distribution and compute the likelihood of the data (i.e., p-value)
% find critical t-value
cutOff = NPermutations - NPermutations * (threshold/2); %2-sided test
critT = permutedT(round(cutOff)); % t-mass of top 10%
sigCluster = abs(dat_clusterSumT) > abs(critT);


%% plot resutls
Fcmap       = load('AGF_cmap.mat');
cbColors    = [0 0 0; 230 159 0; 86 180 233; 0 158 115; 240 228 66;...
              0 114 178; 213 94 0; 204 121 167]./256;

figure(FigNr);clf;
ax(1) = subplot(3,2,1);%corrected weights
topoplotIndie(mean(CorrectedWeightMap,1),chanlocs);
colormap(ax(1),Fcmap.AGF_cmap)
colorbar
title(["Corrected activation pattern in sensor space" "@peak decoding"])
ax(2) = subplot(3,2,2); %searchlight
topoplotIndie(mean(Searchlight,1),chanlocs);
colormap(ax(2),Fcmap.AGF_cmap(34:end,:))
colorbar
title(["Searchlight @peak decoding"])

ax(3) = subplot(3,1,2);hold on %Accuracy
Timecourse=squeeze(nanmean(AverageAccuracy_H1, 1));
SEM=squeeze(nanstd(AverageAccuracy_H1,1)./sqrt(Nsub));
H=shadedErrorBar(TimeWindow,Timecourse',SEM',{'-o', 'color', cbColors(2,:), 'markerfacecolor',cbColors(2,:), 'markerSize', 4, 'lineWidth', 1});
TC = mean(AverageAccuracy_H1);
TW = TimeWindow;
% draw clusters
draw = eachCluster(sigCluster,:);
draw = sort(reshape(draw,1,size(draw,1)*size(draw,2)));
draw = draw(draw>0);
w = zeros(Ntp,1);
w(draw)=1;
a = area(TW, TC.*w');
a.EdgeColor = 'none';
a.FaceColor = [0.8,0.8,0.8];
child = get(a,'Children');
set(child,'FaceAlpha',0.9)
ylabel('Accuracy')
xlabel('Time (ms)')
title(['Corrected Accuracy: peak-time = ' num2str(round(mean(PeakTime))) ' (' num2str(round(std(PeakTime))) ') ms; peak accuracy = ' ...
    num2str(round(mean(PeakAccuracy))) ' (' num2str(round(std(PeakAccuracy))) ') %'])
set(gca,'box', 'off')
yline(.5,LineStyle="--",LineWidth=1)
ylim([chancelvl-.05 max(Timecourse)+.1])

ax(4) = subplot(3,1,3);hold on %Accuracy
shade_the_back(H_H0, [183 183 183]./255, TimeWindow);
H=shadedErrorBar(TimeWindow,Timecourse',SEM',{'-o', 'color', cbColors(3,:), 'markerfacecolor',cbColors(3,:), 'markerSize', 4, 'lineWidth', 1});
ylabel('Accuracy')
xlabel('Time (ms)')
title(['Uncorrected Accuracy'])
yline(.5,LineStyle="--",LineWidth=1)
ylim([chancelvl-.05 max(Timecourse)+.1])

