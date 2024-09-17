function EEG_SVM_ClusterBasedPermutationTest_TG(loadpath,numPerm,threshold,FigNr)
%% Function that plots statistic map for temporal generalization of decoding models
% 
% Input: 
% loadpath      --> path to data (output files of decoding analyses
% numPerm       --> Should be at least 1000 for reasonably stable results.
% threshold     --> threshold for significance
% FigNr         --> Number for the plot

% set up data:
addInfo         = false; %if true this plots some additional info

% how many files do we have and what is their name?
files = dir([loadpath '*.mat']);
Nsub = length(files);

% load first subject to extract info for pre-allocation
VP                  = load([loadpath files(1).name]);
FN                  = fieldnames(VP);
Info                = VP.(FN{1});
Ntp                 = length(Info.info.times);
TimeWindow          = Info.info.times;
TG_H1               = nan(Nsub,Ntp,Ntp);
alpha               = threshold;

rng('shuffle')

% initialize null hypothesis maps
permmaps = nan(numPerm*Nsub,Ntp,Ntp);

tic

disp('Loading data and calculating empirical H0... might be slow...')

for sub = 1:Nsub

    VP = load([loadpath files(sub).name]);
    FN = fieldnames(VP);
    Info  = VP.(FN{1});
    TG_H1(sub,:,:) = Info.Error_Decoding_time_generalization;

    Temp_TG_Map = nan(Ntp,Ntp);
    for permi=1:numPerm

        for tp = 1:Ntp
            % create a subject length array containing randomly assigned
            % labels:
            permArray=shuffle(Info.GT_TG);
            Err =  permArray - Info.Predicted_Class_times(tp).Predicted_Class;
            ACC = mean(Err==0,2)';
            Temp_TG_Map(tp,:) = ACC;
        end

        permmaps(permi+(numPerm*sub-numPerm),:,:) = Temp_TG_Map;
    end

    if floor(rem(sub,(Nsub/10))) == 0
        progress = round(100*(sub/Nsub));
        whatToPrint = strcat(num2str(progress),'% is done.');
        disp(whatToPrint)
        toc
        tic
    end

end

% plot permutation maps to check distibution - these are expected to center
% around zero and approach gausian
if addInfo
    figure(1);clf;
    subplot(2,1,1)
    histogram(permmaps(:,randperm(Ntp,1),randperm(Ntp,1)));
    title('PermMap at random pixel in TG map')
    subplot(2,1,2)
    histogram(permmaps(:,randperm(Ntp,1),randperm(Ntp,1)));
    title('PermMap at random pixel in TG map')
end


%% compute z- and p-values based on normalized distance to H0 distributions (per pixel)
% p-value
pval = alpha;

% convert p-value to Z value
% if you don't have the stats toolbox, set zval=1.6449;
zval = abs(norminv(pval));

% compute mean and standard deviation maps
mean_h0 = squeeze(mean(permmaps));
std_h0  = squeeze(std(permmaps));


if addInfo
    figure(2);clf;
    subplot(2,1,1)
    imagesc(mean_h0)
    axis square
    colorbar
   
    title('H0 Map')
    subplot(2,1,2)
    imagesc(std_h0)
    axis square
    
    title('STD H0 Map')
end

% now threshold real data...
% first Z-score
zmap = (squeeze(mean(TG_H1))-mean_h0) ./ std_h0;

% threshold image at p-value, by setting subthreshold values to 0
zmap(zmap<zval) = 0;



%% cluster correction

% initialize matrices for cluster-based correction
max_cluster_sizes = zeros(1,numPerm);

max_cluster_mass = zeros(1,numPerm);


% loop through permutations
for permi = 1:numPerm
    
    % take each permutation map, and transform to Z
    threshimg = squeeze(mean(permmaps(permi,:,:,:)));
    threshimg = (threshimg-mean_h0)./std_h0;
    
    % threshold image at p-value
    threshimg(threshimg<zval) = 0;
    
    
    % find clusters (need image processing toolbox for this!)
    islands = bwconncomp(threshimg);
    if numel(islands.PixelIdxList)>0
        
        % count sizes of clusters
        tempclustsizes = cellfun(@length,islands.PixelIdxList);

        % store size of biggest cluster
        max_cluster_sizes(permi) = max(tempclustsizes);
            
        tempclustmasses = [];
        for k = 1:numel(islands.PixelIdxList)
        
            tempclustmasses(k) = length(islands.PixelIdxList{k})*abs(nanmean(threshimg(islands.PixelIdxList{k})));

        end

         % store mass of the biggest cluster
        max_cluster_mass(permi) = max(tempclustmasses);
    end
end


% find cluster threshold (need image processing toolbox for this!)
% based on p-value and null hypothesis distribution
cluster_thresh      = prctile(max_cluster_sizes,100-(100*threshold));

cluster_thresh_mass = prctile(max_cluster_mass,100-(100*threshold));

% now find clusters in the real thresholded zmap
% if they are "too small" set them to zero
islands = bwconncomp(zmap);

zmap_mass = zmap;

for i=1:islands.NumObjects
    % if real clusters are too small, remove them by setting to zero!
    if numel(islands.PixelIdxList{i})<cluster_thresh
        zmap(islands.PixelIdxList{i})=0;
    end

    if numel(islands.PixelIdxList{i})*abs(nanmean(zmap_mass(islands.PixelIdxList{i})))<cluster_thresh_mass
        zmap_mass(islands.PixelIdxList{i})=0;
    end
end


%%% now some plotting...
figure(FigNr),clf
subplot(1,3,1)
imagesc(TimeWindow,TimeWindow ,squeeze(mean(TG_H1)))
axis square
colorbar
set(gca,'clim',[0.5 max(max(squeeze(mean(TG_H1))))])
xlabel('Time (ms)'), ylabel('Time (ms)')
title('TG map of raw values')

subplot(1,3,2)
imagesc(TimeWindow,TimeWindow,squeeze(mean(TG_H1)));
axis square
colorbar
hold on
contour(TimeWindow,TimeWindow,logical(zmap),1,'linecolor','k');
set(gca,'clim',[0.5 max(max(squeeze(mean(TG_H1))))])
xlabel('Time (ms)'), ylabel('Time (ms)')
title('significance regions based on cluster size')


subplot(1,3,3)
imagesc(TimeWindow,TimeWindow,zmap);
xlabel('Time (ms)'), ylabel('Time (ms)')
title('z-map, thresholded based on size')
colorbar

return;

