rng(1);
load features.mat
allNames = T.Properties.VariableNames(1:10);
timeF = {'RMS','ZCR','DynRange','NoiseFloor'};
specF = {'Centroid','Flatness','Band1','Band2','Band3','Band4'};
sets = {allNames, timeF, specF};
setNames = ["All","Time-domain","Spectral"];
noises = ["babble","car","street"];
models = ["RandomForest","SVM","kNN"];
Y = T.Level;
rows = {};
for s = 1:3
    X = table2array(T(:, sets{s}));
    for m = models
        accs = zeros(1,3);
        for n = 1:3
            te = T.NoiseType == noises(n); tr = ~te;
            if m == "RandomForest"
                mdl = TreeBagger(100, X(tr,:), Y(tr), 'Method','classification');
                pred = categorical(predict(mdl, X(te,:)), categories(Y));
            elseif m == "SVM"
                mdl = fitcecoc(X(tr,:), Y(tr), 'Learners', templateSVM('Standardize',true));
                pred = predict(mdl, X(te,:));
            else
                mdl = fitcknn(X(tr,:), Y(tr), 'NumNeighbors',5, 'Standardize',true);
                pred = predict(mdl, X(te,:));
            end
            accs(n) = 100*mean(pred == Y(te));
        end
        rows(end+1,:) = {char(setNames(s)), char(m), accs(1), accs(2), accs(3), mean(accs)};
    end
end
res = cell2table(rows, 'VariableNames', {'Features','Model','Babble','Car','Street','Mean'});
disp(res); writetable(res,'comparison.csv');