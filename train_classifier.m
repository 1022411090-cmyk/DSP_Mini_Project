load features.mat          % loads table T
predNames = T.Properties.VariableNames(1:10);
X = table2array(T(:, predNames));
Y = T.Level;

rng(1);
cv  = cvpartition(Y, 'HoldOut', 0.3);
Xtr = X(training(cv),:);  Ytr = Y(training(cv));
Xte = X(test(cv),:);      Yte = Y(test(cv));

rf = TreeBagger(100, Xtr, Ytr, 'Method','classification', ...
    'OOBPredictorImportance','on');
pred = categorical(predict(rf, Xte), categories(Y));

acc = mean(pred == Yte);
C = confusionmat(Yte, pred);
prec = diag(C) ./ sum(C,1)';
rec  = diag(C) ./ sum(C,2);
f1   = 2*prec.*rec ./ (prec+rec);
fprintf('Accuracy: %.2f %%\n', 100*acc);
fprintf('Macro F1: %.3f\n', mean(f1,'omitnan'));

figure; confusionchart(Yte, pred);
title('Noise-level classifier (Random Forest)');
figure; bar(rf.OOBPermutedPredictorDeltaError);
xticklabels(predNames); xtickangle(45);
title('Feature importance');