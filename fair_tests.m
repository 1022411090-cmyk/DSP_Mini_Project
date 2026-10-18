load features.mat
predNames = T.Properties.VariableNames(1:10);
X = table2array(T(:,predNames)); Y = T.Level;

% Test 1: split by sentence
tr = T.Sentence <= 20; te = ~tr;
rf = TreeBagger(100, X(tr,:), Y(tr), 'Method','classification');
pred = categorical(predict(rf, X(te,:)), categories(Y));
fprintf('Sentence-split accuracy: %.2f %%\n', 100*mean(pred==Y(te)));
figure; confusionchart(Y(te), pred); title('Sentence-split');

% Test 2: unseen noise (train street+babble, test car)
tr = T.NoiseType ~= "car"; te = ~tr;
rf = TreeBagger(100, X(tr,:), Y(tr), 'Method','classification');
pred = categorical(predict(rf, X(te,:)), categories(Y));
fprintf('Unseen-noise (car) accuracy: %.2f %%\n', 100*mean(pred==Y(te)));
figure; confusionchart(Y(te), pred); title('Unseen noise: car');