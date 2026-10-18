load features.mat
files = dir('**/*.wav');
S = zeros(height(T),1);
for k = 1:numel(files)
    t = regexp(files(k).name, 'sp(\d+)', 'tokens', 'once');
    S(k) = str2double(t{1});
end
T.Sentence = S;
save('features.mat','T');
disp(unique(S)')