files = dir('**/*.wav');
n = numel(files);
Feat = zeros(n,10);
NoiseType = strings(n,1);
SNR = zeros(n,1);

for k = 1:n
    [x, fs] = audioread(fullfile(files(k).folder, files(k).name));
    x = x(:,1);

    % noise type and SNR from folder name, e.g. babble_0dB
    tok = regexp(files(k).folder, '(babble|car|street)_(\d+)dB', 'tokens', 'once');
    NoiseType(k) = string(tok{1});
    SNR(k) = str2double(tok{2});

    % frame energies (25 ms frames, 10 ms hop)
    fl = round(0.025*fs); hop = round(0.010*fs);
    fr = buffer(x, fl, fl-hop, 'nodelay');
    fe = 10*log10(sum(fr.^2,1) + eps);
    noiseFloor = prctile(fe,10);
    dynRange = prctile(fe,90) - noiseFloor;

    % time-domain features
    r = rms(x);
    zcr = mean(abs(diff(sign(x))) > 0);

    % spectral features
    [pxx, f] = pwelch(x, hamming(256), 128, 512, fs);
    centroid = sum(f.*pxx) / sum(pxx);
    flatness = exp(mean(log(pxx+eps))) / mean(pxx);
    tot = bandpower(pxx, f, [0 fs/2], 'psd');
    b1 = bandpower(pxx, f, [0 500], 'psd') / tot;
    b2 = bandpower(pxx, f, [500 1000], 'psd') / tot;
    b3 = bandpower(pxx, f, [1000 2000], 'psd') / tot;
    b4 = bandpower(pxx, f, [2000 fs/2], 'psd') / tot;

    Feat(k,:) = [r zcr centroid flatness dynRange noiseFloor b1 b2 b3 b4];
end

Level = strings(n,1);
Level(SNR==0) = "High";
Level(SNR==5 | SNR==10) = "Medium";
Level(SNR==15) = "Low";

T = array2table(Feat, 'VariableNames', {'RMS','ZCR','Centroid','Flatness', ...
    'DynRange','NoiseFloor','Band1','Band2','Band3','Band4'});
T.NoiseType = NoiseType; T.SNR = SNR; T.Level = categorical(Level);
save('features.mat','T');
writetable(T,'features.csv');
disp(T(1:5,:)); tabulate(T.Level)