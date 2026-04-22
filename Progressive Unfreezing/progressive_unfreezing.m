%% =====================================================
%  PROGRESSIVE UNFREEZING - EfficientNet-B0
%  Domain Shift için kademeli çözme
%  Aşama 1: FC layer eğit
%  Aşama 2: Son blokları aç
% =====================================================

clc;
clear;
close all;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║   🚀 PROGRESSIVE UNFREEZING - EfficientNet-B0                 ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% GPU Kontrolü
gpuAvailable = canUseGPU();
if gpuAvailable
    gpuInfo = gpuDevice;
    fprintf('✅ GPU: %s (%.1f GB)\n\n', gpuInfo.Name, gpuInfo.TotalMemory/1e9);
else
    fprintf('⚠️  GPU bulunamadı, CPU kullanılacak.\n\n');
end

%% Paths
basePath = "C:\Users\eemre\Desktop\garbage-classification-v2";
realWorldPath = fullfile(basePath, "real_world");
testRealWastePath = fullfile(basePath, "TestRealWaste");
outputModelPath = fullfile(basePath, "trainedNet_progressive.mat");

classNames = {'battery', 'biological', 'cardboard', 'clothes', 'glass', ...
              'metal', 'paper', 'plastic', 'shoes', 'trash'};
classFolders = fullfile(basePath, classNames);

customReadFcn = @(filename) readAndConvertImage(filename);
inputSize = [224 224 3];
numClasses = numel(classNames);

%% VERİLERİ YÜKLE (Kaggle + real_world + TestRealWaste %50)
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    📦 VERİ HAZIRLAMA                              \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Kaggle
kaggleDS = imageDatastore(classFolders, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', {'.jpg','.jpeg','.png','.bmp','.gif'}, ...
    'ReadFcn', customReadFcn);
fprintf('📦 Kaggle: %d\n', numel(kaggleDS.Files));

% real_world
realWorldDS = imageDatastore(realWorldPath, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', {'.jpg','.jpeg','.png','.bmp','.gif'}, ...
    'ReadFcn', customReadFcn);
fprintf('📷 real_world: %d\n', numel(realWorldDS.Files));

% TestRealWaste (mevcut holdout kullan)
holdoutPath = fullfile(basePath, 'testRealWaste_holdout.mat');
if isfile(holdoutPath)
    holdoutData = load(holdoutPath);
    testRealWasteDS = imageDatastore(testRealWastePath, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', 'FileExtensions', {'.jpg','.jpeg','.png','.bmp','.gif'}, ...
        'ReadFcn', customReadFcn);
    testRealWasteDS.Labels = categorical(cellstr(testRealWasteDS.Labels), classNames);
    
    allFiles = testRealWasteDS.Files;
    allLabels = testRealWasteDS.Labels;
    isHoldout = ismember(allFiles, holdoutData.testHoldoutFiles);
    trainRealWasteDS = imageDatastore(allFiles(~isHoldout), 'Labels', allLabels(~isHoldout), 'ReadFcn', customReadFcn);
    fprintf('📷 TestRealWaste (train): %d\n', numel(trainRealWasteDS.Files));
else
    error('testRealWaste_holdout.mat bulunamadı!');
end

% Birleştir
allFiles = [kaggleDS.Files; realWorldDS.Files; trainRealWasteDS.Files];
allLabels = [kaggleDS.Labels; realWorldDS.Labels; trainRealWasteDS.Labels];
combinedDS = imageDatastore(allFiles, 'Labels', allLabels, 'ReadFcn', customReadFcn);
fprintf('📊 TOPLAM: %d\n\n', numel(combinedDS.Files));

% Train/Val split
[trainDS, valDS] = splitEachLabel(combinedDS, 0.85, 'randomized');
fprintf('Eğitim: %d | Validation: %d\n\n', numel(trainDS.Files), numel(valDS.Files));

% Augmentation
imageAugmenter = imageDataAugmenter(...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10], ...
    'RandXScale', [0.95, 1.05], ...
    'RandYScale', [0.95, 1.05], ...
    'RandXReflection', true);

augTrainDS = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb');
augValDS = augmentedImageDatastore(inputSize, valDS, 'ColorPreprocessing', 'gray2rgb');

%% EfficientNet-B0 YÜKLE
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🧠 EfficientNet-B0 HAZIRLAMA                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

net = efficientnetb0;
lgraph = layerGraph(net);

% Son katmanları değiştir
lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|dense|MatMul', 'Softmax', 'classification'});
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'classOutput_new')
];
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool', 'fc_new');

fprintf('✅ EfficientNet-B0 hazır\n\n');

%% ═══════════════════════════════════════════════════════════════════
%  AŞAMA 1: SADECE FC KATMANI EĞİT (Düşük global LR, yüksek FC factor)
%  ═══════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('   🔹 AŞAMA 1: Sadece FC Layer Eğitimi (Alt katmanlar donmuş)      \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Düşük global LR + yüksek FC factor = etkili olarak sadece FC eğitilir
% FC layer'ın WeightLearnRateFactor = 10 olarak ayarlandı (yukarıda)
% Global LR çok düşük olacak, FC hızlı öğrenecek

options1 = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 3e-5, ...  % Çok düşük global LR
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValDS, ...
    'ValidationFrequency', 100, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('📊 Aşama 1 Parametreleri:\n');
fprintf('   • Epochs: 5\n');
fprintf('   • Global LR: 3e-5 (çok düşük)\n');
fprintf('   • FC LR Factor: 10x → Efektif FC LR: 3e-4\n');
fprintf('   • Diğer katmanlar: Çok yavaş öğrenir (donmuş gibi)\n\n');

tic;
[trainedNet1, info1] = trainNetwork(augTrainDS, lgraph, options1);
time1 = toc;
fprintf('\n✅ Aşama 1 tamamlandı! (%.1f dk)\n\n', time1/60);

%% ═══════════════════════════════════════════════════════════════════
%  AŞAMA 2: TÜM MODEL İNCE AYAR (Düşük LR)
%  ═══════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('   🔹 AŞAMA 2: Tüm Model İnce Ayar                                 \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Aşama 1'den gelen modeli kullan
lgraph2 = layerGraph(trainedNet1);

options2 = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...  % Düşük LR (ince ayar)
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValDS, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('📊 Aşama 2 Parametreleri:\n');
fprintf('   • Epochs: 10\n');
fprintf('   • LR: 1e-5 (düşük - ince ayar)\n');
fprintf('   • Tüm katmanlar eğitiliyor (düşük LR ile)\n\n');

tic;
[trainedNet, info2] = trainNetwork(augTrainDS, lgraph2, options2);
time2 = toc;
fprintf('\n✅ Aşama 2 tamamlandı! (%.1f dk)\n\n', time2/60);

%% MODEL KAYDET
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    💾 MODEL KAYDETME                              \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

save(outputModelPath, 'trainedNet');
fprintf('✅ Model kaydedildi: trainedNet_progressive.mat\n\n');

%% ÖZET
totalTime = time1 + time2;
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    📊 EĞİTİM ÖZETİ                             ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Yöntem: Progressive Unfreezing                               ║\n');
fprintf('║  Aşama 1: FC eğitimi (5 epoch, LR=1e-3)                       ║\n');
fprintf('║  Aşama 2: Son 50 katman (10 epoch, LR=1e-5)                   ║\n');
fprintf('║  Toplam Süre: %.1f dakika                                     ║\n', totalTime/60);
fprintf('║  Çıktı: trainedNet_progressive.mat                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% YARDIMCI FONKSİYON
function img = readAndConvertImage(filename)
    try
        img = imread(filename);
    catch
        img = uint8(zeros(224, 224, 3));
    end
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) == 4
        img = img(:,:,1:3);
    end
end
