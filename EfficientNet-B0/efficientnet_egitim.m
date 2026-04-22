%% =====================================================
%  EfficientNet-B0 TRANSFER LEARNING
%  Kaggle + real_world + TestRealWaste (%50)
%  Sabit holdout kullanılır
% =====================================================

clc;
clear;
close all;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       🚀 EfficientNet-B0 TRANSFER LEARNING                    ║\n');
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
outputModelPath = fullfile(basePath, "trainedNet_efficientnet.mat");  % Farklı dosya

classNames = {'battery', 'biological', 'cardboard', 'clothes', 'glass', ...
              'metal', 'paper', 'plastic', 'shoes', 'trash'};
classFolders = fullfile(basePath, classNames);

customReadFcn = @(filename) readAndConvertImage(filename);
inputSize = [224 224 3];  % EfficientNet-B0 için
numClasses = numel(classNames);

%% 1️⃣ PRE-TRAINED EFFICIENTNET-B0 YÜKLE
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🧠 PRE-TRAINED EfficientNet-B0                 \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

try
    net = efficientnetb0;
    lgraph = layerGraph(net);
    fprintf('✅ Pre-trained EfficientNet-B0 yüklendi (ImageNet ağırlıkları)\n\n');
catch
    fprintf('⚠️  EfficientNet-B0 bulunamadı. Yüklemek için:\n');
    fprintf('   1. Add-On Explorer aç\n');
    fprintf('   2. "Deep Learning Toolbox Model for EfficientNet-B0 Network" ara\n');
    fprintf('   3. Install et\n');
    error('EfficientNet-B0 yüklü değil!');
end

%% 2️⃣ KAGGLE DATASET
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    📦 KAGGLE DATASET                              \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

kaggleDS = imageDatastore(classFolders, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}, ...
    'ReadFcn', customReadFcn);

fprintf('📦 Kaggle: %d fotoğraf\n', numel(kaggleDS.Files));

%% 3️⃣ REAL_WORLD DATASET
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                    📷 REAL_WORLD DATASET                          \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

realWorldDS = imageDatastore(realWorldPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}, ...
    'ReadFcn', customReadFcn);

fprintf('📷 real_world: %d fotoğraf\n', numel(realWorldDS.Files));

%% 4️⃣ TESTREALWASTE - MEVCUT HOLDOUT KULLAN
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                    📷 TESTREALWASTE (Sabit Holdout)               \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Mevcut holdout'u yükle
holdoutPath = fullfile(basePath, 'testRealWaste_holdout.mat');
if ~isfile(holdoutPath)
    error('testRealWaste_holdout.mat bulunamadı!');
end

holdoutData = load(holdoutPath);
testHoldoutFiles = holdoutData.testHoldoutFiles;
fprintf('✅ Mevcut holdout yüklendi: %d fotoğraf\n', numel(testHoldoutFiles));

% Tüm TestRealWaste dosyalarını al
testRealWasteDS = imageDatastore(testRealWastePath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}, ...
    'ReadFcn', customReadFcn);

testRealWasteDS.Labels = categorical(cellstr(testRealWasteDS.Labels), classNames);

% Holdout olmayanları eğitim için kullan
allFiles = testRealWasteDS.Files;
allLabels = testRealWasteDS.Labels;
isHoldout = ismember(allFiles, testHoldoutFiles);
trainFiles = allFiles(~isHoldout);
trainLabels = allLabels(~isHoldout);

trainRealWasteDS = imageDatastore(trainFiles, 'Labels', trainLabels, 'ReadFcn', customReadFcn);
fprintf('   Eğitim için: %d fotoğraf\n\n', numel(trainRealWasteDS.Files));

%% 5️⃣ TÜM VERİLERİ BİRLEŞTİR
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🔗 VERİLERİ BİRLEŞTİRME                        \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

allFiles = [kaggleDS.Files; realWorldDS.Files; trainRealWasteDS.Files];
allLabels = [kaggleDS.Labels; realWorldDS.Labels; trainRealWasteDS.Labels];

combinedDS = imageDatastore(allFiles, 'Labels', allLabels, 'ReadFcn', customReadFcn);

fprintf('📊 Birleştirilmiş Eğitim Verisi:\n');
fprintf('   Kaggle:               %d\n', numel(kaggleDS.Files));
fprintf('   real_world:           %d\n', numel(realWorldDS.Files));
fprintf('   TestRealWaste:        %d\n', numel(trainRealWasteDS.Files));
fprintf('   TOPLAM:               %d\n\n', numel(combinedDS.Files));

%% 6️⃣ TRAIN/VAL SPLIT
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    ✂️  TRAIN/VAL AYIRMA                           \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

[trainDS, valDS] = splitEachLabel(combinedDS, 0.85, 'randomized');
fprintf('📊 Eğitim: %d | Validation: %d\n\n', numel(trainDS.Files), numel(valDS.Files));

%% 7️⃣ DATA AUGMENTATION
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🔄 DATA AUGMENTATION                           \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

imageAugmenter = imageDataAugmenter(...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10], ...
    'RandXScale', [0.95, 1.05], ...
    'RandYScale', [0.95, 1.05], ...
    'RandXReflection', true);

augTrainDS = augmentedImageDatastore(inputSize, trainDS, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augValDS = augmentedImageDatastore(inputSize, valDS, ...
    'ColorPreprocessing', 'gray2rgb');

fprintf('✅ Augmentation hazır\n\n');

%% 8️⃣ MODEL HAZIRLAMA (EfficientNet için)
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🧠 MODEL HAZIRLAMA                             \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% EfficientNet-B0 son katmanlarını bul ve değiştir
lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|dense|MatMul', ...
    'Softmax', 'classification'});

% Yeni katmanları ekle
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'classOutput_new')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool', 'fc_new');

fprintf('✅ EfficientNet-B0 Transfer Learning modeli hazır\n\n');

%% 9️⃣ EĞİTİM
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🚀 EĞİTİM BAŞLIYOR                             \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...  % EfficientNet daha büyük, batch size küçük
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValDS, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Eğitim Parametreleri:\n');
fprintf('   • Model: EfficientNet-B0\n');
fprintf('   • Epochs: 20\n');
fprintf('   • Learning Rate: 1e-4\n');
fprintf('   • Batch Size: 16\n\n');

tic;
trainedNet = trainNetwork(augTrainDS, lgraph, options);
trainingTime = toc;

fprintf('\n✅ Eğitim tamamlandı! Süre: %.1f dakika\n\n', trainingTime/60);

%% 🔟 MODEL KAYDETME
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    💾 MODEL KAYDETME                              \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

save(outputModelPath, 'trainedNet');
fprintf('✅ Model kaydedildi: trainedNet_efficientnet.mat\n\n');

%% ÖZET
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    📊 EĞİTİM ÖZETİ                             ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Model: EfficientNet-B0 + Transfer Learning                   ║\n');
fprintf('║  Eğitim Süresi: %.1f dakika                                   ║\n', trainingTime/60);
fprintf('║  Çıktı: trainedNet_efficientnet.mat                           ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% YARDIMCI FONKSİYON
function img = readAndConvertImage(filename)
    try
        img = imread(filename);
    catch
        try
            info = imfinfo(filename);
            if strcmpi(info.ColorType, 'CMYK')
                img = imread(filename);
                cmyk = double(img) / 255;
                c = cmyk(:,:,1); m = cmyk(:,:,2); y = cmyk(:,:,3); k = cmyk(:,:,4);
                r = (1-c).*(1-k); g = (1-m).*(1-k); b = (1-y).*(1-k);
                img = uint8(cat(3,r,g,b)*255);
            else
                error('Cannot read');
            end
        catch
            warning('Could not read: %s', filename);
            img = uint8(zeros(224, 224, 3));
        end
    end
    
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) == 4
        img = img(:,:,1:3);
    end
end
