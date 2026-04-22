%% =====================================================
%  BİRLEŞİK TRANSFER LEARNING
%  Pre-trained ResNet18 + Kaggle + real_world + RealWaste(%50)
%  Sıfırdan tutarlı eğitim
% =====================================================

clc;
clear;
close all;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       🚀 BİRLEŞİK TRANSFER LEARNING EĞİTİMİ                   ║\n');
fprintf('║   Pre-trained ResNet18 + Tüm Veri Kaynakları                  ║\n');
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
outputModelPath = fullfile(basePath, "trainedNet.mat");

% Kaggle sınıf klasörleri
classNames = {'battery', 'biological', 'cardboard', 'clothes', 'glass', ...
              'metal', 'paper', 'plastic', 'shoes', 'trash'};
classFolders = fullfile(basePath, classNames);

customReadFcn = @(filename) readAndConvertImage(filename);
inputSize = [224 224 3];
numClasses = numel(classNames);

%% 1️⃣ PRE-TRAINED RESNET18 YÜKLE
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🧠 PRE-TRAINED RESNET18                        \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

net = resnet18;
lgraph = layerGraph(net);
fprintf('✅ Pre-trained ResNet18 yüklendi (ImageNet ağırlıkları)\n\n');

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
kaggleCounts = countEachLabel(kaggleDS);
disp(kaggleCounts);

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
realWorldCounts = countEachLabel(realWorldDS);
disp(realWorldCounts);

%% 4️⃣ TESTREALWASTE DATASET (%50)
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                    📷 TESTREALWASTE (%50)                         \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% TestRealWaste klasör isimleri zaten doğru (sadece battery ve shoes yok)
testRealWasteDS = imageDatastore(testRealWastePath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}, ...
    'ReadFcn', customReadFcn);

% Etiketleri model sınıflarıyla uyumlu hale getir
testRealWasteDS.Labels = categorical(cellstr(testRealWasteDS.Labels), classNames);

fprintf('📷 TestRealWaste toplam: %d fotoğraf\n', numel(testRealWasteDS.Files));

% %50 eğitim için ayır
[trainRealWasteDS, testHoldoutDS] = splitEachLabel(testRealWasteDS, 0.5, 'randomized');

fprintf('   Eğitim için: %d fotoğraf (%%50)\n', numel(trainRealWasteDS.Files));
fprintf('   Test için ayrıldı: %d fotoğraf (%%50)\n\n', numel(testHoldoutDS.Files));

% Test holdout kaydet
testHoldoutFiles = testHoldoutDS.Files;
testHoldoutLabels = testHoldoutDS.Labels;
save(fullfile(basePath, 'testRealWaste_holdout.mat'), 'testHoldoutFiles', 'testHoldoutLabels');
fprintf('📦 Test verisi kaydedildi: testRealWaste_holdout.mat\n');

%% 5️⃣ TÜM VERİLERİ BİRLEŞTİR
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🔗 VERİLERİ BİRLEŞTİRME                        \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Tüm dosya ve etiketleri birleştir
allFiles = [kaggleDS.Files; realWorldDS.Files; trainRealWasteDS.Files];
allLabels = [kaggleDS.Labels; realWorldDS.Labels; trainRealWasteDS.Labels];

% Birleşik datastore
combinedDS = imageDatastore(allFiles, 'Labels', allLabels, 'ReadFcn', customReadFcn);

fprintf('📊 Birleştirilmiş Eğitim Verisi:\n');
fprintf('   Kaggle:               %d\n', numel(kaggleDS.Files));
fprintf('   real_world:           %d\n', numel(realWorldDS.Files));
fprintf('   TestRealWaste (%%50):  %d\n', numel(trainRealWasteDS.Files));
fprintf('   ─────────────────────────\n');
fprintf('   TOPLAM:               %d\n\n', numel(combinedDS.Files));

combinedCounts = countEachLabel(combinedDS);
disp(combinedCounts);

%% 6️⃣ TRAIN/VAL/TEST SPLIT
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
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

fprintf('✅ Augmentation hazır (Rotation, Translation, Scale, Flip)\n\n');

%% 8️⃣ MODEL HAZIRLAMA (Transfer Learning)
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🧠 MODEL HAZIRLAMA                             \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Son katmanları değiştir (10 sınıf için)
lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'classOutput_new')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fc_new');

% Not: Yeni katmanlar 10x hızlı öğrenir (WeightLearnRateFactor=10)
% Eski katmanlar düşük learning rate ile yavaş güncellenir

fprintf('✅ Transfer Learning modeli hazır:\n');
fprintf('   - Pre-trained ResNet18 (ImageNet)\n');
fprintf('   - Yeni FC katman 10x hızlı öğrenir\n');
fprintf('   - Son FC katman yeni (10 sınıf)\n\n');

%% 9️⃣ EĞİTİM
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    🚀 EĞİTİM BAŞLIYOR                             \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
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
fprintf('   • Epochs: 20\n');
fprintf('   • Learning Rate: 1e-4\n');
fprintf('   • Batch Size: 32\n');
fprintf('   • Toplam eğitim verisi: %d\n\n', numel(trainDS.Files));

tic;
trainedNet = trainNetwork(augTrainDS, lgraph, options);
trainingTime = toc;

fprintf('\n✅ Eğitim tamamlandı! Süre: %.1f dakika\n\n', trainingTime/60);

%% 🔟 MODEL KAYDETME
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                    💾 MODEL KAYDETME                              \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% Eski modeli yedekle
backupPath = fullfile(basePath, sprintf('trainedNet_backup_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
if isfile(outputModelPath)
    copyfile(outputModelPath, backupPath);
    fprintf('📦 Eski model yedeklendi: %s\n', backupPath);
end

% Yeni modeli kaydet
save(outputModelPath, 'trainedNet');
fprintf('✅ Yeni model kaydedildi: trainedNet.mat\n\n');

% Orijinal (fine-tuning öncesi) olarak da kaydet
save(fullfile(basePath, 'trainedNet_combined.mat'), 'trainedNet');
fprintf('✅ Ayrıca kaydedildi: trainedNet_combined.mat\n\n');

%% ÖZET
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    📊 EĞİTİM ÖZETİ                             ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Model: Pre-trained ResNet18 + Transfer Learning              ║\n');
fprintf('║                                                                ║\n');
fprintf('║  Eğitim Verisi:                                                ║\n');
fprintf('║    • Kaggle:              %5d fotoğraf                       ║\n', numel(kaggleDS.Files));
fprintf('║    • real_world:          %5d fotoğraf                       ║\n', numel(realWorldDS.Files));
fprintf('║    • TestRealWaste (50%%): %5d fotoğraf                       ║\n', numel(trainRealWasteDS.Files));
fprintf('║    • TOPLAM:              %5d fotoğraf                       ║\n', numel(combinedDS.Files));
fprintf('║                                                                ║\n');
fprintf('║  Eğitim Süresi: %.1f dakika                                   ║\n', trainingTime/60);
fprintf('║                                                                ║\n');
fprintf('║  ⚠️  Test için ayrılan: %d fotoğraf                           ║\n', numel(testHoldoutDS.Files));
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('📱 GUI ile test: atik_gui\n');
fprintf('📊 Model karşılaştırma: export_for_colab\n');

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
