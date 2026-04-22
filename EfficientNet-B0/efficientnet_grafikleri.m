clc;
clear;
close all;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       📊 MODEL KARŞILAŞTIRMA GRAFİKLERİ                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

basePath = "C:\Users\eemre\Desktop\garbage-classification-v2";
customReadFcn = @(filename) readAndConvertImage(filename);

%% Modelleri yükle
fprintf('📂 Modeller yükleniyor...\n');
load(fullfile(basePath, 'trainedNet_efficientnet.mat'));
efficientNet = trainedNet;
fprintf('✅ EfficientNet-B0 yüklendi\n');

load(fullfile(basePath, 'trainedNet.mat'));  % ResNet-18 
resNet = trainedNet;
fprintf('✅ ResNet-18 yüklendi\n\n');

%% Holdout verisini yükle
fprintf('📂 Holdout verisi yükleniyor...\n');
load(fullfile(basePath, 'testRealWaste_holdout.mat'));
testDS = imageDatastore(testHoldoutFiles, 'Labels', testHoldoutLabels, 'ReadFcn', customReadFcn);
augTestDS = augmentedImageDatastore([224 224 3], testDS, 'ColorPreprocessing', 'gray2rgb');
fprintf('✅ Holdout: %d fotoğraf\n\n', numel(testDS.Files));

%% Tahminler
fprintf('🔮 Tahminler yapılıyor...\n');
YPred_eff = classify(efficientNet, augTestDS);
YPred_res = classify(resNet, augTestDS);
YTrue = testDS.Labels;

%% Metrikleri hesapla
acc_eff = mean(YPred_eff == YTrue) * 100;
acc_res = mean(YPred_res == YTrue) * 100;
f1_eff = computeMacroF1(YTrue, YPred_eff) * 100;
f1_res = computeMacroF1(YTrue, YPred_res) * 100;

fprintf('\n📊 SONUÇLAR:\n');
fprintf('   EfficientNet-B0: Accuracy=%.1f%%, F1-Score=%.1f%%\n', acc_eff, f1_eff);
fprintf('   ResNet-18:       Accuracy=%.1f%%, F1-Score=%.1f%%\n\n', acc_res, f1_res);

%% 1. MODEL PERFORMANS KARŞILAŞTIRMASI
fprintf('📊 Grafik 1: Model Performans Karşılaştırması...\n');
figure('Name', 'Model Performans', 'Position', [100, 100, 800, 400]);

metrics = categorical({'Accuracy', 'F1-Score (Macro)'});
metrics = reordercats(metrics, {'Accuracy', 'F1-Score (Macro)'});

resnet_scores = [acc_res, f1_res];
eff_scores = [acc_eff, f1_eff];

b = bar(metrics, [resnet_scores; eff_scores]', 'grouped');
b(1).FaceColor = [0.2, 0.6, 0.9];  % Mavi - ResNet
b(2).FaceColor = [0.2, 0.8, 0.4];  % Yeşil - EfficientNet

ylabel('Score (%)', 'FontSize', 12);
title('Model Performans Karşılaştırması', 'FontSize', 14, 'FontWeight', 'bold');
legend({'ResNet18', 'EfficientNet-B0'}, 'Location', 'northeast');
ylim([0 105]);
grid on;
set(gca, 'GridAlpha', 0.3);

% Değerleri yaz
xtips = b(1).XEndPoints;
ytips = b(1).YEndPoints;
text(xtips, ytips + 2, string(round(resnet_scores, 1)) + "%", 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
xtips = b(2).XEndPoints;
ytips = b(2).YEndPoints;
text(xtips, ytips + 2, string(round(eff_scores, 1)) + "%", 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

saveas(gcf, fullfile(basePath, 'matlab_model_performans.png'));
fprintf('✅ Kaydedildi: matlab_model_performans.png\n');

%% 2. PER-CLASS METRICS
fprintf('📊 Grafik 2: Per-Class Metrics...\n');
figure('Name', 'Per-Class Metrics', 'Position', [100, 100, 1400, 400]);

allClasses = categories(YTrue);

% battery ve shoes sınıflarını çıkar
excludeClasses = {'battery', 'shoes'};
classes = allClasses(~ismember(allClasses, excludeClasses));
numClasses = numel(classes);

% Metrikleri hesapla
[prec_res, rec_res, f1_res_c] = computePerClassMetrics(YTrue, YPred_res, classes);
[prec_eff, rec_eff, f1_eff_c] = computePerClassMetrics(YTrue, YPred_eff, classes);

x = 1:numClasses;
width = 0.35;

% Precision
subplot(1, 3, 1);
bar(x - width/2, prec_res, width, 'FaceColor', [0.2, 0.6, 0.9]);
hold on;
bar(x + width/2, prec_eff, width, 'FaceColor', [0.2, 0.8, 0.4]);
ylabel('Score');
title('Precision', 'FontWeight', 'bold');
set(gca, 'XTick', x, 'XTickLabel', classes, 'XTickLabelRotation', 45);
legend({'ResNet18', 'EfficientNet-B0'}, 'Location', 'southwest');
ylim([0 1.1]);
grid on;

% Recall
subplot(1, 3, 2);
bar(x - width/2, rec_res, width, 'FaceColor', [0.2, 0.6, 0.9]);
hold on;
bar(x + width/2, rec_eff, width, 'FaceColor', [0.2, 0.8, 0.4]);
ylabel('Score');
title('Recall', 'FontWeight', 'bold');
set(gca, 'XTick', x, 'XTickLabel', classes, 'XTickLabelRotation', 45);
legend({'ResNet18', 'EfficientNet-B0'}, 'Location', 'southwest');
ylim([0 1.1]);
grid on;

% F1-Score
subplot(1, 3, 3);
bar(x - width/2, f1_res_c, width, 'FaceColor', [0.2, 0.6, 0.9]);
hold on;
bar(x + width/2, f1_eff_c, width, 'FaceColor', [0.2, 0.8, 0.4]);
ylabel('Score');
title('F1-Score', 'FontWeight', 'bold');
set(gca, 'XTick', x, 'XTickLabel', classes, 'XTickLabelRotation', 45);
legend({'ResNet18', 'EfficientNet-B0'}, 'Location', 'southwest');
ylim([0 1.1]);
grid on;

sgtitle('Per-Class Metrics Karşılaştırması', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(basePath, 'matlab_per_class_metrics.png'));
fprintf('✅ Kaydedildi: matlab_per_class_metrics.png\n');

%% 3. CONFUSION MATRIX
fprintf('📊 Grafik 3: Confusion Matrix...\n');
figure('Name', 'Confusion Matrix', 'Position', [100, 100, 1200, 500]);

% battery ve shoes olmayan örnekleri filtrele
validIdx = ~ismember(cellstr(YTrue), excludeClasses);
YTrue_filtered = removecats(YTrue(validIdx), excludeClasses);
YPred_res_filtered = removecats(YPred_res(validIdx), excludeClasses);
YPred_eff_filtered = removecats(YPred_eff(validIdx), excludeClasses);

% ResNet-18
subplot(1, 2, 1);
cm_res = confusionmat(YTrue_filtered, YPred_res_filtered);
confusionchart(cm_res, classes, 'Title', sprintf('ResNet-18\nAccuracy: %.1f%%', acc_res), ...
    'RowSummary', 'off', 'ColumnSummary', 'off');

% EfficientNet-B0
subplot(1, 2, 2);
cm_eff = confusionmat(YTrue_filtered, YPred_eff_filtered);
confusionchart(cm_eff, classes, 'Title', sprintf('EfficientNet-B0\nAccuracy: %.1f%%', acc_eff), ...
    'RowSummary', 'off', 'ColumnSummary', 'off');

sgtitle('Model Karşılaştırması - Confusion Matrix', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(basePath, 'matlab_confusion_matrix.png'));
fprintf('✅ Kaydedildi: matlab_confusion_matrix.png\n');

%% 4. HER SINIFTAN ÖRNEK
fprintf('📊 Grafik 4: Sınıf Örnekleri...\n');
figure('Name', 'Sınıf Örnekleri', 'Position', [50, 100, 1400, 600]);

for i = 1:numClasses
    className = classes{i};
    classIndices = find(YTrue == className);
    randomIdx = classIndices(randi(numel(classIndices)));
    
    img = imread(testDS.Files{randomIdx});
    
    subplot(2, 4, i);
    imshow(img);
    title(className, 'FontSize', 12, 'FontWeight', 'bold');
end

sgtitle('Her Sınıftan Örnek Görüntüler', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(basePath, 'matlab_sinif_ornekleri.png'));
fprintf('✅ Kaydedildi: matlab_sinif_ornekleri.png\n');

%% 5. TAHMİN SONUÇLARI (EfficientNet)
fprintf('📊 Grafik 5: Tahmin Sonuçları...\n');
figure('Name', 'Tahmin Sonuçları', 'Position', [50, 100, 1200, 400]);

numSamples = 6;
perm = randperm(numel(testDS.Files), numSamples);

for i = 1:numSamples
    idx = perm(i);
    img = imread(testDS.Files{idx});
    
    trueLabel = char(YTrue(idx));
    predLabel = char(YPred_eff(idx));
    isCorrect = strcmp(trueLabel, predLabel);
    
    subplot(2, 3, i);
    imshow(img);
    
    if isCorrect
        title(sprintf('Gerçek: %s\nTahmin: %s', trueLabel, predLabel), ...
            'Color', [0.2, 0.7, 0.2], 'FontWeight', 'bold', 'FontSize', 10);
    else
        title(sprintf('Gerçek: %s\nTahmin: %s', trueLabel, predLabel), ...
            'Color', [0.9, 0.2, 0.2], 'FontWeight', 'bold', 'FontSize', 10);
    end
end

sgtitle('EfficientNet-B0 Tahmin Sonuçları (Yeşil=Doğru, Kırmızı=Yanlış)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(basePath, 'matlab_tahmin_sonuclari.png'));
fprintf('✅ Kaydedildi: matlab_tahmin_sonuclari.png\n');

%% 6. DATASET DAĞILIMI (4'lü grafik)
fprintf('📊 Grafik 6: Dataset Dağılımı...\n');
figure('Name', 'Dataset Dağılımı', 'Position', [50, 100, 1600, 400]);

classNames10 = {'battery', 'biological', 'cardboard', 'clothes', 'glass', ...
                'metal', 'paper', 'plastic', 'shoes', 'trash'};

% 1. KAGGLE
subplot(1, 4, 1);
classFolders = fullfile(basePath, classNames10);
kaggleDS = imageDatastore(classFolders, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
kaggleCounts = countEachLabel(kaggleDS);
bar(categorical(kaggleCounts.Label), kaggleCounts.Count, 'FaceColor', [0.2, 0.6, 0.9]);
ylabel('Fotoğraf Sayısı', 'FontSize', 10);
title('Kaggle', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabelRotation', 45);
grid on;

% 2. REAL_WORLD
subplot(1, 4, 2);
realWorldPath = fullfile(basePath, 'real_world');
realWorldDS = imageDatastore(realWorldPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
realWorldCounts = countEachLabel(realWorldDS);
bar(categorical(realWorldCounts.Label), realWorldCounts.Count, 'FaceColor', [0.9, 0.5, 0.2]);
ylabel('Fotoğraf Sayısı', 'FontSize', 10);
title('real\_world', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabelRotation', 45);
grid on;

% 3. TESTREALWASTE
subplot(1, 4, 3);
testRealWastePath = fullfile(basePath, 'TestRealWaste');
testRealWasteDS = imageDatastore(testRealWastePath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testRealWasteCounts = countEachLabel(testRealWasteDS);
bar(categorical(testRealWasteCounts.Label), testRealWasteCounts.Count, 'FaceColor', [0.95, 0.7, 0.2]);
ylabel('Fotoğraf Sayısı', 'FontSize', 10);
title('TestRealWaste', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabelRotation', 45);
grid on;

% 4. BİRLEŞİK
subplot(1, 4, 4);
combinedCounts = zeros(numel(classNames10), 1);
for i = 1:numel(classNames10)
    className = classNames10{i};
    count = 0;
    % Kaggle
    idx = find(string(kaggleCounts.Label) == className);
    if ~isempty(idx), count = count + kaggleCounts.Count(idx); end
    % real_world
    idx = find(string(realWorldCounts.Label) == className);
    if ~isempty(idx), count = count + realWorldCounts.Count(idx); end
    % TestRealWaste (sadece eğitime dahil olan %50)
    idx = find(string(testRealWasteCounts.Label) == className);
    if ~isempty(idx), count = count + floor(testRealWasteCounts.Count(idx) / 2); end
    combinedCounts(i) = count;
end
bar(categorical(classNames10), combinedCounts, 'FaceColor', [0.3, 0.7, 0.4]);
ylabel('Fotoğraf Sayısı', 'FontSize', 10);
title('Birleşik', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabelRotation', 45);
grid on;

sgtitle('Dataset Dağılım Analizi', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(basePath, 'matlab_dataset_dagilimi.png'));
fprintf('✅ Kaydedildi: matlab_dataset_dagilimi.png\n');

fprintf('\n✅ Tüm grafikler oluşturuldu!\n');

%% YARDIMCI FONKSİYONLAR
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

function macroF1 = computeMacroF1(yTrue, yPred)
    classes = categories(yTrue);
    f1Scores = zeros(numel(classes), 1);
    
    for i = 1:numel(classes)
        tp = sum(yPred == classes{i} & yTrue == classes{i});
        fp = sum(yPred == classes{i} & yTrue ~= classes{i});
        fn = sum(yPred ~= classes{i} & yTrue == classes{i});
        
        precision = tp / (tp + fp + eps);
        recall = tp / (tp + fn + eps);
        f1Scores(i) = 2 * (precision * recall) / (precision + recall + eps);
    end
    
    macroF1 = mean(f1Scores);
end

function [precision, recall, f1] = computePerClassMetrics(yTrue, yPred, classes)
    numClasses = numel(classes);
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);
    
    for i = 1:numClasses
        tp = sum(yPred == classes{i} & yTrue == classes{i});
        fp = sum(yPred == classes{i} & yTrue ~= classes{i});
        fn = sum(yPred ~= classes{i} & yTrue == classes{i});
        
        precision(i) = tp / (tp + fp + eps);
        recall(i) = tp / (tp + fn + eps);
        f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
    end
end
