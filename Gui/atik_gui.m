function atik_gui()
%% =====================================================
%  ATIK SINIFLANDIRMA GUI - Garbage Classification UI
%  Modern, Professional MATLAB App with Camera Support
% =====================================================

%% Check if model exists
modelPath = fullfile(fileparts(mfilename('fullpath')), 'trainedNet_efficientnet.mat');
if ~isfile(modelPath)
    errordlg('trainedNet_efficientnet.mat bulunamadı!', 'Model Bulunamadı');
    return;
end

% Load the trained network
data = load(modelPath);
trainedNet = data.trainedNet;

% Camera object (will be initialized when needed)
cam = [];

% Current image for crop operations
currentImage = [];

%% Turkish labels for classes
classLabels = {
    'battery',    'Pil',           [0.90, 0.20, 0.20];  % Red - Hazardous
    'biological', 'Organik Atık',  [0.55, 0.35, 0.20];  % Brown
    'cardboard',  'Karton',        [0.85, 0.65, 0.40];  % Tan
    'clothes',    'Kıyafet',       [0.70, 0.40, 0.70];  % Purple
    'glass',      'Cam',           [0.20, 0.70, 0.30];  % Green
    'metal',      'Metal',         [0.50, 0.50, 0.55];  % Gray
    'paper',      'Kağıt',         [0.95, 0.90, 0.80];  % Cream
    'plastic',    'Plastik',       [0.20, 0.60, 0.85];  % Blue
    'shoes',      'Ayakkabı',      [0.40, 0.25, 0.15];  % Dark Brown
    'trash',      'Genel Çöp',     [0.30, 0.30, 0.30];  % Dark Gray
};

%% Create Main Figure
fig = uifigure('Name', 'Atık Sınıflandırma Sistemi', ...
    'Position', [100, 100, 900, 650], ...
    'Color', [0.15, 0.15, 0.18], ...
    'Resize', 'off', ...
    'CloseRequestFcn', @closeApp);

%% Title Panel
titleLabel = uilabel(fig, ...
    'Text', '♻️ ATIK SINIFLANDIRMA SİSTEMİ', ...
    'Position', [0, 590, 900, 50], ...
    'FontSize', 24, ...
    'FontWeight', 'bold', ...
    'FontColor', [0.3, 0.85, 0.5], ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.12, 0.12, 0.15]);

%% Left Panel - Image Display
imagePanel = uipanel(fig, ...
    'Title', 'Görüntü', ...
    'Position', [20, 150, 420, 420], ...
    'BackgroundColor', [0.2, 0.2, 0.23], ...
    'ForegroundColor', [0.8, 0.8, 0.8], ...
    'FontSize', 14, ...
    'FontWeight', 'bold');

imageAxes = uiaxes(imagePanel, ...
    'Position', [10, 10, 395, 370], ...
    'Color', [0.25, 0.25, 0.28]);
imageAxes.XTick = [];
imageAxes.YTick = [];
imageAxes.Box = 'on';
imageAxes.XColor = [0.4, 0.4, 0.4];
imageAxes.YColor = [0.4, 0.4, 0.4];

% Placeholder text
text(imageAxes, 0.5, 0.5, 'Görüntü Yükleyin veya Kamera Kullanın', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 14, ...
    'Color', [0.5, 0.5, 0.5], ...
    'Units', 'normalized');

%% Right Panel - Results
resultPanel = uipanel(fig, ...
    'Title', 'Sonuç', ...
    'Position', [460, 150, 420, 420], ...
    'BackgroundColor', [0.2, 0.2, 0.23], ...
    'ForegroundColor', [0.8, 0.8, 0.8], ...
    'FontSize', 14, ...
    'FontWeight', 'bold');

% Predicted Class Label
predictionLabel = uilabel(resultPanel, ...
    'Text', 'Sınıf Bekleniyor...', ...
    'Position', [10, 320, 395, 50], ...
    'FontSize', 28, ...
    'FontWeight', 'bold', ...
    'FontColor', [0.7, 0.7, 0.7], ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.25, 0.25, 0.28]);

% Confidence Label
confidenceLabel = uilabel(resultPanel, ...
    'Text', 'Güven: --%', ...
    'Position', [10, 280, 395, 30], ...
    'FontSize', 16, ...
    'FontColor', [0.6, 0.6, 0.6], ...
    'HorizontalAlignment', 'center');

% Confidence Bars Axes
barAxes = uiaxes(resultPanel, ...
    'Position', [10, 10, 395, 260]);
barAxes.Color = [0.25, 0.25, 0.28];
barAxes.XColor = [0.6, 0.6, 0.6];
barAxes.YColor = [0.6, 0.6, 0.6];
barAxes.FontSize = 10;
title(barAxes, 'Sınıf Olasılıkları', 'Color', [0.8, 0.8, 0.8], 'FontSize', 12);

%% Bottom Buttons - Row 1 (main actions)
uploadBtn = uibutton(fig, 'push', ...
    'Text', '📁 Dosyadan Yükle', ...
    'Position', [30, 55, 140, 45], ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.2, 0.6, 0.3], ...
    'FontColor', 'white');

cameraBtn = uibutton(fig, 'push', ...
    'Text', '📸 Kamera', ...
    'Position', [180, 55, 120, 45], ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.2, 0.5, 0.7], ...
    'FontColor', 'white');

captureBtn = uibutton(fig, 'push', ...
    'Text', '📷 Çek', ...
    'Position', [310, 55, 100, 45], ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.6, 0.4, 0.7], ...
    'FontColor', 'white', ...
    'Enable', 'off');

% Crop buttons
cropBtn = uibutton(fig, 'push', ...
    'Text', '✂️ Manuel Kırp', ...
    'Position', [420, 55, 130, 45], ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.7, 0.5, 0.2], ...
    'FontColor', 'white', ...
    'Enable', 'off');

autoCropBtn = uibutton(fig, 'push', ...
    'Text', '🎯 Merkez Kırp', ...
    'Position', [560, 55, 130, 45], ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.5, 0.6, 0.3], ...
    'FontColor', 'white', ...
    'Enable', 'off');

clearBtn = uibutton(fig, 'push', ...
    'Text', '🗑️ Temizle', ...
    'Position', [700, 55, 110, 45], ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.6, 0.25, 0.25], ...
    'FontColor', 'white');

% Auto center crop checkbox
autoCropCheck = uicheckbox(fig, ...
    'Text', 'Otomatik Merkez Kırpma', ...
    'Position', [30, 15, 200, 25], ...
    'FontSize', 11, ...
    'FontColor', [0.7, 0.7, 0.7], ...
    'Value', false);

% Crop percentage dropdown
cropLabel = uilabel(fig, ...
    'Text', 'Kırpma Oranı:', ...
    'Position', [240, 15, 100, 25], ...
    'FontSize', 11, ...
    'FontColor', [0.6, 0.6, 0.6]);

cropRatioDropdown = uidropdown(fig, ...
    'Items', {'%50 (Agresif)', '%60', '%70 (Varsayılan)', '%80', '%90 (Hafif)'}, ...
    'ItemsData', [0.50, 0.60, 0.70, 0.80, 0.90], ...
    'Value', 0.70, ...
    'Position', [340, 15, 140, 25], ...
    'FontSize', 11);

% Status label
statusLabel = uilabel(fig, ...
    'Text', '📋 Hazır', ...
    'Position', [30, 110, 840, 25], ...
    'FontSize', 11, ...
    'FontColor', [0.5, 0.7, 0.5], ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.18, 0.18, 0.21]);

%% Button Callbacks
uploadBtn.ButtonPushedFcn = @(~,~) loadFromFile();
cameraBtn.ButtonPushedFcn = @(~,~) toggleCamera();
captureBtn.ButtonPushedFcn = @(~,~) captureAndClassify();
cropBtn.ButtonPushedFcn = @(~,~) manualCrop();
autoCropBtn.ButtonPushedFcn = @(~,~) autoCenterCrop();
clearBtn.ButtonPushedFcn = @(~,~) clearAll();

%% Nested Functions

    function loadFromFile()
        % Stop camera if running
        stopCamera();
        
        % Open file dialog
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.gif', 'Görüntü Dosyaları'}, ...
            'Bir görüntü seçin');
        
        if isequal(file, 0)
            return;
        end
        
        imgPath = fullfile(path, file);
        
        try
            img = imread(imgPath);
            currentImage = img;  % Save for crop operations
            
            % Enable crop buttons
            cropBtn.Enable = 'on';
            autoCropBtn.Enable = 'on';
            
            % Apply auto center crop if checked
            if autoCropCheck.Value
                img = applyCenterCrop(img, cropRatioDropdown.Value);
                statusLabel.Text = ['📋 Dosya yüklendi + Merkez kırpma uygulandı: ' file];
            else
                statusLabel.Text = ['📋 Dosya yüklendi: ' file];
            end
            
            classifyImage(img);
        catch ME
            errordlg(['Görüntü işlenirken hata: ' ME.message], 'Hata');
        end
    end

    function toggleCamera()
        if isempty(cam) || ~isvalid(cam)
            % Start camera
            try
                statusLabel.Text = '📋 Kamera başlatılıyor...';
                drawnow;
                
                cam = webcam;
                cameraBtn.Text = '⏹️ Kamerayı Kapat';
                cameraBtn.BackgroundColor = [0.7, 0.3, 0.3];
                captureBtn.Enable = 'on';
                statusLabel.Text = '📋 Kamera aktif - Fotoğraf çekmek için butona basın';
                
                % Start preview loop
                startPreview();
                
            catch ME
                errordlg(['Kamera açılamadı: ' ME.message], 'Kamera Hatası');
                statusLabel.Text = '❌ Kamera bulunamadı!';
            end
        else
            stopCamera();
        end
    end

    function startPreview()
        while ~isempty(cam) && isvalid(cam) && isvalid(fig)
            try
                frame = snapshot(cam);
                frame = flip(frame, 2);  % Horizontal flip - ayna düzeltmesi
                cla(imageAxes);
                imshow(frame, 'Parent', imageAxes);
                drawnow limitrate;
            catch
                break;
            end
        end
    end

    function stopCamera()
        if ~isempty(cam) && isvalid(cam)
            clear cam;
        end
        cam = [];
        cameraBtn.Text = '📸 Kamera';
        cameraBtn.BackgroundColor = [0.2, 0.5, 0.7];
        captureBtn.Enable = 'off';
        statusLabel.Text = '📋 Hazır';
    end

    function captureAndClassify()
        if ~isempty(cam) && isvalid(cam)
            try
                % Capture frame
                img = snapshot(cam);
                img = flip(img, 2);  % Horizontal flip - ayna düzeltmesi
                
                % Stop camera preview
                stopCamera();
                
                % Save for crop operations
                currentImage = img;
                
                % Enable crop buttons
                cropBtn.Enable = 'on';
                autoCropBtn.Enable = 'on';
                
                % Apply auto center crop if checked
                if autoCropCheck.Value
                    img = applyCenterCrop(img, cropRatioDropdown.Value);
                    statusLabel.Text = '📋 Fotoğraf çekildi + Merkez kırpma uygulandı!';
                else
                    statusLabel.Text = '📋 Fotoğraf çekildi ve sınıflandırıldı!';
                end
                
                % Show captured image
                cla(imageAxes);
                imshow(img, 'Parent', imageAxes);
                
                % Classify
                classifyImage(img);
                
            catch ME
                errordlg(['Fotoğraf çekilemedi: ' ME.message], 'Hata');
            end
        end
    end

    function classifyImage(img)
        try
            % Handle grayscale
            if size(img, 3) == 1
                img = cat(3, img, img, img);
            end
            
            % Handle RGBA
            if size(img, 3) == 4
                img = img(:,:,1:3);
            end
            
            % Display image
            cla(imageAxes);
            imshow(img, 'Parent', imageAxes);
            
            % Preprocess for network (224x224)
            imgResized = imresize(img, [224, 224]);
            
            % Classify
            [predictedClass, scores] = classify(trainedNet, imgResized);
            
            % Get class info
            classIdx = find(strcmp(classLabels(:,1), char(predictedClass)));
            if ~isempty(classIdx)
                turkishName = classLabels{classIdx, 2};
                classColor = classLabels{classIdx, 3};
            else
                turkishName = char(predictedClass);
                classColor = [0.5, 0.5, 0.5];
            end
            
            % Update prediction label
            maxScore = max(scores) * 100;
            predictionLabel.Text = upper(turkishName);
            predictionLabel.FontColor = classColor;
            predictionLabel.BackgroundColor = classColor * 0.3;
            
            % Update confidence label
            confidenceLabel.Text = sprintf('Güven: %.1f%%', maxScore);
            if maxScore >= 70
                confidenceLabel.FontColor = [0.3, 0.85, 0.4];
            elseif maxScore >= 40
                confidenceLabel.FontColor = [0.9, 0.7, 0.2];
            else
                confidenceLabel.FontColor = [0.9, 0.3, 0.3];
            end
            
            % Update bar chart
            updateBarChart(scores, predictedClass);
            
        catch ME
            errordlg(['Sınıflandırma hatası: ' ME.message], 'Hata');
        end
    end

    function updateBarChart(scores, predictedClass)
        cla(barAxes);
        classes = categories(predictedClass);
        numClasses = length(classes);
        barColors = zeros(numClasses, 3);
        turkishNames = cell(numClasses, 1);
        
        for i = 1:numClasses
            idx = find(strcmp(classLabels(:,1), classes{i}));
            if ~isempty(idx)
                turkishNames{i} = classLabels{idx, 2};
                barColors(i,:) = classLabels{idx, 3};
            else
                turkishNames{i} = classes{i};
                barColors(i,:) = [0.5, 0.5, 0.5];
            end
        end
        
        % Create horizontal bar chart
        scoreValues = double(scores) * 100;
        barh(barAxes, 1:numClasses, scoreValues, 'FaceColor', 'flat');
        barAxes.Children.CData = barColors;
        barAxes.YTick = 1:numClasses;
        barAxes.YTickLabel = turkishNames;
        barAxes.XLim = [0, 100];
        xlabel(barAxes, 'Olasılık (%)', 'Color', [0.7, 0.7, 0.7]);
        barAxes.Color = [0.25, 0.25, 0.28];
        barAxes.XColor = [0.6, 0.6, 0.6];
        barAxes.YColor = [0.8, 0.8, 0.8];
        grid(barAxes, 'on');
        barAxes.GridColor = [0.4, 0.4, 0.4];
        barAxes.GridAlpha = 0.3;
    end

    function clearAll()
        stopCamera();
        
        % Clear current image
        currentImage = [];
        
        % Disable crop buttons
        cropBtn.Enable = 'off';
        autoCropBtn.Enable = 'off';
        
        % Clear image
        cla(imageAxes);
        text(imageAxes, 0.5, 0.5, 'Görüntü Yükleyin veya Kamera Kullanın', ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 14, ...
            'Color', [0.5, 0.5, 0.5], ...
            'Units', 'normalized');
        
        % Reset labels
        predictionLabel.Text = 'Sınıf Bekleniyor...';
        predictionLabel.FontColor = [0.7, 0.7, 0.7];
        predictionLabel.BackgroundColor = [0.25, 0.25, 0.28];
        confidenceLabel.Text = 'Güven: --%';
        confidenceLabel.FontColor = [0.6, 0.6, 0.6];
        
        % Clear bar chart
        cla(barAxes);
        title(barAxes, 'Sınıf Olasılıkları', 'Color', [0.8, 0.8, 0.8], 'FontSize', 12);
        
        statusLabel.Text = '📋 Temizlendi - Hazır';
    end

    function manualCrop()
        % Manuel kırpma - kullanıcı seçsin
        if isempty(currentImage)
            statusLabel.Text = '⚠️ Önce bir görüntü yükleyin!';
            return;
        end
        
        try
            statusLabel.Text = '📋 Fare ile kırpma alanını seçin...';
            drawnow;
            
            % Create figure for cropping
            cropFig = figure('Name', 'Görüntü Kırpma', 'Position', [200, 200, 800, 600]);
            imshow(currentImage);
            title('Fare ile kırpma alanını seçin, sonra çift tıklayın', 'FontSize', 14);
            
            % Let user select crop rectangle
            rect = imrect;
            position = wait(rect);
            
            if ~isempty(position)
                % Crop the image
                croppedImg = imcrop(currentImage, position);
                close(cropFig);
                
                % Update current image and classify
                currentImage = croppedImg;
                classifyImage(croppedImg);
                statusLabel.Text = '✂️ Manuel kırpma uygulandı!';
            else
                close(cropFig);
                statusLabel.Text = '⚠️ Kırpma iptal edildi.';
            end
        catch ME
            statusLabel.Text = ['❌ Kırpma hatası: ' ME.message];
        end
    end

    function autoCenterCrop()
        % Otomatik merkez kırpma
        if isempty(currentImage)
            statusLabel.Text = '⚠️ Önce bir görüntü yükleyin!';
            return;
        end
        
        try
            ratio = cropRatioDropdown.Value;
            croppedImg = applyCenterCrop(currentImage, ratio);
            
            % Update current image and classify
            currentImage = croppedImg;
            classifyImage(croppedImg);
            statusLabel.Text = sprintf('🎯 Merkez kırpma uygulandı (%%%d)', round(ratio*100));
        catch ME
            statusLabel.Text = ['❌ Kırpma hatası: ' ME.message];
        end
    end

    function croppedImg = applyCenterCrop(img, ratio)
        % Merkez kırpma fonksiyonu
        [h, w, ~] = size(img);
        
        % Yeni boyutlar
        newH = round(h * ratio);
        newW = round(w * ratio);
        
        % Merkez koordinatları
        startY = round((h - newH) / 2) + 1;
        startX = round((w - newW) / 2) + 1;
        
        % Kırp
        croppedImg = img(startY:startY+newH-1, startX:startX+newW-1, :);
    end

    function closeApp(~, ~)
        stopCamera();
        delete(fig);
    end

end
