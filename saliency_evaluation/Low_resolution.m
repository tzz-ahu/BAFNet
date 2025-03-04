% ָ�����������ļ���
inputFolder = 'D:/Dataset/VT5000/VT5000/Test/T/';  % �滻Ϊ��������ļ���·��
outputFolder = 'D:/Dataset/VT5000/VT5000/Test/96_T/';  % �滻Ϊ�������ļ���·��

% ��ȡ�����ļ����е�����ͼ���ļ�
imageFiles = dir(fullfile(inputFolder, '*.jpg'));  % ����ͼ���ʽΪjpg�����Ը���ʵ������޸�

% ����ÿ��ͼ���ļ�
for i = 1:length(imageFiles)
    % ����ͼ�������·��
    imagePath = fullfile(inputFolder, imageFiles(i).name);
    
    % ��ȡ640x480��ͼ��
    originalImage = imread(imagePath);
    
    % ��ͼ�񽵲�����64x64�ķֱ��ʣ�ʹ��bicubic��ֵ��ʽ
    resizedImage = imresize(originalImage, [96, 96], 'bicubic');
    
    % �������ͼ�������·��
    [~, fileName, fileExtension] = fileparts(imageFiles(i).name);
    outputImagePath = fullfile(outputFolder, [fileName  fileExtension]);
    
    % ���潵�������ͼ��
    imwrite(resizedImage, outputImagePath);
    
    fprintf('Processed: %s\n', imageFiles(i).name);
end

disp('Batch resizing completed.');
