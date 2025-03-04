% 指定输入和输出文件夹
inputFolder = 'D:/Dataset/VT5000/VT5000/Test/T/';  % 替换为你的输入文件夹路径
outputFolder = 'D:/Dataset/VT5000/VT5000/Test/96_T/';  % 替换为你的输出文件夹路径

% 获取输入文件夹中的所有图像文件
imageFiles = dir(fullfile(inputFolder, '*.jpg'));  % 假设图像格式为jpg，可以根据实际情况修改

% 遍历每个图像文件
for i = 1:length(imageFiles)
    % 构造图像的完整路径
    imagePath = fullfile(inputFolder, imageFiles(i).name);
    
    % 读取640x480的图像
    originalImage = imread(imagePath);
    
    % 将图像降采样到64x64的分辨率，使用bicubic插值方式
    resizedImage = imresize(originalImage, [96, 96], 'bicubic');
    
    % 构造输出图像的完整路径
    [~, fileName, fileExtension] = fileparts(imageFiles(i).name);
    outputImagePath = fullfile(outputFolder, [fileName  fileExtension]);
    
    % 保存降采样后的图像
    imwrite(resizedImage, outputImagePath);
    
    fprintf('Processed: %s\n', imageFiles(i).name);
end

disp('Batch resizing completed.');
