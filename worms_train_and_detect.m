%ML Project 4
%Part 1
clc
clear all
close all


steps = 100;
p = 5;
Winital = .00000001; %
new_rows = 28;
new_columns = 28;

tot_num_img = 15665;
num_train_imgs = 15655;
numb_test_imgs = tot_num_img - num_train_imgs;

%enter training data
%[trainImg, trainLabel] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');
training_folder = './worm_images_scott';
trainImg = resize_multiple_imgs(new_rows, new_columns, training_folder,0);
all_Labels = load('target.mat');
trainLabel = all_Labels.t(1:num_train_imgs);


K = max(trainLabel)+1;
[sz,sz2,N] = size(trainImg);
W = Winital*ones(1+sz^2,K); %intalize
%reshaping into vectors
%premute switches the rows and columns pretty much transpose

phi = cast([ones(1,N); reshape(permute(trainImg,[2 1 3]),[sz^2 N])],'double')./25600;

y = zeros(N,K);
t = zeros(N,K);
for n =1:N
    %zeroindexed compensate by adding one
    t(n,trainLabel(n)+1) = 1;
end
%clear trainLabel trainImg
for step = 1:steps
    fprintf('Step: %d\n',step)
    dE = zeros(1+sz^2,K);
    for n = 1:N
        y(n,:) = exp(sum(W.*phi(:,n)))./sum(exp(sum(W.*phi(:,n))))'; %guess calc
        for j = 1:K
        dE(:,j) = dE(:,j)+(y(n,j)-t(n,j)).*phi(:,n);  %error calc add up error of each class     1           
        end
    end
    
    %fprintf('ave error: %f\n',mean(mean(dE)));
    
    %Note: p changes with the amount of steps done for tighter fitting as
    %   the soultion progresses
    W = W - (p/(1+K*p*step/steps))*dE;    
    if(any(isnan(W)))           %NaN occurs for large w*phi
        error('nan error')
    end
end

%testing data
%[testImg, testLabel] = mnist_parse('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
testImg = resize_multiple_imgs(new_rows, new_columns, training_folder,0);
testLabel =  all_Labels.t(num_train_imgs:tot_num_img);
N = numb_test_imgs;

phi = cast([ones(1,N); reshape(permute(testImg,[2 1 3]),[sz^2 N])],'double')./25600;

result = -1*ones(numel(testLabel),1);
for n = 1:N
        y(n,:) = exp(sum(W.*phi(:,n)))./sum(exp(sum(W.*phi(:,n))))';
        [prob, result(n)] = max(y(n,:));
end
result = result - 1;
check = (result==testLabel);
percent = 100*sum(check)/numel(testLabel);

fprintf('p: %f, Wi: %f, Steps: %d, Percent: %f%%\n',p,Winital,steps,percent)

%function for reading mnist images by: Lawrence
function [images, labels] = mnist_parse(path_to_digits, path_to_labels)
% Open files
fid1 = fopen(path_to_digits, 'r');
% The labels file
fid2 = fopen(path_to_labels, 'r');
% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
%fprintf('Magic Number - Images: %d\n', magicNumber1);
A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
%fprintf('Magic Number - Labels: %d\n', magicNumber2);
% Read in total number of images
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');
totalImages = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end
%fprintf('Total number of images: %d\n', totalImages);
% Read in number of rows
A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));
% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));
%fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);
% For each image, store into an individual slice
images = zeros(numRows, numCols, totalImages, 'uint8');
for k = 1 : totalImages
    % Read in numRows*numCols pixels at a time
    A = fread(fid1, numRows*numCols, 'uint8');
    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    images(:,:,k) = reshape(uint8(A), numCols, numRows).';
end
% Read in the labels
labels = fread(fid2, totalImages, 'uint8');
% Close the files
fclose(fid1);
fclose(fid2);
end


function y = resize_multiple_imgs(new_rows, new_columns, folder_name,save) %resizes all images inside new folder named "resized_folder" to desired dimensions. returns resized images as vectors
mkdir resized_folder
filelist = dir(folder_name);
offset = 2; %subtract foulder name and location and .mat file
len = length(filelist);
for i=offset : len
  filename = filelist(i);
  if ~strcmp(filename.name , '.') && ~strcmp(filename.name , '..')
      image = strcat(filename.folder,'\',filename.name);
      filename.name;
      image = imread(image);
      image = imresize(image,[new_rows new_columns]);
      if save == 1
          imwrite(image,strcat('.\resized_folder\', num2str(i-offset +1) ,'.tiff'));
      end
      y(:, :,i-2) = image;
  end
end
end