%ML Project 4
%Part 1
clc
clear all
close all


steps = 100;
p = 5;
Winital = .00000001; %

%enter training data
[trainImg, trainLabel] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');
N = numel(trainLabel);
K = 10;
sz = 28;
W = Winital*ones(1+sz^2,K); %intalize
%reshaping into vectors
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
        dE(:,j) = dE(:,j)+(y(n,j)-t(n,j)).*phi(:,n);  %error calc             
        end
    end
    
    %fprintf('ave error: %f\n',mean(mean(dE)));
    
    %Note: p changes with the amount of steps done for tighter fitting as
    %   the soultion progresses
    W = W - (p/(1+10*p*step/steps))*dE;    
    if(any(isnan(W)))           %NaN occurs for large w*phi
        error('nan error')
    end
end

%testing data
[testImg, testLabel] = mnist_parse('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
N = numel(testLabel);

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