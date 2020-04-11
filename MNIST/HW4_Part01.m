%ML Project 4
%Part 1
close all; clc; clear;

epochs = 100;    %max number of iterations
p = 5;          %learning rate
Winital = .00000001; %initialize Weights

%% input mnist training data
[trainImg, trainLabel] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');
N_train = numel(trainLabel);  %num samples
K = 10;                 %num classes (0-9)
sz = 28.*28;            %image size (28x28 pixels)

%intalize weight matrix (sz + 1 because offset)
W = Winital*ones(1+sz,K);

%reshape each image into a vector (unroll)
phi = cast([ones(1,N_train); reshape(permute(trainImg,[2 1 3]),[sz N_train])],'double')./25600;

%% one-hot-encode training labels
t = zeros(N_train,K);
for n = 1:N_train
    %zeroindexed, so compensate by adding one
    t(n,trainLabel(n)+1) = 1;
end

%% train
y = zeros(N_train,K); %prediction matrix
for step = 1:epochs
    fprintf('Step: %d\n',step)
    
    %calculate gradient
    dE = zeros(1+sz,K);
    for n = 1:N_train
        y(n,:) = sigmoid(W.*phi(:,n));  %make prediction
        dE = dE + (y(n,:)-t(n,:)).*phi(:,n);
    end
    
    %Update weights
    %   Note: learning rate (p) changes with the amount of epochs done for tighter fitting as the soultion progresses
    W = W - (p/(1+10*p*step/epochs))*dE;    
    
    if(any(isnan(W)))           %NaN occurs for large w*phi
        error('nan error')
    end
end

%% calculate training accuracy and error

y = zeros(N_train,K); %prediction matrix
num_identified = 0;
for n = 1:N_train
    y(n,:) = sigmoid(W.*phi(:,n));
    [prob, result(n)] = max(y(n,:));
    
    if (result(n) - 1) == trainLabel(n)
        num_identified = num_identified + 1;
    end
    
end

training_accuracy = 100.*num_identified./N_train;

%calculate training error using least squares error function (?)
err = abs(y - t);
training_error = sum(err(:))./(N_train.*10);

fprintf('\ntraining accuracy: %f%%\ntraining error: %f\n\n', training_accuracy, training_error);

clear phi y t err
%% input testing data
[testImg, testLabel] = mnist_parse('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
N_test = numel(testLabel);   %num images

phi = cast([ones(1,N_test); reshape(permute(testImg,[2 1 3]),[sz N_test])],'double')./25600;

%calculate predictions
result = -1*ones(N_test,1);
y = zeros(N_test,K);
num_identified = 0;
for n = 1:N_test
    y(n,:) = exp(sum(W.*phi(:,n)))./sum(exp(sum(W.*phi(:,n))))';
    [prob, result(n)] = max(y(n,:));
    
    if (result(n) - 1) == testLabel(n)
        num_identified = num_identified + 1;
    end
end

%training accuracy
training_accuracy = 100.*num_identified./N_test;

%calculate test error
%(need to one-hot-encode test labels)
t = zeros(N_test,K);
for n = 1:N_test
    %zeroindexed, so compensate by adding one
    t(n,testLabel(n)+1) = 1;
end
err = abs(y - t);
test_error = sum(err(:))./(N_test.*10);

fprintf('test accuracy: %f%%\ntest error: %f\n',training_accuracy,test_error);

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

function pred = sigmoid(x)
    pred = exp(sum(x))./sum(exp(sum(x)))';
end