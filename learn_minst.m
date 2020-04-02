clc
clear
close all

% readMNIST by Siddharth Hegde
%
% Description:
% Read digits and labels from raw MNIST data files
% File format as specified on http://yann.lecun.com/exdb/mnist/
% Note: The 4 pixel padding around the digits will be remove
%       Pixel values will be normalised to the [0...1] range
%
% Usage:
% [imgs labels] = readMNIST(imgFile, labelFile, readDigits, offset)
%
% Parameters:
% imgFile = name of the image file
% labelFile = name of the label file
% readDigits = number of digits to be read
% offset = skips the first offset number of digits before reading starts
%
% Returns:
% imgs = 20 x 20 x readDigits sized matrix of digits
% labels = readDigits x 1 matrix containing labels for each digit
%



[imgs labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen('train-labels.idx1-ubyte', 'r')
B1 = fread(fid1, 1, 'uint8'); % majic number should be zero
B2 = fread(fid1, 1, 'uint8'); % majic number should be zero
B3 = fread(fid1, 1, 'uint8'); % type of data 8 implies unsigned byte 
B4 = fread(fid1, 1, 'uint8'); % 1 for vector, 2 for matrix 3 for more
fprintf("%x %x %x %d\n",B1, B2, B3, B4);

D1 = fread(fid1, 1, 'uint32');
D2 = fread(fid1, 1, 'uint32');
D3 = fread(fid1, 1, 'uint32');
D1 = swapbytes(uint32(D1));
D2 = swapbytes(uint32(D2));
D3 = swapbytes(uint32(D3));
fprintf("%d %d %d\n",D1,D2,D3);

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
B = dec2hex(A);
fprintf('Magic Number decimal? - Images: %d\n', A);
fprintf('hex magic number  %d\n', B)
print('hex magic number ',int2str(B))
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
bmagic = dec2hex(magicNumber1);
fprintf('hex magicNumber %d\n', bmagic);
whos A
whos magicNumber1
fprintf('Magic Number - Images: %d\n', magicNumber1);

A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of images
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');

totalImages = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end
fprintf('Total number of images: %d\n', totalImages);

% Read in number of rows


A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

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