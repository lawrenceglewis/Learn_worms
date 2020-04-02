clc
close all
clear all
%read in mnist data
[train_imgs ,train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

%logistic regression for a multiclass problem K>2
M = 3;% M is the complexity of the logarithic regression
k = 10;% k is the number of classes
W = sym('W%d%d', [(M+1) k]);


k = .5;
p = 0:.01:.99;
y = bernoli(p,k);

softmatrix = [-3 -4 -7 0 6 9];
output = softmaxrow(softmatrix);


T = target_vector_frm_mnist(train_labels); % Each row has a list of the probabilities for each diget colomn 1 is the pobability of 0 column 2 is probability of 1
firugre
imshow(train_imags(:,:,1)

%design_rbf_matrix(X,Xi,N)

function y = sigmoid(x)
y = 1./(1+exp(-x));
end
function y = bernoli(p,k)
y = (p.^k).*(1-p).^(1-k);
end
function y = target_vector_frm_mnist(input_vector)
%number of digets to classify is 10
[row,col] = size(input_vector);
y = zeros(row,10);
for i = 1:row
    y(i,input_vector(i)+1) = 1;
end

end

function y = softmaxrow(input_matrix)%will softmax each and every row inside a matrix
[row, col]= size(input_matrix);
y = zeros(row,col);
for i = 1:row
    bottom_sum = 0;
    for j = 1:col
        bottom_sum = bottom_sum + exp(input_matrix(i,j));
    end
    for j = 1:col
        y(i,j) = exp(input_matrix(i,j))./bottom_sum;
    end
end
end
function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen(path_to_labels, 'r');

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
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
%% functions

%Gaussian dist.
function phi_out = phi(x,mu)
    s = 0.1;
    phi_out = exp(-(x-mu).^2 / (2.*(s.^2)) );
end

%create matrix of Gaussian rbf's.
%N = num datapoints in Xi = num columns in output matrix
%length(X) = num rows in output matrix
function [design_matrix_out] = design_rbf_matrix(input_data, M)

    design_matrix_out = ones(length(X),1);
    for i=1:N
        design_matrix_out(:,i+1) = phi(X,Xi(i));
    end

end

