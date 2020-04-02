clc
close all
clear all

% folder = 'C:\Users\Lawrence\machine learning\Saraf_machine_learning\worm_images';  % You specify this!
% fullMatFileName = fullfile(folder,  'mapt.mat')
% if ~exist(fullMatFileName, 'file')
%   message = sprintf('%s does not exist', fullMatFileName);
%   uiwait(warndlg(message));
% else
%   t = load(fullMatFileName);
% end
worms_folder = 'C:\Users\Lawrence\machine learning\Saraf_machine_learning\worm_images_scott';
all_images = resize_multiple_imgs(28,28,worms_folder,0);
all_images = zero_pad_to_largest_image(worms_folder,0);

%option to save if save = 1;
function y = resize_multiple_imgs(new_rows, new_columns, folder_name,save) %resizes all images inside new folder named "resized_folder" to desired dimensions. returns resized images as vectors
mkdir resized_folder
filelist = dir(folder_name);
offset = 3; %subtract foulder name and location and .mat file
len = length(filelist)-offset ;
for i=offset : len
  filename = filelist(i);
  if ~strcmp(filename.name , '.') && ~strcmp(filename.name , '..')
      image = strcat(filename.folder,'\',filename.name);
      filename.name
      image = imread(image);
      image = imresize(image,[new_rows new_columns]);
      if save == 1
          imwrite(image,strcat('.\resized_folder\', num2str(i-offset +1) ,'.tiff'));
      end
      y(:, :,i-2) = image;
  end
end
end
function y = zero_pad_to_largest_image(folder_name,save)
mkdir zero_pad_folder
filelist = dir(folder_name);
offset = 3; %subtract foulder name and location and .mat file
len = length(filelist)-offset ;
max_row = 1;
max_col = 1;
for i=offset : len
  filename = filelist(i);
  if ~strcmp(filename.name , '.') && ~strcmp(filename.name , '..')
      image = strcat(filename.folder,'\',filename.name);
      %filename.name
      image = imread(image);
      [new_row, new_col] = size(image);
      if new_row > max_row
          max_row = new_row;
      end
      if new_col > max_col
          max_col = new_col;
      end
  end
end
for i=offset : len
    filename = filelist(i);
    if ~strcmp(filename.name , '.') && ~strcmp(filename.name , '..')
        image = strcat(filename.folder,'\',filename.name);
        filename.name
        image = imread(image);
        [current_row, current_col] = size(image);
        padded_image = [image,zeros(current_row,max_col-current_col)]; %horizontal concatenation
        padded_image = [padded_image;zeros(max_row - current_row, max_col)]; %virtical c
        if save == 1
          imwrite(padded_image,strcat('.\zero_pad_folder\', num2str(i-offset +1) ,'.tiff'));
        end
        y(:, :,i-2) = padded_image;
    end
end
end

