clear all; 
% close all;
clc

    load('s.mat');
    
    b = [s.BoundingBox];
    index = 1;
    for i=4:4:size(b, 2)
        h(index) = b(1,i);
        index = index + 1;
    end
    g = h > 35 & h < 55;
    s = s(g);
    s



%     h = h.'
%     h = h.'






% img = im2gray(imread ("../numbers/image_3.jpg"));
% bw2 = imbinarize(img);
% 
% 
% for i=0:9
%     name = strcat('../numbers/', int2str(i), '.jpg');
%     gt = im2uint8(imbinarize(im2gray(imread(name))));
%     img = im2uint8(imresize(bw2, size(gt)));
% 
%     disp(name);
%     results(i+1) = immse(gt, img);
%     results(i+1) = imsim(gt, img);
% 
% end
% 
% [~, idx] = min(results);
% disp(idx-1);