clear all; 
close all;
clc

%% image alignment

imOriginalColor  = imread('../dataset/base_loteria.jpg');
im2Color = imread('../dataset/boleto-loteria.jpeg');

imOriginal = rgb2gray(imOriginalColor);
im2 = rgb2gray(im2Color);

points  = detectSURFFeatures(imOriginal);
points2 = detectSURFFeatures(im2);
[features,  validPoints]  = extractFeatures(imOriginal,  points);
[features2, validPoints2] = extractFeatures(im2, points2);
indexPairs = matchFeatures(features, features2);
matched  = validPoints(indexPairs(:,1));
matched2 = validPoints2(indexPairs(:,2));
%Show machings points
figure;
showMatchedFeatures(imOriginal,im2,matched,matched2);
title('Matched points');

[tform, inlierIdx] = estimateGeometricTransform2D(matched2, matched, 'similarity');
inliermoving = matched2(inlierIdx, :);
inlierfixed  = matched(inlierIdx, :);
output = imref2d(size(imOriginal));


recovered  = imwarp(im2,tform,'OutputView',output);
figure, imshow(recovered);

%% segona part de homografia

[optimizer,metric] = imregconfig('multimodal');
movingRegisteredDefault = imregister(recovered,imOriginal,'affine',optimizer,metric);
figure, imshow(movingRegisteredDefault)

%% Numeros

level = graythresh(recovered);
im_binaria = imbinarize(recovered,level);
figure,imshow(im_binaria);
im_binaria = imcomplement(im_binaria);

label = bwlabel(im_binaria);
regions = regionprops(label,'Area');
area = cat(1,regions.Area);


propied = regionprops(label, 'BoundingBox', 'Area');

for i = 1:size(propied)
    if (propied(i).Area > 550 && propied(i).Area < 1500)
        rectangle('Position', propied(i).BoundingBox,'EdgeColor', 'r','LineWidth', 2)
    end
end
