clear all; 
close all;
clc

%% image alignment

fixed  = im2gray(imread('../dataset/base_loteria.jpg'));
moving = im2gray(imread('../dataset/2.jpg'));


% https://www.mathworks.com/help/vision/ug/find-image-rotation-and-scale-using-automated-feature-matching.html
ptsfixed  = detectSURFFeatures(fixed);
ptsmoving = detectSURFFeatures(moving);
[featuresfixed,  validPtsfixed]  = extractFeatures(fixed,  ptsfixed);
[featuresmoving, validPtsmoving] = extractFeatures(moving, ptsmoving);
indexPairs = matchFeatures(featuresfixed, featuresmoving);
matchedfixed  = validPtsfixed(indexPairs(:,1));
matchedmoving = validPtsmoving(indexPairs(:,2));
figure;
showMatchedFeatures(fixed,moving,matchedfixed,matchedmoving);
title('Putatively matched points (including outliers)');

[tform, inlierIdx] = estimateGeometricTransform2D(...
    matchedmoving, matchedfixed, 'similarity');
inliermoving = matchedmoving(inlierIdx, :);
inlierfixed  = matchedfixed(inlierIdx, :);
figure;
showMatchedFeatures(fixed,moving,inlierfixed,inliermoving);
title('Matching points (inliers only)');
legend('ptsfixed','ptsmoving');

Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scaleRecovered = sqrt(ss*ss + sc*sc);
thetaRecovered = atan2(ss,sc)*180/pi;

outputView = imref2d(size(fixed));
recovered  = imwarp(moving,tform,'OutputView',outputView);
figure, imshowpair(fixed,recovered,'montage');


%% segona part de homografia
% [optimizer,metric] = imregconfig('multimodal');
% movingRegisteredDefault = imregister(recovered,fixed,'affine',optimizer,metric);
% figure, imshowpair(fixed,movingRegisteredDefault,'montage')

%% trobar nums
% https://www.mathworks.com/help/vision/ug/recognize-text-using-optical-character-recognition-ocr.html
% https://www.mathworks.com/matlabcentral/answers/377444-why-ocr-function-doesn-t-recognize-the-numbers
% https://www.mathworks.com/matlabcentral/answers/225781-deleting-or-selecting-rows-of-a-struct-with-a-condition

level = graythresh(recovered);
im_binaria = imbinarize(recovered,level);
figure,imshow(im_binaria);
im_binaria = imcomplement(im_binaria);
imshow(im_binaria);
se = strel('square',3);
im_binaria2 = imopen(im_binaria, se);
im_binaria2 = imclose(im_binaria2, se);
imshowpair(im_binaria2, im_binaria, 'montage');
my_image = im_binaria2;

imshow(my_image)
s = regionprops(my_image,'BoundingBox', 'Area');
areas = [s.Area];
g = areas > 750 & areas < 1500;
s = s(g);
bboxes = vertcat(s(:).BoundingBox);

% Sort boxes by image height
% [~,ord] = sort(bboxes(:,2));
% bboxes = bboxes(ord,:);

% eliminem els laterals per tal que no toquin les zones blanques.
e = strel('square',2);
my_image = imerode(my_image, se);
ocrResults = ocr(my_image,bboxes,'CharacterSet','0123456789','TextLayout','Character');
words = {ocrResults(:).Text}';








% label = bwlabel(im_binaria);
% regions = regionprops(label,'Area');
% area = cat(1,regions.Area);
% 
% 
% propied = regionprops(label, 'BoundingBox', 'Area');
% 
% index = 1;
% for i = 1:size(propied)
%     if (propied(i).Area > 750 && propied(i).Area < 1500)
%         rectangle('Position', propied(i).BoundingBox,'EdgeColor', 'r','LineWidth', 2)
%         roi(index) = propied(i);
%         index = index +1;
%     end
% end
% 
% % eliminem els laterals per tal que no toquin les zones blanques.
% e = strel('square',2);
% im_binaria2 = imerode(im_binaria2, se);
% imshow(im_binaria2);
% 
% for i=1:size(roi, 2)
%     box = roi(1, i).BoundingBox;
%     results(index) = ocr(im_binaria2, box, 'CharacterSet' ,'0123456789', 'TextLayout', 'Character')
% 
% end


