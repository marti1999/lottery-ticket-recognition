clear all; 
close all;
clc

%% image alignment

fixed  = im2gray(imread('../dataset/base_loteria.jpg'));
moving = im2gray(imread('../dataset/IMG_20191217_153350-1024x758.jpg'));

% imshowpair(fixed,moving,'montage')
% tformEstimate = imregcorr(moving,fixed);
% Rfixed = imref2d(size(fixed));
% movingReg = imwarp(moving,tformEstimate,'OutputView',Rfixed);
% imshowpair(fixed,movingReg,'montage')

% [optimizer,metric] = imregconfig('multimodal');
% movingRegisteredDefault = imregister(moving,fixed,'affine',optimizer,metric);
% imshowpair(fixed,movingRegisteredDefault,'montage')


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
% imshowpair(fixed,recovered,'montage')
tformEstimate = imregcorr(recovered,fixed);
Rfixed = imref2d(size(fixed));
movingReg = imwarp(moving,tformEstimate,'OutputView',Rfixed);
figure, imshowpair(fixed,movingReg,'montage')

[optimizer,metric] = imregconfig('multimodal');
movingRegisteredDefault = imregister(recovered,fixed,'affine',optimizer,metric);
figure, imshowpair(fixed,movingRegisteredDefault,'montage')
