clear all; 
close all;
clc

%% image alignment
affine = true;
fixed  = rgb2gray(imread('../dataset/base_loteria.jpg'));
imMove = imread('../dataset/2.jpg');
moving = rgb2gray(imMove);



%%%%
% 1 ---------2
% |          |
% |          |
% |          |
% 4 ---------3
%%%%

[x, y] = obtenirPunts(imMove, "Seleccionar les cantonades");
save("puntsHomo2.mat", "x", "y");
result = homografiaManual(x, y, moving);

% result = homografia(fixed, result, BW1, closimg,2);

% figure(), imshow(result, []);

if (affine)
    BW1 = edge(fixed,'canny');
    BW2 = edge(result,'canny');
else
    BW1 = fixed;
    BW2 = result;
end

result = homografia(fixed, result, BW1, BW2, 1);


%% segona part de homografia
% [optimizer,metric] = imregconfig('multimodal');
% movingRegisteredDefault = imregister(recovered,fixed,'affine',optimizer,metric);
% figure, imshowpair(fixed,movingRegisteredDefault,'montage')

%% trobar nums
% https://www.mathworks.com/help/vision/ug/recognize-text-using-optical-character-recognition-ocr.html
% https://www.mathworks.com/matlabcentral/answers/377444-why-ocr-function-doesn-t-recognize-the-numbers
% https://www.mathworks.com/matlabcentral/answers/225781-deleting-or-selecting-rows-of-a-struct-with-a-condition

% figure(), imshow(result);
icrop = imcrop(result, [200 45 280 80]);
figure(), imshow(icrop);

level = graythresh(icrop); 
im_binaria = imbinarize(icrop,level);
figure,imshow(im_binaria);
im_binaria = imcomplement(im_binaria);
imshow(im_binaria);
se = strel('square',3);
im_binaria2 = imopen(im_binaria, se);
im_binaria2 = imclose(im_binaria2, se);
% imshowpair(im_binaria2, im_binaria, 'montage');
my_image = im_binaria2;

imshow(my_image)
s = regionprops(my_image,'BoundingBox', 'Area');
areas = [s.Area];
g = areas > 600 & areas < 1500;
s = s(g);
bboxes = vertcat(s(:).BoundingBox);

for i=1:size(bboxes,1)
    rectangle('Position',bboxes(i,:), 'EdgeColor', 'r', 'LineWidth', 2);
    crop = imcrop(my_image, bboxes(i,:));
    number = num2str(i);
    name = strcat('../numbers/image_', number, '.jpg');
    imwrite(crop, name);
end


% eliminem els laterals per tal que no toquin les zones blanques.
e = strel('square',2);
my_image = imerode(my_image, se);

% ocr automàtic
ocrResults = ocr(my_image,bboxes,'CharacterSet','0123456789','TextLayout','Character');
wordsAutomatic = {ocrResults(:).Text}';

% el segon parametre de moment pot ser 'mse, 'imsim' i 'psnr'
wordsManual = ocr_manual(my_image, 'psnr', bboxes);

% ocr manual


function words = ocr_manual(img_original, type, bboxes)
    words = zeros(1, 5);

    for j=1:5

        bw2 = imcrop(img_original, bboxes(j,:));
        for i=0:9
            name = strcat('../numbers/', int2str(i), '.jpg');
            gt = im2uint8(imbinarize(im2gray(imread(name))));
            img = im2uint8(imresize(bw2, size(gt)));
            
            if (strcmp(type, 'mse'))
                results(i+1) = -immse(gt, img);
                continue;
            end
            if (strcmp(type, 'imsim'))
                results(i+1) = imsim(gt, img);
                continue;
            end
            if (strcmp(type, 'psnr'))
                results(i+1) = psnr(gt, img);
                continue;
            end
        end
        [~, idx] = max(results);
        words(j) = idx-1;

    end

    
end

function result = homografia(fixed, moving, img, img2, homografia)

    ptsfixed  = detectSURFFeatures(img);
    % ptsfixed  = detectFASTFeatures(fixed);
    ptsmoving = detectSURFFeatures(img2);
    % ptsmoving = detectFASTFeatures(moving);
    [featuresfixed,  validPtsfixed]  = extractFeatures(fixed,  ptsfixed);
    [featuresmoving, validPtsmoving] = extractFeatures(moving, ptsmoving);
    indexPairs = matchFeatures(featuresfixed, featuresmoving);
    matchedfixed  = validPtsfixed(indexPairs(:,1));
    matchedmoving = validPtsmoving(indexPairs(:,2));
%     figure;
%     showMatchedFeatures(fixed,moving,matchedfixed,matchedmoving);
%     title('Putatively matched points (including outliers)');
    if(homografia == 1)
        [tform, inlierIdx] = estimateGeometricTransform2D(matchedmoving, matchedfixed, 'similarity');
    else
        [tform, inlierIdx] = estimateGeometricTransform2D(matchedmoving, matchedfixed, 'similarity');
        
    end
%     inliermoving = matchedmoving(inlierIdx, :);
%     inlierfixed  = matchedfixed(inlierIdx, :);
%     figure;
%     showMatchedFeatures(fixed,moving,inlierfixed,inliermoving);
%     title('Matching points (inliers only)');
%     legend('ptsfixed','ptsmoving');
    
    
    outputView = imref2d(size(fixed));
    result  = imwarp(moving,tform,'OutputView',outputView);
    figure, imshowpair(fixed,result,'montage');

end

function result = homografiaManual(x1, y1, im)
    load puntsProva.mat;
    M = [];
    for i=1:4
       M = [  M ;
            x(i) y(i) 1 0 0 0 -x1(i)*x(i) -x1(i)*y(i) -x1(i);
            0 0 0 x(i) y(i) 1 -y1(i)*x(i) -y1(i)*y(i) -y1(i)];
    end
    % soluciono el sistema
    [u,s,v] = svd( M );
    H = reshape( v(:,end), 3, 3 )';
    H = H / H(3,3);
    % fi DLT 1
    
    tform12 = projective2d(inv(H')); % marc de referència
    Orig = imref2d(size(im));
    result = imwarp(im,Orig,tform12);
end

function [x, y] = obtenirPunts(im , text)
    imshow(im);
    title(text)
    x=[]; y=[];
    for j=1:4
        zoom on;  
        pause();
        zoom off;
        [x(j),y(j)]=ginput(1);
        zoom out;
    end
end

