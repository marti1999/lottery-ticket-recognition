clc
clear
close all

% a=imread('D:\iit mandi\Sem2\Computer Vision\assignment 1\transformed.png');
% a = imread('D:\tuts1\images\1small.jpg');
% a = imread('D:\tuts1\computer vision\u.png');
[file,path] = uigetfile('*.*');
f1 = fullfile(path,file);
if prod(double(file) == 0) && prod(double(path) == 0)
    return
end   
a = imread(f1);
tic
b=size(a);

if size(b,2)==3
    a = rgb2gray(a);
end

n=3;
n1=ceil(n/2);
a=double(a);

hpf = [1,1,1;0 0 0;-1 -1 -1];
lpf=[1,0,-1;1,0,-1;1,0,-1];

c=0;
h=0;
d = zeros(b(1)-2*n1,b(2)-2*n1);
g = zeros(b(1)-2*n1,b(2)-2*n1);

%Loop for finding Ix
for i=n1:b(1)-n1
    for j=n1:b(2)-n1
        p=1;
        for k=1:n
            for l=1:n
            c=c+a(i-n1+k,j-n1+l)*lpf(k,l);
            p=p+1;
            end
        end
        d(i,j)=c;
        c=0;
    end
end
Ix=d;

%Loop for finding Iy
for i=n1:b(1)-n1
    for j=n1:b(2)-n1
        for k=1:n
            for l=1:n
            h=h+a(i-n1+k,j-n1+l)*hpf(k,l);
            end
        end
        
        g(i,j)=h;
        h=0;
    end
end
Iy=g;

Ix2 = Ix.*Ix;
Iy2 = Iy.*Iy;

Ixy = Ix.*Iy;

b1 = size(Ix);

m=5;
m1=ceil(m/2);
kh = 0.04;
T = 100000;
Cor = zeros(b1(1)-2*m1,b1(2)-2*m1);
Edg = zeros(b1(1)-2*m1,b1(2)-2*m1);
Fla = zeros(b1(1)-2*m1,b1(2)-2*m1);
R = zeros(b1(1)-2*m1,b1(2)-2*m1);
for i=m1:b1(1)-m1                                        %i scans rows of image 
    for j=m1:b1(2)-m1                                    %j scans columns of image  
        p=1;                                             %p is for pixels inside the filter on image 
        for k=1:m                                        %k scans the rows of filter on image 
            for l=1:m                                    %l scans the columns of filter on image 
            Ix2c(p)=Ix2(i-m1+k,j-m1+l);                  %putting the pixels inside the filter to a vector 
            Iy2c(p)=Iy2(i-m1+k,j-m1+l); 
            Ixyc(p)=Ixy(i-m1+k,j-m1+l); 
            p=p+1;
            end
        end
        H = 1/(m^2)*[sum(Ix2c),sum(Ixyc);sum(Ixyc),sum(Iy2c)];   %The harris matrix
            R(i,j) = det(H)-kh*(trace(H)).^2;                           %R threshold
            if R(i,j) > T
                Cor(i,j) = 1;
            elseif R(i,j)<-T*0.5
                Edg(i,j) = 1;
            else
                Fla(i,j)=1;
            end
            
    end
end

figure;
imshow(uint8(a));
figure;
imagesc(R);
figure;
imshow(Cor);
figure;
imshow(Edg);
figure;
imshow(Fla);

toc