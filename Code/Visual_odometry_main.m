clear all;
close all;
% Video Writer
v = VideoWriter('Testv11','MPEG-4');
v.FrameRate = 30;
open(v);

%Extract the camera parameters for each image
[fx, fy, cx, cy, G_camera_image, LUT] = ReadCameraModel('../input/stereo/centre','../input/model');
k = [fx, 0, cx;
0, fy, cy;
0, 0, 1];

cameraParams = cameraParameters('IntrinsicMatrix',k');
Drift_traj = zeros(3873,1);
pos_init = [0 0 0];
position_1 = [0 0 0];
Rotation_pos_1 = [1 0 0;
                 0 1 0
                 0 0 1];
position_2 = [0 0 0];
Rotation_pos_2 = [1 0 0;
                  0 1 0
                  0 0 1];
position_3 = [0 0 0];
Rotation_pos_3 = [1 0 0;
                  0 1 0
                  0 0 1];
              
cd ../input/stereo/centre
images.filename = ls('*png');
%  size_im = size(images.filename);
cd ../../../Code 
for Image_seq_number = 8:3872

cd ../input/stereo/centre

I_a = imread(images.filename(Image_seq_number,:));
J = demosaic(I_a,'gbrg');
% imshow(RGB)

I_b = imread(images.filename(Image_seq_number+1,:));
I_b = demosaic(I_b,'gbrg');

% cd ../../
I_a1 = UndistortImage(J, LUT);
I_b1 = UndistortImage(I_b, LUT);
I_a1 = imgaussfilt(I_a1, 0.8);
I_b1 = imgaussfilt(I_b1, 0.8);

I_a1 = rgb2gray(I_a1);
I_b1 = rgb2gray(I_b1);

cd ../../../Code 
disp(Image_seq_number);
%% Feature extraction from both the images 
h1 = detectSURFFeatures(I_a1);
h2 = detectSURFFeatures(I_b1);

[features1,valid_points1] = extractFeatures(I_a1, h1);
[features2,valid_points2] = extractFeatures(I_b1, h2);

indexPairs = matchFeatures(features1,features2, 'MaxRatio', 0.3);
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);


%% Fundamental Matrix with RANSAC using MATLAB FUNCTION
%Reference:https://www.mathworks.com/help/vision/ref/estimatefundamentalmatrix.html
[F_RANSAC, inliersIndex] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'Method','RANSAC','NumTrials',2000,'DistanceThreshold',1e-3);

p1=matchedPoints1.Location;
p2=matchedPoints2.Location;

pp1=p1';pp2=p2';
pp1(3,:) = matchedPoints1.Scale'; 
pp2(3,:)=matchedPoints2.Scale';
p1=p1';p2=p2';
F = Fnda_matrix(p1, p2);

% FundRansac Matrix Reference(imported from link below and made changes as per our need
% :http://inside.mines.edu/~whoff/courses/EENG512/lectures/ 
% To verify the error between 8 point and 
% MLESAC (maximum likelihood estimator sample consensus) algorithms
[F1,~] = fundRANSAC(pp1,pp2,I_a1,I_b1);

P1_X = matchedPoints1.Location(:,1);
P1_Y = matchedPoints1.Location(:,2);
inliers_1 = [P1_X(inliersIndex) P1_Y(inliersIndex)];

P2_X = matchedPoints2.Location(:,1);
P2_Y = matchedPoints2.Location(:,2);
inliers_2 = [P2_X(inliersIndex) P2_Y(inliersIndex)];

%% Essential Matrix
p1=p1';p2=p2';
[E1, R1, t1] = Essential_Mat(F,k,cameraParams,p1, p2);

%Using Matlab Functions
[E2, R2, t2] = Essential_Mat(F_RANSAC,k,cameraParams, inliers_1, inliers_2);
[E3, R3, t3] = Essential_Mat(F1,k,cameraParams,p1, p2);
% [relativeOrientation,relativeLocation] = relativeCameraPose(F,cameraParams,inliers_1,inliers_2);
% R3=relativeOrientation;
% t3 =relativeLocation;
%% Positions of Points 
%Without using Matlab functions
Rotation_pos_1 = R1 * Rotation_pos_1;
position_1 = position_1 + t1 * Rotation_pos_1;
%USing Matlab Functions
Rotation_pos_2 = R2 * Rotation_pos_2;
position_2 = position_2 + t2 * Rotation_pos_2;
%Custom Ransac Algorithm 
Rotation_pos_3 = R3 * Rotation_pos_3;
position_3 = position_3 + t3 * Rotation_pos_3;

Drift_traj(Image_seq_number) = norm(position_3 - position_2);

figure(1)
subplot(2,1,1)
showMatchedFeatures(I_a1,I_b1,matchedPoints1,matchedPoints2);
title('Matched Features');

subplot(2,1,2)
plot(position_1(1),position_1(3),'b*',position_2(1),position_2(3),'ro',-position_3(1),position_3(3),'g*')
% title(['Blue* =Est.,RedO = RANSAC, Error is: ' num2str(Drift_traj(Image_seq_number))])
title('Blue:8 point,Red:Maltab Estimation Function,Green:Custom RANSAC Function')
% text(200,200,Drift_traj(Image_seq_number),'Color','red','FontSize',20)


hold on

frame = getframe(gcf);
writeVideo(v,frame);

pause(0.001);
end

Total_Drift = rms(Drift_traj);
Total_Drift

cd ../../..
close(v)


