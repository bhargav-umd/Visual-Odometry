function [E, R, t] = Essential_Mat(F,k,cameraParams, inliers1, inliers2)
E = k' * F * k;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract motion parameters from essential matrix.           %%
% We know that E = [tx] R, where                             %%
%  [tx] = [ 0 -t3 t2; t3 0 -t1; -t2 t1 0]                    %%
%                                                            %%                  
% If we take SVD of E, we get E = U diag(1,1,0) V'           %%
% t is the last column of U                                  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U, D, V] = svd(E);
% The solution to AP=0 is the singular vector of A corresponding to the
    % smallest singular value; that is, the last column of V in A=UDV'
e = (D(1,1) + D(2,2)) / 2;
D(1,1) = e;
D(2,2) = e;
D(3,3) = 0;
E = U * D * V';
[U, ~, V] = svd(E);

W = [0 -1 0;
     1 0 0; 
     0 0 1];
Z = [0 1 0; 
    -1 0 0; 
     0 0 0];

R1 = U * W * V';
% make sure each rotation component is a legal rotation matrix
if det(R1) < 0
    R1 = -R1;
end
R2 = U * W' * V';
if det(R2) < 0
    R2 = -R2;
end

Tx = U * Z * U';
t1 = [Tx(3, 2), Tx(1, 3), Tx(2, 1)];
t2 = -t1;
Rs = cat(3, R1, R1, R2, R2);%4 combinations of R and T
Ts = cat(1, t1, t2, t1, t2);%4 combinations of R and T

%% Choose the right solution 
num_Negatives = zeros(1, 4);
P1 = cameraMatrix(cameraParams, eye(3), [0,0,0]);
for kcount = 1:size(Ts, 1)
   P2 = cameraMatrix(cameraParams,Rs(:,:,kcount)', Ts(kcount, :));
% Triangulation
   points3D_1 = zeros(size(inliers1, 1), 3, 'like', inliers1);
   P1_a = P1';
   P2_a = P2';

   Camera_M1 = P1_a(1:3, 1:3);
   Camera_M2 = P2_a(1:3, 1:3);

   c1 = -Camera_M1 \ P1_a(:,4);
   c2 = -Camera_M2 \ P2_a(:,4);
%%making sure depth is positive 
   for counter = 1:size(inliers1,1)
      u1 = [inliers1(counter,:), 1]';
      u2 = [inliers2(counter,:), 1]';
      a1 = Camera_M1 \ u1;
      a2 = Camera_M2 \ u2;
      A = [a1, -a2];
      y = c2 - c1;

      alpha = (A' * A) \ A' * y;
      p = (c1 + alpha(1) * a1 + c2 + alpha(2) * a2) / 2;
      points3D_1(counter, :) = p';
   end
   points3D_2 = bsxfun(@plus, points3D_1 * Rs(:,:,kcount)', Ts(kcount, :));
   num_Negatives(kcount) = sum((points3D_1(:,3) < 0) | (points3D_2(:,3) < 0));
end

[~, idx] = min(num_Negatives);
R = Rs(:,:,idx)';
t = Ts(idx, :);
tNorm = norm(t);
if tNorm ~= 0
    t = t ./ tNorm;
end


R = R';
t = -t * R;
end
