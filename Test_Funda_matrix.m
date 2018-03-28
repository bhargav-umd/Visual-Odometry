% function [F,R,T] = Funda_matrix(p1, p2)
% Fit a fundamental matrix to the corresponding points in p1 and p2.
% p1,p2 are each 3xN in size, where each column is [x;y;1].

% Check if they passed in a 2xN matrix instead of a 3xN matrix.
if size(p1,1) == 2
    p1 = [p1; ones(1,size(p1,2))];   % append 1's to the last row
end
if size(p2,1) == 2
    p2 = [p2; ones(1,size(p2,2))];   % append 1's to the last row
end   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scale and translate image points so that the centroid of
% the points is at the origin, and the average distance of the points to the
% origin is equal to sqrt(2).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xn = p1(1:2,:);             % xn is a 2xN matrix
N = size(xn,2);
t = (1/N) * sum(xn,2);      % this is the (x,y) centroid of the points
xnc = xn - t*ones(1,N);     % center the points; xnc is a 2xN matrix
dc = sqrt(sum(xnc.^2));     % dist of each new point to 0,0; dc is 1xN vector
davg = (1/N)*sum(dc);       % average distance to the origin
s = sqrt(2)/davg;           % the scale factor, so that avg dist is sqrt(2)
T1 = [s*eye(2), -s*t ; 0 0 1];
p1s = T1 * p1;

xn = p2(1:2,:);             % xn is a 2xN matrix
N = size(xn,2);
t = (1/N) * sum(xn,2);      % this is the (x,y) centroid of the points
xnc = xn - t*ones(1,N);     % center the points; xnc is a 2xN matrix
dc = sqrt(sum(xnc.^2));     % dist of each new point to 0,0; dc is 1xN vector
davg = (1/N)*sum(dc);       % average distance to the origin
s = sqrt(2)/davg;           % the scale factor, so that avg dist is sqrt(2)
T2 = [s*eye(2), -s*t ; 0 0 1];
p2s = T2 * p2;


% Compute fundamental matrix F from point correspondences.
% We know that p1s' F p2s = 0, where p1s,p2s are the scaled image coords.
% We write out the equations in the unknowns F(i,j)
%   A x = 0
A = [p1s(1,:)'.*p2s(1,:)'   p1s(1,:)'.*p2s(2,:)'  p1s(1,:)' ...
    p1s(2,:)'.*p2s(1,:)'   p1s(2,:)'.*p2s(2,:)'  p1s(2,:)' ...
    p2s(1,:)'             p2s(2,:)'  ones(length(p1s),1)];

% The solution to Ax=0 is the singular vector of A corresponding to the
% smallest singular value; that is, the last column of V in A=UDV'
[U,D,V] = svd(A);
x = V(:,size(V,2));                  % get last column of V

% Put unknowns into a 3x3 matrix.  Transpose because Matlab's "reshape"
% uses the order F11 F21 F31 F12 ...
Fscale = reshape(x,3,3)';

% Force rank=2
[U,D,V] = svd(Fscale);
Fscale = U*diag([D(1,1) D(2,2) 0])*V';
% Undo scaling
F = T1' * Fscale * T2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract motion parameters from essential matrix.
% We know that E = [tx] R, where
%  [tx] = [ 0 -t3 t2; t3 0 -t1; -t2 t1 0]
%
% If we take SVD of E, we get E = U diag(1,1,0) V'
% t is the last column of U
[U,D,V] = svd(F);
W = [0 -1 0; 1 0 0; 0 0 1];

Hresult_c2_c1(:,:,1) = [ U*W*V'   U(:,3) ; 0 0 0 1];
Hresult_c2_c1(:,:,2) = [ U*W*V'  -U(:,3) ; 0 0 0 1];
Hresult_c2_c1(:,:,3) = [ U*W'*V'  U(:,3) ; 0 0 0 1];
Hresult_c2_c1(:,:,4) = [ U*W'*V' -U(:,3) ; 0 0 0 1];
 
% make sure each rotation component is a legal rotation matrix
for k=1:4
    if det(Hresult_c2_c1(1:3,1:3,k)) < 0
        Hresult_c2_c1(1:3,1:3,k) = -Hresult_c2_c1(1:3,1:3,k);
    end
end
 
% Pick the first point to reconstruct
p1x = [ 0        -p1(3,1)   p1(2,1);   % skew symmetric matrix
        p1(3,1)   0        -p1(1,1);
       -p1(2,1)   p1(1,1)   0  ];
 
p2x = [ 0        -p2(3,1)   p2(2,1);
        p2(3,1)   0        -p2(1,1);
       -p2(2,1)   p2(1,1)   0  ];
 
M1 = [ 1 0 0 0; 0 1 0 0; 0 0 1 0 ];
for i=1:4
    Hresult_c1_c2 = inv(Hresult_c2_c1(:,:,i));
    M2 = Hresult_c1_c2(1:3,1:4);
    
    A = [ p1x * M1; p2x * M2 ];
    % The solution to AP=0 is the singular vector of A corresponding to the
    % smallest singular value; that is, the last column of V in A=UDV'
    [U,D,V] = svd(A);
    P = V(:,4);                     % get last column of V
    P1est = P/P(4);                 % normalize
 
    P2est = Hresult_c1_c2 * P1est;
 
    if P1est(3) > 0 && P2est(3) > 0
        Hest_c2_c1 = Hresult_c2_c1(:,:,i);    % We've found a good solution
        break;      % break out of for loop; can stop searching
    end
end
R = Hest_c2_c1(1:3,1:3);
T= Hest_c2_c1(1:3,4);

% end
 


