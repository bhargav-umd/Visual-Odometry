function [F,fError] = Fnda_matrix(p1, p2)
% Fit a fundamental matrix to the corresponding points in p1 and p2.
% p1,p2 are each 2xN in size, where each column is [x;y;1].

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
F = T2' * Fscale * T1;
fError = false; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%our fundamental matrix 
end
 


