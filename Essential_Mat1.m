function [E,R,T] =Essential_Mat1(F,k,p1,p2)
% Fit a fundamental matrix to the corresponding points in p1 and p2.
% p1,p2 are each 3xN in size, where each column is [x;y;1].

E = k' * F * k;

[U,D,V] = svd(E);
W = [0 -1 0;
    1 0 0;
    0 0 1];

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
pos_count =zeros(4,1);

for counter = 1:size(p1,2)

% Pick the first point to reconstruct
p1x = [ 0           -p1(3,counter)   p1(2,counter);   % skew symmetric matrix
    p1(3,counter)       0             -p1(1,counter);
    -p1(2,counter)   p1(1,counter)          0  ];

p2x = [ 0        -p2(3,counter)   p2(2,counter);
    p2(3,counter)   0        -p2(1,counter);
    -p2(2,counter)   p2(1,counter)   0  ];

M1 = [ 1 0 0 0; 0 1 0 0; 0 0 1 0 ];
% condition_1=0;
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
%         Hest_c2_c1 = Hresult_c2_c1(:,:,i);    % We've found a good solution
        %         condition_1=1;
%         break;      % break out of for loop; can stop searching
    pos_count(i)=pos_count(i)+1;
    end
    
end
end
% if condition_1==0
%      Hest_c2_c1 = Hresult_c2_c1(:,:,1)
% end
[~,indexx] = max(pos_count);
Hest_c2_c1 = Hresult_c2_c1(:,:,indexx);
R = Hest_c2_c1(1:3,1:3);
T= Hest_c2_c1(1:3,4);

end



