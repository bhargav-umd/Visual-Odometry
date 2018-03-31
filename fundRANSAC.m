function [F,indicesBest] = fundRANSAC(f1match, f2match, I1, I2)
% Fit a fundamental matrix to corresponding points.  
% Inputs:
%   f1match, f2match:  corresponding points, each col is [x;y;scale;theta]
%   I1,I2:  the two images.
% Outputs:
%   F:  the 3x3 fundamental matrix
%   indicesBest:  is true where we have an inlier
% 
% Uses the MLESAC (maximum likelihood estimator sample consensus) algorithm
% to find inliers.  See the paper: "MLESAC: A new robust estimator with
% application to estimating image geometry" by Torr and Zisserman, at
% http://www.robots.ox.ac.uk/~vgg/publications/2000/Torr00/torr00.pdf

N = size(f1match,2);    % Number of corresponding point pairs
pts1 = f1match(1:2,:)'; % Get x,y coordinates from image 1, size is (N,2)
pts2 = f2match(1:2,:)'; % Get x,y coordinates from image 2, size is (N,2)

% Estimate the uncertainty of point locations in the second image.  We'll
% assume Gaussian distributed residual errors, with a sigma of S pixels.
S = 1.0;    % Sigma of errors of points found in lowest scale

% Create a Nx1 matrix of sigmas for each point.  Points at larger scales
% have correspondingly larger sigmas.
sigs = S * f2match(3,:)';  % Assume same for x,y

% Now, if a point is an outlier, its residual error can be anything; i.e.,
% any value between 0 and the size of the image.  Let's
% assume a uniform probability distribution.
[height,width] = size(I1);
pOutlier = 1/max(height,width);

% Estimate the fraction of inliers.  We'll actually estimate this later
% from the data, but start with a worst case scenario assumption.
inlierFraction = 0.1;

% Let's say that we want to get a good sample with probability Ps.
Ps = 0.99;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine the required number of iterations.  Since we are using the 8
% point algorithm, we need to take samples of 8 points to fit a fundamental
% matrix.

% Estimated number of inliers among the N points.
nInlier = round(N*inlierFraction); 

% The number of ways to pick a set of k=8 points, out of N points is
% "N-choose-8", which is fact(N)/(fact(k)*fact(N-k)).

% This is the number of ways to choose 8 points among all the points.
n = N*(N-1)*(N-2)*(N-3)*(N-4)*(N-5)*(N-6)*(N-7)/factorial(8);

% This is the number of ways to choose 8 points among all the inliers.
m = nInlier*(nInlier-1)*(nInlier-2)*(nInlier-3)* ...
    (nInlier-4)*(nInlier-5)*(nInlier-6)*(nInlier-7)/factorial(8);

p = m/n;        % probability that any sample of 8 points is all inliers
nIterations = log(1-Ps) / log(1 - p);
nIterations = ceil(nIterations);


sample_count = 0;   % The number of Ransac trials
pBest = -Inf;       % Best probability found so far (actually, the log)
while sample_count < nIterations
    % Grab 8 matching points at random from the set of all correspondences.
    v = randperm(N);
    p1 = pts1(v(1:8),:);
    p2 = pts2(v(1:8),:);
    p1 = p1';       % Get points as a 2xN matrix
    p2 = p2';
    
    % Fit a fundamental matrix to the corresponding points in this subset.
   [F,fError] = Fnda_matrix(p1, p2)
    if fError
        continue;
    end
   
    sample_count = sample_count + 1;
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get error residuals for all pairs of points.
    dp = zeros(N,1);
    for i=1:N
        % The product l=F*p2 is the equation of the epipolar line corresponding
        % to p2, in the first image.  Here, l=(a,b,c), and the equation of the
        % line is ax + by + c = 0.
        x2 = pts2(i,:)';        % Point in second image
        l = F * [x2;1];         % Epipolar line in first image
        
        % The equation of the line is ax + by + c = 0.
        % The distance from a point p1=(x1,y1,1) to a line with parameters
        % l=(a,b,c) is   d = abs(p1' * l)/sqrt( a^2 + b^2 )
        % (see
        % http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html)
        x1 = pts1(i,:)';        % Point in first image
        dp(i) = abs(([x1;1]' * l))/sqrt( l(1)^2 + l(2)^2 );
    end
        
    % Compute the probability that each residual error, dpi, could be an
    % inlier.  This is the equation of a Gaussian; ie.
    %   G(dpi) = exp(-dpi^2/(2*sig^2))/(sqrt(2*pi)*sig)
    dp = dp ./ sigs;      % Scale by the sigmas
    rsq = dp.^2;    % residual squared distance errors
    numerator = exp(-rsq/2);    % Numerator of Gaussians, size is Nx1
    denominator = sqrt(2*pi)*sigs;    % Denominator, size Nx1
    
	% These are the probabilities of each point, if they are inliers (ie,
	% this is just the Gaussian probability of the error).
    pInlier = numerator./denominator;
    
    % Let's define inliers to be those points whose inlier probability is
    % greater than the outlier probability.
    indicesInliers = (pInlier > pOutlier);
    nInlier = sum(indicesInliers);
    
    % Update the number of iterations required
    if nInlier/N > inlierFraction
        inlierFraction = nInlier/N;
        
        % This is the number of ways to choose 8 points among all the inliers.
        m = nInlier*(nInlier-1)*(nInlier-2)*(nInlier-3)* ...
            (nInlier-4)*(nInlier-5)*(nInlier-6)*(nInlier-7)/factorial(8);

        p = m/n;  % probability that any sample of 4 points is all inliers
        nIterations = log(1-Ps) / log(1 - p);
        nIterations = ceil(nIterations);

    end
    
    % Net probability of the data (log)
    p = sum(log(pInlier(indicesInliers))) + (N-nInlier)*log(pOutlier);
    
    % Keep this solution if probability is better than the best so far.
    if p>pBest
        pBest = p;
        nBest = nInlier;
        indicesBest = indicesInliers;
    
        o = size(I1,2) ;
        for i=1:size(f1match,2)
            x1 = f1match(1,i);
            y1 = f1match(2,i);
            x2 = f2match(1,i);
            y2 = f2match(2,i);
            
            if ~isempty(find(i==v(1:8), 1))
                % This is one of the points in the initial random sample.
                mycolor = 'b';
            elseif indicesBest(i)
                % This is one of the inliers.
                mycolor = 'g';
            else
                % This is an outlier.
                mycolor = 'r';
            end
%             
%      
        end            
%      
    end
end


% Ok, refit the fundamental matrix using all the inliers. 
p1 = pts1(indicesBest,:);
p2 = pts2(indicesBest,:);
p1 = p1';       % Get points as a 2xN matrix
p2 = p2';

% Fit a fundamental matrix to the inlier points.
[F, ~] = Fnda_matrix(p1,p2);

% Get error residuals for all points, using the final F.
dp = zeros(N,1);
for i=1:N
    % The product l=F*p2 is the equation of the epipolar line corresponding
    % to p2, in the first image.  Here, l=(a,b,c), and the equation of the
    % line is ax + by + c = 0.
    x2 = pts2(i,:)';        % Point in second image
    l = F * [x2;1];         % Epipolar line in first image
    
    % The equation of the line is ax + by + c = 0.
    % The distance from a point p1=(x1,y1,1) to a line with parameters
    % l=(a,b,c) is   d = abs(p1' * l)/sqrt( a^2 + b^2 )
    % (see
    % http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html)
    x1 = pts1(i,:)';        % Point in first image
    dp(i) = abs(([x1;1]' * l))/sqrt( l(1)^2 + l(2)^2 );
end
dp = dp ./ sigs;      % Scale by the sigmas
rsq = dp.^2;    % residual squared distance errors
    
rsq = rsq(indicesBest);     % Keep only values for the inliers

% fprintf('Final RMS error=%f pixels\n', sqrt(sum(rsq)/nBest));

return

        