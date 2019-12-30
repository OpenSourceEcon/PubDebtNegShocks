% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=numerator(X)

global par dt index kt;

%Define Mt as identified in Equation 20.
mt = par.b .* (kt.^par.epsilon) + (1-par.b);

%The calculation is long, so I'm dividing it into two parts.
partA = ( exp(X+par.mu(index)) * par.b) .* (kt.^(par.epsilon-1)) .*(mt.^((1-par.epsilon)/par.epsilon));
partB = ( ( exp(X+par.mu(index)) * par.b) .* (kt.^(par.epsilon-1)) .*(mt.^((1-par.epsilon)/par.epsilon)).*kt + dt).^(-par.gamma(index));

xxx= partA.*partB;

%[Old: xxx =  par.b.*exp(X + par.mu(index)).*(par.b*exp(X + par.mu(index)).*kt + dt).^(-par.gamma(index));]

% **********************************************************************
% ********************************************************************