% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=denominator(X) %Where X is a random draw from a normal variable of mean zero

global par dt index kt;

%Define Mt as identified in Equation 20.
mt = par.b .* (kt.^par.epsilon) + (1-par.b);

%The calculation is long, so I'm dividing it into two parts.
partB = ( ( exp(X+par.mu(index)) * par.b) .* (kt.^(par.epsilon-1)) .*(mt.^((1-par.epsilon)/par.epsilon)).*kt + dt).^(1-par.gamma(index));

xxx= partB;

%[Old: xxx =  par.b.*exp(X + par.mu(index)).*(par.b*exp(X + par.mu(index)).*kt + dt).^(-par.gamma(index));]

% **********************************************************************
% ********************************************************************