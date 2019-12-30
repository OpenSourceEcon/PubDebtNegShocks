% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=denominator(epsi)

global par dt index S rf;


%numerator fo the right handside equation in Euler equation
xxx =  par.b.*exp(epsi + par.mu(index)) .* (par.b.*exp(epsi + par.mu(index)).*S + rf.*dt).^(-par.gamma(index));







% **********************************************************************
% ********************************************************************