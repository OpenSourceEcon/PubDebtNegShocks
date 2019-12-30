% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=numeratorrf(epsi)

global par index Endow;


%numerator fo the right handside equation in Euler equation
xxx =  par.b.*exp(epsi + par.mu(index)).*(par.b*exp(epsi + par.mu(index)).*(par.beta.*(1-par.b).*exp(par.mu(index) + 0.5*par.sigma^2) + Endow)).^(1-par.gam);







% **********************************************************************
% ********************************************************************