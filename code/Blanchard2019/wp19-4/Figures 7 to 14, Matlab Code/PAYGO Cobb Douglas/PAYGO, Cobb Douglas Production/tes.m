% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2notransfer(epsi)

global par index S;


%Consumption at second period for old generation
xxx(:,1) = (par.b.*exp(epsi + par.mu(index)).*S(1,:,index)).^(1-par.gamma(index));



% **********************************************************************
% ********************************************************************