% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=denom(epsi)

global par index epsilon S;


%Consumption at second period for old generation
xxx(:,1) = (par.b.*exp(epsi + par.mu(index)).*S(2,:,index) + epsilon(1,:,index)).^(-par.gamma(index));




% **********************************************************************
% ********************************************************************