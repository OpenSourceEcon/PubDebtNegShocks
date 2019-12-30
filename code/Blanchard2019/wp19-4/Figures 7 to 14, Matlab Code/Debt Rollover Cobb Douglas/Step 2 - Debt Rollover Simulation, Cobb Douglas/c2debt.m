% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2debt(epsi)

global par index epsilon S;


%Consumption at second period for old generation
xxx(:,1) = (par.b.*exp(epsi + par.mu(index)).*S(2,:,index) + epsilon(1,:,index)).^(1-par.gamma(index));




% **********************************************************************
% ********************************************************************