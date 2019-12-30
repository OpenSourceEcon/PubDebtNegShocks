% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2debtpdvsingle(epsi)

global par index transfer c_d kt_debt; 

%Rtplus1 is just R(t+1)
Rtplus1 = par.b.*exp(epsi+par.mu(index)).*(kt_debt(c_d,index).^(par.b-1));
C2 = Rtplus1.*kt_debt(c_d,index) + transfer(1,1,index);

xxx(:,1) = (C2).^(1-par.gamma(index));
