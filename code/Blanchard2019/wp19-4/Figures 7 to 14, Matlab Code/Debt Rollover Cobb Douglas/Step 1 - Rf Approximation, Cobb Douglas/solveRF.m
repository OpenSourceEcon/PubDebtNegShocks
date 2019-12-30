function f = solveRF(x)

global K par index Q At2; 

    for j=1:size(At2,1)
    temp_r = par.b*At2(j,index)* K^(par.b-1);
    RHSnum(j) = temp_r * (temp_r + Q * (x-temp_r))^(-par.gamma(index)); 
    RHSden(j) = (temp_r + Q * (x-temp_r))^(-par.gamma(index));
    
%     RHSnum(j) = (par.b * At2(j,index) * K^(par.b-1)) * (  (par.b * At2(j,index) * K^(par.b-1)) + Q * (x - (par.b * At2(j,index) * K^(par.b-1))) )^(-par.gamma(index)); 
%     RHSden(j) = ( (par.b * At2(j,index) * K^(par.b-1)) + Q * (x - (par.b * At2(j,index) * K^(par.b-1)) ) ) ^(-par.gamma(index));
    end

    RHSnummean=mean(RHSnum);
    RHSdenmean=mean(RHSden);
                    
f = RHSnummean/RHSdenmean - x;
                    
