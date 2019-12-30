function f = solveRF(x)

global par index Q At2; 

    for j=1:size(At2,1)            
    temp_r = par.b*At2(j,index);
    RHSnum(j) = temp_r * (temp_r + Q * (x-temp_r))^(-par.gamma(index)); 
    RHSden(j) = (temp_r + Q * (x-temp_r))^(-par.gamma(index));
    end

    RHSnummean=mean(RHSnum);
    RHSdenmean=mean(RHSden);
                    
f = RHSnummean/RHSdenmean - x;                    
