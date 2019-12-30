function f = solveK(x)

global W par index endowment D At; 

    for j=1:size(At,1)
        a=At(j,index);
    RHSnum(j) = par.b * At(j,index) * (x^(par.b-1)) * (par.b * At(j,index) * (x^(par.b)) + D)^(-par.gamma(index)); 
    RHSden(j) = (par.b * At(j,index) * x^(par.b) + D)^(1-par.gamma(index));
    end

    RHSnummean=mean(RHSnum);
    RHSdenmean=mean(RHSden);
                    
f = W+endowment(index) - ((1-par.beta(index))/par.beta(index)) * (RHSnummean/RHSdenmean)^(-1) - D - x;
                    
