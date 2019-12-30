*===============================================================================
*
*
*				Low rates project - Historical analysis - US - comparison R - G 
*				
*				dataset from MeasuringWorth, Shiller, OMB, CBO forecasts after 2017
*
*
*										created from Main do.file
*							ADDED R and R_ADJ FOR MATURITY AND TAX ADJUSTMENT
*										
*				Amended version of Thomas Pellet's do file "r-g evidence 02 28 bis"
*										
*				Colombe Ladreit, Peterson institute for International economics
*
*									  7-10-2018 - version 1.0.0

*
*===============================================================================

*using yearly dataset based on MeasuringWorth and Shiller

clear all

*set maxvar 32000

*cd "R:\Colombe\requests from OB\MPK vs growth\US interest rates"


use "USinterestrates_final.dta"

set more off

*gen caseid = _n


tsset year


*===============================================================================
*
*				V-Generate 1+r/1+g
*
*===============================================================================
*/
/* modified september 18.  new series for the two adjusted rates, to correct for error in earlier formula. imported from r-g september 17.xls */

*Constructing 1+R/1+G using different interest rates
*===============================================================================

*With interest rate on 3 months r1y
*-----------------------------------------

gen rog2 = (1+r1y/100)/(1+g/100)
label variable rog2 "1+ 1Y T over 1 + nominal g."



*with interest rate on 10 year goverment bonds
*---------------------------------------------

gen rlog2 = (1+r10y/100)/(1+g/100)
label variable rlog2 "1+ 10-y T over 1+ nominal g."


*With interest rate adjusted for maturity
*-----------------------------------------

gen rog_mat2 = (1+r2/100)/(1+g/100)
label variable rog_mat2 "1+ 1Y T adj for mat over 1 + nominal g."


*with interest rate adjusted for maturity and tax
*---------------------------------------------

gen rog_adj2 = (1+r_adj/100)/(1+g/100)
label variable rog_adj2 "1+ 1Y T adj for mat and tax over 1+ nominal g."



*===============================================================================
*
*				V-bis. Generate implied debt path starting in 1960,1970,1980,1990,2000
*
*===============================================================================

*generate debt path conditional on realized growth and safe rate

foreach year in 1950 1960 1970 1980 1990 2000  {

	foreach var of varlist rog2  rlog2 rog_mat2 rog_adj2  {
		*short rate
		
		*path of r and g
		gen debt`var'`year' =`var'
		*initial debt to GDP
		replace debt`var'`year' = 100 if year == `year'
		replace debt`var'`year' = . if year < `year'
		* generate path conditional on initial debt to GDP
		replace debt`var'`year' = debt`var'`year'*debt`var'`year'[_n-1] if year > `year'
	

		}
}

pause on

* Figure 5
*rate with maturity adjustment (ojb, august 24, 2018)
*tsline debtrog_mat1950 debtrog_mat1960 debtrog_mat1970 debtrog_mat1980 debtrog_mat1990 debtrog_mat2000 if year > 1949, ytitle(index) lcolor(gs14 gs13 gs10 gs7 gs5 black) xlab(1950(10)2020) legend(off) graphregion(color(white)) bgcolor(white) 
tsline debtrog_mat21950 debtrog_mat21960 debtrog_mat21970 debtrog_mat21980 debtrog_mat21990 debtrog_mat22000 if year > 1949, ytitle("index") lcolor(green blue dkorange dkgreen red black) xlab(1950(10)2020) legend(off) graphregion(color(white)) bgcolor(white) 
* note("Source: Shiller, BEA, CBO, Treasury, FRED")
*graph export output\figures\dd.pdf, as(pdf) replace
*graph export output\figures\figure6bis_mat.eps, as(eps) replace
*writepsfrag output\figures\figure6bis_mat.eps using output\figures\allfigures.tex, append body(figure, caption(United States - Debt path implied by r adjusted for maturities and g) label(debtpath))

* Figure 6
* rate with maturity and tax adjustment (ojb, august 24, 2018)
*tsline debtrog_adj1950 debtrog_adj1960 debtrog_adj1970 debtrog_adj1980 debtrog_adj1990 debtrog_adj2000 if year > 1949, ytitle("% of GDP") lcolor(gs14 gs13 gs10 gs7 gs5 black) xlab(1950(10)2020) legend(off) graphregion(color(white)) bgcolor(white) 
* color coding change
tsline debtrog_adj21950 debtrog_adj21960 debtrog_adj21970 debtrog_adj21980 debtrog_adj21990 debtrog_adj22000 if year > 1949, ytitle("index") lcolor(green blue dkorange dkgreen red black) xlab(1950(10)2020) legend(off) graphregion(color(white)) bgcolor(white) 
*note("Source: Shiller, BEA, CBO, Treasury, FRED")
*graph export output\figures\figure6bis_adj.pdf, as(pdf) replace
*graph export output\figures\figure6bis_adj.eps, as(eps) replace
*writepsfrag output\figures\figure6bis_adj.eps using output\figures\allfigures.tex, append body(figure, caption(United States - Debt path implied by  r adjusted for maturities and tax and g) label(debtpath))

*save r_g_evidence_tax.dta, replace
*export excel using " r_g_evidence_tax", sheetreplace firstrow(variables)

